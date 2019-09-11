# coding=utf-8
# Created by Meteorix at 2019/7/30
import logging
import torch
from typing import List

import os
import json
from collections import defaultdict

import numpy as np
from service_streamer import ManagedModel
from pytorch_transformers import BertConfig, BertTokenizer, BertEcomCommentMultiPolarV4


from data_preprocess import convert_text
from utils_ecom_senti import (
    RawResult,
    SquadExample,
    convert_examples_to_features,
    convert_polar_examples_to_features, find_positions
)
from mtl_manual import write_predictions


logging.basicConfig(level=logging.ERROR)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class EcomSentiModel(object):
    def __init__(self, max_sent_len=256, model_path=None, target_device='cpu', subtype_dict='general'):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.model_path, 'vocab.txt'))
        config = BertConfig.from_pretrained(os.path.join(self.model_path, 'config.json'))
        print('model config is', config.__dict__)
        self.bert = BertEcomCommentMultiPolarV4.from_pretrained(os.path.join(self.model_path, 'pytorch_model.bin'),
                                                               config=config)
        self.bert.eval()
        self.target_device = target_device
        self.bert.to(self.target_device)
        self.label_map = {
            '0': 1,
            '1': 3,
            '2': 5
        }
        self.trans_subtype = json.load(open(os.path.join(os.path.dirname(__file__), 'SUBTYPE.json')))[subtype_dict]
        self.subtype_list = list(self.trans_subtype)
        self.doc_stride = 128
        self.max_query_length = 20
        self.max_sent_len = max_sent_len

    def _find_subtype(self, idx):
        return self.subtype_list[idx]

    def _calc_polar(self, text, opinions, inputs, seq_outputs):

        opinion_masks = []
        # generate opinion_mask
        for r in opinions:
            opinion_mask = [0] * self.max_sent_len
            op_start, op_end = find_positions(text, [r['opinionTerm']])
            if op_start == -2 or op_start > self.max_sent_len - 5:
                opinion_mask[0] = 1
            else:
                for i in range(op_start, op_end):
                    # 跳过 'CLS' '维' '度' 'SEP' 共4个TOKEN
                    opinion_mask[min(i + 4, self.max_sent_len - 1)] = 1
            opinion_masks.append(opinion_mask)

        all_opinion_mask = torch.tensor(opinion_masks, dtype=torch.float32).to(torch.device(self.target_device))

        with torch.no_grad():
            inputs.update({
                'opinion_mask': all_opinion_mask,
                'seq_embeddings': seq_outputs.repeat(len(opinion_masks), 1, 1),
                'attention_mask': inputs['attention_mask'].repeat(len(opinion_masks), 1)
            })

            outputs = self.bert(**inputs)
        return np.argmax(outputs.detach().cpu().numpy(), axis=1).tolist()

    def predict(self, batch: List[dict]) -> List[dict]:
        # extract features
        examples = []
        batch_outputs = []

        for idx, b in enumerate(batch):
            text = convert_text(b[:250])  # TODO 目前处理不了长句子
            if len(text) < 2:
                text = f'文本太短{text}'

            batch_outputs.append(dict(text=text, opinions=[]))
            example = SquadExample(
                qas_id=idx,
                question_text=list(self.trans_subtype.values())[0],
                doc_tokens=text,
                label=1
            )
            examples.append(example)
            idx += 1

        if len(examples) == 0:
            return batch_outputs

        try:
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=self.tokenizer,
                                                    max_seq_length=self.max_sent_len,
                                                    doc_stride=self.doc_stride,
                                                    max_query_length=self.max_query_length,
                                                    is_training=False,
                                                    label_list=list(self.label_map.values()),
                                                    trans_subtype=self.trans_subtype)

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(torch.device(self.target_device))
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(torch.device(self.target_device))
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(torch.device(self.target_device))
            all_question_ids = torch.tensor([-1] * len(features), dtype=torch.long).to(torch.device(self.target_device))

            # model inference
            with torch.no_grad():
                inputs = {
                    'input_ids': all_input_ids,
                    'attention_mask': all_input_mask,
                    'token_type_ids': all_segment_ids,
                    'question_ids': all_question_ids,
                }
                outputs = self.bert(**inputs)
                # outputs format: start_logits_list, end_logits_list, op_start_logits_list, op_end_logits_list,
                # num_subtype * [batch * seq]
            # store raw result of examples by subtype
            all_results = defaultdict(list)
            all_op_results = defaultdict(list)  # batch * len(subtype)

            for h in range(len(batch)):
                for i in range(len(self.trans_subtype)):
                    result = RawResult(unique_id=features[h].unique_id,
                                       start_logits=to_list(outputs[0][i][h]),
                                       end_logits=to_list(outputs[1][i][h]))
                    op_result = RawResult(unique_id=features[h].unique_id,
                                          start_logits=to_list(outputs[2][i][h]),
                                          end_logits=to_list(outputs[3][i][h]))

                    all_results[i].append(result)
                    all_op_results[i].append(op_result)

            # postprocess for final opinions
            kwargs = dict(n_best_size=5,
                          max_answer_length=20,
                          do_lower_case=True,
                          verbose_logging=False,
                          version_2_with_negative=True,
                          null_score_diff_threshold=0)

            idx = 0
            # generate predictions of all examples in a batch by subtype
            for r, opr in (zip(all_results.values(), all_op_results.values())):

                asp, nbest_asp = write_predictions(examples, features, r, **kwargs)  # 特征词
                op, nbest_op = write_predictions(examples, features, opr, **kwargs)  # 情感词
                for h in range(len(batch)):
                    asp_ret = asp[h].replace(" ", "")
                    op_ret = op[h].replace(" ", "")
                    if len(asp_ret) > 1:
                        batch_outputs[h]['opinions'].append({
                            'aspectSubtype': self._find_subtype(idx),
                            'aspectTerm': asp_ret,
                            'opinionTerm': op_ret,
                        })
                idx += 1
                if idx > len(self.trans_subtype) - 1:
                    break  # 不接受多余的subtype结果

            # 为每条text预测其情感
            for idx, example in enumerate(examples):
                if len(batch_outputs[idx]['opinions']) > 0:
                    cache_inputs = {
                        'input_ids': all_input_ids[idx].view(1, -1),
                        'attention_mask': all_input_mask[idx].view(1, -1),
                        'token_type_ids': all_segment_ids[idx].view(1, -1),
                        'question_ids': all_question_ids[idx].view(1, -1),
                    }

                    polars = self._calc_polar(example.doc_tokens, batch_outputs[idx]['opinions'], cache_inputs,
                                              outputs[4][idx].view(1, *outputs[4][idx].size()))
                    for op, polar in zip(batch_outputs[idx]['opinions'], polars):
                        if len(op['opinionTerm']) < 1:
                            op['polarity'] = 3
                        else:
                            op['polarity'] = self.label_map.get(str(polar), 3)
        except Exception as e:
            batch_outputs[0].update({'err': e})
        return batch_outputs


class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = EcomSentiModel()

    def predict(self, batch):
        return self.model.predict(batch)


if __name__ == "__main__":
    batch = ["还是网上买东西实惠，宝贝收到了，我超喜欢，呵呵，服务态度也不错，很有心的店家，以后常光顾。",
             "欧莱雅的洗面奶一直在用，挺好的，干爽不紧绷，挺好的，京东的服务也很不错，会一直回购"]

    model = EcomSentiModel()
    outputs = model.predict(batch)
    print(outputs)
