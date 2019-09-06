# coding=utf-8
# Created by Meteorix at 2019/7/30
import logging
import torch
from typing import List

import os
from collections import defaultdict

import numpy as np
from pytorch_transformers import *


from data_preprocess import convert_text
from utils_skincare import TRANS_SUBTYPE
from utils_skincare_v2 import (
    RawResult,
    SquadExample,
    convert_examples_to_features,
    convert_polar_examples_to_features, find_positions
)
from mtl_manual import write_predictions
from service_streamer import ManagedModel


logging.basicConfig(level=logging.ERROR)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


subtype_list = list(TRANS_SUBTYPE.keys())


def find_subtype(idx):
    return subtype_list[idx]


class EcomSentiModel(object):
    def __init__(self, max_sent_len=256, model_path=None):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.model_path, 'vocab.txt'))
        self.config = BertConfig.from_pretrained(os.path.join(self.model_path, 'config.json'))
        print(config.__dict__)
        self.bert = BertEcomCommentMultiPolarV4.from_pretrained(os.path.join(self.model_path, 'pytorch_model.bin'),
                                                                config=self.config)
        self.bert.eval()
        self.target_device = 'cpu'
        self.bert.to(self.target_device)
        self.label_map = {
            '0': 1,
            '1': 3,
            '2': 5
        }
        self.doc_stride = 128
        self.max_query_length = 20
        self.max_sent_len = max_sent_len

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
        # extrac features
        examples = []
        batch_outputs = []
        for idx, b in enumerate(batch):
            text = convert_text(b[:252])  # TODO 目前处理不了长句子
            batch_outputs.append(dict(text=text, opinions=[]))
            example = SquadExample(
                qas_id=idx,
                question_text=list(TRANS_SUBTYPE.values())[0],
                doc_tokens=text,
                label=1
            )
            examples.append(example)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                max_seq_length=self.max_sent_len,
                                                doc_stride=self.doc_stride,
                                                max_query_length=self.max_query_length,
                                                is_training=False,
                                                label_list=list(self.label_map.values()),
                                                trans_subtype=TRANS_SUBTYPE)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(
            torch.device(self.target_device))
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(
            torch.device(self.target_device))
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(
            torch.device(self.target_device))
        all_question_ids = torch.tensor([-1] * len(features), dtype=torch.long).to(
            torch.device(self.target_device))

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
            for i in range(len(TRANS_SUBTYPE)):
                result = RawResult(unique_id=features[h].unique_id,
                                   start_logits=to_list(outputs[0][i][h]),
                                   end_logits=to_list(outputs[1][i][h]))
                op_result = RawResult(unique_id=features[h].unique_id,
                                      start_logits=to_list(outputs[2][i][h]),
                                      end_logits=to_list(outputs[3][i][h]))

                all_results[i].append(result)
                all_op_results[i].append(op_result)

        # postprocess for final opinions
        ret = {
            'text': text,
            'opinions': []
        }
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
                        'aspectSubtype': find_subtype(idx),
                        'aspectTerm': asp_ret,
                        'opinionTerm': op_ret,
                    })
            idx += 1
            if idx > len(TRANS_SUBTYPE) - 1: break  # 不接受多余的subtype结果

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
        return batch_outputs


class TextInfillingModel(object):
    def __init__(self, max_sent_len=16):
        self.model_path = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertForMaskedLM.from_pretrained(self.model_path)
        self.bert.eval()
        self.bert.to("cuda")
        self.max_sent_len = max_sent_len

    def predict(self, batch: List[str]) -> List[str]:
        """predict masked word"""
        batch_inputs = []
        masked_indexes = []

        for text in batch:
            tokenized_text = self.tokenizer.tokenize(text)
            if len(tokenized_text) > self.max_sent_len - 2:
                tokenized_text = tokenized_text[: self.max_sent_len - 2]
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            tokenized_text += ['[PAD]'] * (self.max_sent_len - len(tokenized_text))
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            batch_inputs.append(indexed_tokens)
            masked_indexes.append(tokenized_text.index('[MASK]'))
        tokens_tensor = torch.tensor(batch_inputs).to("cuda")

        with torch.no_grad():
            # prediction_scores: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            prediction_scores = self.bert(tokens_tensor)[0]

        batch_outputs = []
        for i in range(len(batch_inputs)):
            predicted_index = torch.argmax(prediction_scores[i, masked_indexes[i]]).item()
            predicted_token = self.tokenizer.convert_ids_to_tokens(predicted_index)
            batch_outputs.append(predicted_token)

        return batch_outputs


class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = TextInfillingModel()

    def predict(self, batch):
        return self.model.predict(batch)


if __name__ == "__main__":
    batch = ["twinkle twinkle [MASK] star.",
             "Happy birthday to [MASK].",
             'the answer to life, the [MASK], and everything.']
    model = TextInfillingModel()
    outputs = model.predict(batch)
    print(outputs)
