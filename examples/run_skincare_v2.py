# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertEcomCommentMultiPolar, BertEcomCommentMultiV2,
                                  BertEcomCommentMultiPolarV4,
                                  BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule


from utils_glue import processors, compute_metrics
from utils_skincare_v2 import (convert_examples_to_features,
                                     RawResult, write_predictions, read_ecom_examples, convert_polar_examples_to_features,
                                     RawResultExtended, write_predictions_extended, acc_and_f1, TRANS_SUBTYPE)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertEcomCommentMultiPolar, BertTokenizer),
    'multi_v2': (BertConfig, BertEcomCommentMultiV2, BertTokenizer),
    'multi_v3': (BertConfig, BertEcomCommentMultiPolarV4, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def fetch_batch(dataloaderiters, loader_counter):
    valid_key = [k for k, v in loader_counter.items() if v > 0]
    pick_key = random.choice(valid_key)
    loader_counter[pick_key] -= 1

    return next(dataloaderiters[pick_key]), pick_key


def train(args, train_dataset_dict, model, tokenizer, num_tasks):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloaders = {}
    sum_batch = 0
    batch_counter = {}
    for idx, train_dataset in train_dataset_dict.items():
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        train_dataloaders[idx] = train_dataloader
        batch_counter[idx] = len(train_dataloader)
        sum_batch += len(train_dataloader)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (sum_batch // args.gradient_accumulation_steps) + 1
    else:
        t_total = sum_batch // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", (sum_batch * args.train_batch_size))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # 定义负面极性的损失权重
    polar_weights = [1] * args.max_seq_length
    polar_weights[7] = 2
    polar_start_weights = torch.FloatTensor(polar_weights).to(args.device)
    polar_weights = [1] * args.max_seq_length
    polar_weights[8] = 2
    polar_end_weights = torch.FloatTensor(polar_weights).to(args.device)

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    loss_avg = RunningAverage()
    for _ in train_iterator:

        epoch_iterator = tqdm(range(sum_batch), desc="Iteration", disable=args.local_rank not in [-1, 0])
        load_counter = batch_counter.copy()
        print('load counter is', load_counter)
        train_dataloader_iters = {}
        for k, v in train_dataloaders.items():
            train_dataloader_iters[k] = iter(v)

        for step in epoch_iterator:
            # 从不同的数据集中交替取batch 直到某个数据集完结
            batch, pick_key = fetch_batch(train_dataloader_iters, load_counter)
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # 此处需要根据极性/特征 区分batch
            if pick_key < num_tasks:
                # 特征/情感词发现的训练batch
                inputs = {'input_ids':       batch[0],
                          'attention_mask':  batch[1],
                          'token_type_ids':  None if args.model_type == 'xlm' else batch[2],
                          'start_positions': batch[3],
                          'end_positions':   batch[4],
                          'op_start_positions': batch[5],
                          'op_end_positions': batch[6],
                          # 'polar_start_positions': batch[7],
                          # 'polar_end_positions': batch[8],
                          'question_ids': batch[9],
                          'labels': batch[10],
                          'kw_ids': batch[11]}
                          # 'polar_start_weights': polar_start_weights,
                          # 'polar_end_weights': polar_end_weights}
            else:
                # 情感极性的训练batch
                inputs = {'input_ids':       batch[0],
                          'attention_mask':  batch[1],
                          'token_type_ids':  None if args.model_type == 'xlm' else batch[2],
                          'labels': batch[3],
                          'question_ids': batch[4],
                          'kw_ids': batch[5],
                          'opinion_mask': batch[6]
                          }
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':    batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            loss_avg.update(loss.item())
            epoch_iterator.set_postfix(loss='{:05.3f}'.format(loss_avg()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.export_scalars_to_json(os.path.join(args.output_dir, 'all_scalars.json'))
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate_polar(args, model, tokenizer, prefix="",label_list=["1", "3", "5"], num_tasks=6):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ('ecom', )
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_polar_examples(args, eval_task, tokenizer, evaluate=True, label_list=label_list, num_tasks=num_tasks)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3],
                          'question_ids':   batch[4],
                          'kw_ids': batch[5],
                          'opinion_mask': batch[6]}
                outputs = model(**inputs)

                logits = outputs[1]
                # print(logits)

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # print('pred is', preds)
        preds = np.argmax(preds, axis=1)

        result = acc_and_f1(preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        raw_eval_file = os.path.join(eval_output_dir, 'eval_raw.csv')
        with open(raw_eval_file, 'w') as writer:
            logger.info("***** raw eval results {} *****".format(prefix))
            for m, n in zip(preds, out_label_ids):
                writer.write(f"{m},{n} \n")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results




def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args,
                                                          tokenizer,
                                                          evaluate=True,
                                                          output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    op_all_results = []
    polar_all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2],  # XLM don't use segment_ids
                      'question_ids':  batch[4],
                      'kw_ids': batch[5]
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:

                result = RawResult(unique_id = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits = to_list(outputs[1][i]))

                op_result = RawResult(unique_id = unique_id,
                                   start_logits = to_list(outputs[2][i]),
                                   end_logits = to_list(outputs[3][i]))
                # polar_result = RawResult(unique_id=unique_id,
                #                       start_logits=to_list(outputs[4][i]),
                #                       end_logits=to_list(outputs[5][i]))
                # polar_result = to_list(outputs[4][i])

            all_results.append(result)
            op_all_results.append(op_result)
            # polar_all_results.append(polar_result)

    # Compute predictions
    output_polar_file = os.path.join(args.output_dir, "polarity_preds_{}.csv".format(prefix))
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    op_output_prediction_file = os.path.join(args.output_dir, "op_predictions_{}.json".format(prefix))
    op_output_nbest_file = os.path.join(args.output_dir, "op_nbest_predictions_{}.json".format(prefix))
    polar_output_prediction_file = os.path.join(args.output_dir, "polar_predictions_{}.json".format(prefix))
    polar_output_nbest_file = os.path.join(args.output_dir, "polar_nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
        op_output_null_log_odds_file = os.path.join(args.output_dir, "op_null_odds_{}.json".format(prefix))
        polar_output_null_log_odds_file = os.path.join(args.output_dir, "polar_null_odds_{}.json".format(prefix))

    else:
        output_null_log_odds_file = None
        op_output_null_log_odds_file = None
        polar_output_null_log_odds_file = None


    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        # write_polar_predictions(examples, all_polar_results, output_polar_file, label_list=label_list)

        write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
        write_predictions(examples, features, op_all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, op_output_prediction_file,
                        op_output_nbest_file, op_output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
        # write_predictions(examples, features, polar_all_results, args.n_best_size,
        #                 args.max_answer_length, args.do_lower_case, polar_output_prediction_file,
        #                 polar_output_nbest_file, polar_output_null_log_odds_file, args.verbose_logging,
        #                 args.version_2_with_negative, args.null_score_diff_threshold)

    # # Evaluate with the official SQuAD script
    # evaluate_options = EVAL_OPTS(data_file=args.predict_file,
    #                              pred_file=output_prediction_file,
    #                              na_prob_file=output_null_log_odds_file)
    # results = evaluate_on_squad(evaluate_options)
    # return results


def load_and_cache_polar_examples(args, task, tokenizer, evaluate=False, label_list=None, num_tasks=None):
    # 生成subtype id列表
    subtype_ids = {}
    for k, m in enumerate(TRANS_SUBTYPE.values()):
        m = tokenizer.tokenize(m)
        subtype_ids[k] = tokenizer.convert_tokens_to_ids(m)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    processor = processors[task]()
    output_mode = 'classification'
    dataset_dict = {}
    dirlist = os.listdir(args.multi_subtype_dir) if not evaluate else [args.ecom_subtype]
    for dirname in dirlist:
        subtype = dirname.replace('.', '/').replace('_', ' ')
        if subtype not in TRANS_SUBTYPE:
            continue
        # Load data features from cache or dataset file
        input_dir = os.path.join(args.multi_subtype_dir, dirname)
        cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", input_dir)
            examples = processor.get_dev_examples(input_dir) if evaluate else processor.get_train_examples(input_dir)
            features = convert_polar_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                num_tasks,
                cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_question_ids = torch.tensor([f.question_id for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_subtype_ids = torch.tensor([subtype_ids[f.question_id % len(TRANS_SUBTYPE)] for f in features], dtype=torch.long)
        all_opinion_mask = torch.tensor([f.opinion_mask for f in features], dtype=torch.float32)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_question_ids, all_subtype_ids, all_opinion_mask)
        dataset_dict[features[0].question_id] = dataset
    if evaluate:
        return dataset
    else:
        return dataset_dict


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # 生成subtype id列表
    subtype_ids = {}
    for k, m in enumerate(TRANS_SUBTYPE.values()):
        m = tokenizer.tokenize(m)
        subtype_ids[k] = tokenizer.convert_tokens_to_ids(m)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    dataset_dict = {}
    dirlist = os.listdir(args.multi_subtype_dir) if not evaluate else [args.ecom_subtype]
    for dirname in dirlist:
        subtype = dirname.replace('.','/').replace('_',' ')
        if subtype not in TRANS_SUBTYPE:
            continue
        filename = 'dev.json' if evaluate else 'train.json'
        input_file = os.path.join(args.multi_subtype_dir, dirname, filename)

        cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length)))

        if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", input_file)
            examples = read_ecom_examples(input_file=input_file,
                                          is_training=not evaluate,
                                          subtype=dirname)
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=args.max_seq_length,
                                                    doc_stride=args.doc_stride,
                                                    max_query_length=args.max_query_length,
                                                    is_training=not evaluate,
                                                    label_list=[1, 3 ,5])
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_question_ids = torch.tensor([f.question_id for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_subtype_ids = torch.tensor([subtype_ids[f.question_id % len(TRANS_SUBTYPE)] for f in features], dtype=torch.long)

        if evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_question_ids, all_subtype_ids, all_cls_index, all_p_mask)
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_op_start_positions = torch.tensor([f.op_start_position for f in features], dtype=torch.long)
            all_op_end_positions = torch.tensor([f.op_end_position for f in features], dtype=torch.long)
            all_polar_start_positions = torch.tensor([f.polar_start_position for f in features], dtype=torch.long)
            all_polar_end_positions = torch.tensor([f.polar_end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_op_start_positions, all_op_end_positions,
                                    all_polar_start_positions, all_polar_end_positions,
                                    all_question_ids, all_labels, all_subtype_ids, all_cls_index, all_p_mask)
        dataset_dict[features[0].question_id] = dataset
    if output_examples:
        return dataset, examples, features
    return dataset_dict


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--ecom_subtype', type=str, required=True, help="which ecom subtype your model" )
    parser.add_argument('--multi_subtype_dir', type=str, default="",
                        help="data dir, set to train multi-subtype simultaneously." )
    parser.add_argument("--polar", action='store_true',
                        help="eval polar")
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    label_list = ["1", "3", "5"]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset_dict = {}
        train_dataset_dict = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        train_dataset_dict.update(load_and_cache_polar_examples(args, 'ecom', tokenizer, evaluate=False,
                                                                label_list=label_list, num_tasks=config.num_tasks))
        global_step, tr_loss = train(args, train_dataset_dict, model, tokenizer, config.num_tasks)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Save the trained model and the tokenizer
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            if args.polar:
                result = evaluate_polar(args, model, tokenizer, prefix=global_step, label_list=label_list, num_tasks=config.num_tasks)
                result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
                results.update(result)
            else:
                result = evaluate(args, model, tokenizer, prefix=global_step)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
