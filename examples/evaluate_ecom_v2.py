# -*- coding: utf-8 -*-

# @File    : evaluate-v2.py.py
# @Date    : 2019-04-26
# @Author  : skym


"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import time
import collections
import json
import numpy as np
import os
import re
import string
import sys


from utils_squad import read_ecom_examples, read_multi_examples, TRANS_SUBTYPE
from utils_squad_op import read_ecom_examples as read_ecom_examples_op
from utils_squad_op import read_multi_examples as read_multi_examples_op


OPTS = None


FUNC_LOADER = {
    'op_multi': read_multi_examples_op,
    'op_single': read_ecom_examples_op,
    'multi': read_multi_examples,
    'single': read_ecom_examples
}


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
    parser.add_argument('pred_dir', metavar='pred_dir', help='Model predictions.')
    parser.add_argument('--data-filename', '-d', metavar='data_filename', help='Input data JSON file.')

    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                        help='Model estimates of probability of no answer.')
    parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                        help='Predict "" if no-answer probability exceeds this (default = 1.0).')
    parser.add_argument('--out-image-dir', '-p', metavar='out_images', default=None,
                        help='Save precision-recall curves to directory.')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--multi', '-m', action='store_true')
    parser.add_argument('--with_opinion', '-op', action='store_true', help="opinion term f1 calculation")
    parser.add_argument('--subtype_en', '-s', metavar='subtype_en', help='subtype_en.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_qid_to_has_ans(dataset, with_opinion=False, subtype=None):
    qid_to_has_ans = {}
    for example in dataset:
        if example.question_text == subtype:
            qid_to_has_ans[example.qas_id] =not example.is_op_impossible if with_opinion else not example.is_impossible
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s)


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    """

    此函数十分关键, 因为需要计算empty情况下的F1

    :param a_gold:
    :param a_pred:
    :return:
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        ret = int(gold_toks == pred_toks)
        # print('f1-no-ans', ret, f'<prediction:{a_pred}>', f'<truth:{a_gold}>')
        return ret
    if num_same == 0:
        # print('f1', 0, f'<prediction:{a_pred}>', f'<truth:{a_gold}>')
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('f1', f1, f'<prediction:{a_pred}>', f'<truth:{a_gold}>')
    return f1


def get_raw_scores(dataset, preds, with_opinion=False, subtype=None):
    exact_scores = {}
    f1_scores = {}
    raw_answers = {}
    for qa in dataset:
        qid = str(qa.qas_id)
        # print(qa.doc_tokens.replace(' ',''))
        if qa.question_text != subtype:
            continue
        if not with_opinion:
            gold_answers = [qa.orig_answer_text if normalize_answer(qa.orig_answer_text) else '']

        else:
            gold_answers = [qa.op_answer_text if normalize_answer(qa.op_answer_text) else '']
        if qid not in preds:
            print('Missing prediction for %s' % qid)
            continue
        a_pred = preds[qid].replace(' ', '')
        raw_answers[qid] = {
            'text': qa.doc_tokens.replace(" ",""),
            'label': gold_answers[0],
            'pred': a_pred
        }
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)

    return exact_scores, f1_scores, raw_answers


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    """

    :param scores: 得分
    :param na_probs: null得分 - best no-null
    :param qid_to_has_ans: 有答案的那些question id
    :param na_prob_thresh: null_score需要超出的部分
    :return:
    """
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            # 对于预测null的, 如果此时预测出来有ans那么分数直接清零0
            # 感觉这里不必要???
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def plot_pr_curve(precisions, recalls, out_image, title):
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0
    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / float(i+1)
        cur_r = true_pos / float(num_true_pos)
        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
            # i.e., if we can put a threshold after this point
            avg_prec += cur_p * (cur_r - recalls[-1])
            precisions.append(cur_p)
            recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {'ap': 100.0 * avg_prec}


def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs,
                                  qid_to_has_ans, out_image_dir):
    if out_image_dir and not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    pr_exact = make_precision_recall_eval(
        exact_raw, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_exact.png'),
        title='Precision-Recall curve for Exact Match score')
    pr_f1 = make_precision_recall_eval(
        f1_raw, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_f1.png'),
        title='Precision-Recall curve for F1 score')
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    pr_oracle = make_precision_recall_eval(
        oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
        out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
        title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')
    merge_eval(main_eval, pr_exact, 'pr_exact')
    merge_eval(main_eval, pr_f1, 'pr_f1')
    merge_eval(main_eval, pr_oracle, 'pr_oracle')


def histogram_na_prob(na_probs, qid_list, image_dir, name):
    if not qid_list:
        return
    x = [na_probs[k] for k in qid_list]
    weights = np.ones_like(x) / float(len(x))
    plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
    plt.xlabel('Model probability of no-answer')
    plt.ylabel('Proportion of dataset')
    plt.title('Histogram of no-answer probability: %s' % name)
    plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
    plt.clf()


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """
    选择一个thresh 只会对null的score产生影响, 同意吗?
    :param preds:
    :param scores:
    :param na_probs:
    :param qid_to_has_ans:
    :return:
    """
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
#     print('initial_score', best_score, cur_score)
#     time.sleep(2)
    # 按照null可能性小到大的顺序
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                # 如果preds有内容, 扣分
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
#             print('更新cur score', best_score, cur_score)
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):

    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def main():
    data_filename = OPTS.data_filename if OPTS.data_filename else OPTS.pred_dir
    subtype_en = OPTS.subtype_en if OPTS.subtype_en else data_filename
    subtype_cn = TRANS_SUBTYPE[subtype_en.replace('.', '/').replace('_',' ')]
    data_dir = f'/data/projects/bert_pytorch/ecom_aspect_bak'

    prefix = "op_" if OPTS.with_opinion else ""
    pred_file_path = f'{data_dir}_out/{OPTS.pred_dir}/{prefix}predictions_.json'

    data_file_path = f'{data_dir}/{OPTS.pred_dir}/dev.json'
    na_prob_file_path = f'{data_dir}_out/{OPTS.pred_dir}/{prefix}null_odds_.json'
    if OPTS.multi:
        flag = f'{prefix}multi'
        dataset = FUNC_LOADER[flag](data_dir, is_training=True, filename='dev.json')
    else:
        flag = f'{prefix}single'
        dataset = FUNC_LOADER[flag](data_file_path, is_training=True, subtype=subtype_en)

    with open(pred_file_path) as f:
        preds = json.load(f)
    if os.path.isfile(na_prob_file_path):
        with open(na_prob_file_path) as f:
            na_probs = json.load(f)
    else:
        na_probs = {k: 0.0 for k in preds}
    qid_to_has_ans = make_qid_to_has_ans(dataset, prefix=='op_', subtype=subtype_cn)  # maps qid to True/False

    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

    exact_raw, f1_raw, raw_answers = get_raw_scores(dataset, preds, prefix == 'op_', subtype=subtype_cn)  # 返回的都是字典


    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          OPTS.na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       OPTS.na_prob_thresh)

    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    if OPTS.na_prob_file or os.path.isfile(na_prob_file_path):
        find_all_best_thresh(out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
    if OPTS.na_prob_file and OPTS.out_image_dir:
        run_precision_recall_analysis(out_eval, exact_raw, f1_raw, na_probs,
                                      qid_to_has_ans, OPTS.out_image_dir)
        histogram_na_prob(na_probs, has_ans_qids, OPTS.out_image_dir, 'hasAns')
        histogram_na_prob(na_probs, no_ans_qids, OPTS.out_image_dir, 'noAns')

    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(out_eval, f)
            f.write('\n')
            json.dump(raw_answers, f)

    else:
        print(json.dumps(out_eval, indent=2))


if __name__ == '__main__':
    OPTS = parse_args()
#     print('OPTS', OPTS)
    if OPTS.out_image_dir:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    main()
