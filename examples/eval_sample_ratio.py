# -*- coding: utf-8 -*-

# @File    : eval_sample_ratio.py.py
# @Date    : 2019-09-05
# @Author  : skym



import os
import pandas as pd
import json

import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_SERVER_URL = 'http://127.0.0.1:9001' # mlflow server address
mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
client = MlflowClient()
HOST_USER = 'ymai'


polarity_map = {
    0: 1,
    1: 3,
    2: 5
}


def eval_aspect_result(runs_name, experiment_name=None,
                       ckpt_suffix=None, data_path=None,
                       write_mlflow=False, trans_subtype=None, subdict=None):
    if ckpt_suffix is not None:
        run_name = f'{runs_name}-{ckpt_suffix}'  # 有时需要加上checkpoint后缀
    else:
        run_name = f'{runs_name}'

    ret = {}
    ret_op = {}

    dirlist = os.listdir(data_path)
    for filename in dirlist:
        subtype = filename.replace('.', '/').replace('_', " ")
        if subtype in trans_subtype:
            try:
                print(f'try to assess {subtype}')
                command = f"python evaluate_ecom_asop.py -st aspect -sd {subdict} -rn {run_name} -tn {experiment_name} {filename} |tail -n 15 > tmp.json "
                os.system(command)
                ret[subtype] = json.load(open('tmp.json'))
                command = f"python evaluate_ecom_asop.py -st op -sd {subdict} -rn {run_name} -tn {experiment_name} {filename} |tail -n 15 > tmp.json "
                os.system(command)
                ret_op[subtype] = json.load(open('tmp.json'))
            except Exception as e:
                print('err', subtype, e)

    data = []
    for k in ret.keys():
        data.append(
            (k,
             ret[k]['exact'], ret[k]['f1'], ret[k]['total'],
             ret[k]['NoAns_exact'], ret[k]['NoAns_f1'], ret[k]['NoAns_total'],
             ret_op[k]['exact'], ret_op[k]['f1'], ret_op[k]['total'],
             ret_op[k]['NoAns_exact'], ret_op[k]['NoAns_f1'], ret_op[k]['NoAns_total'],
             )
        )

    aspect_columns = ['exact', 'f1', 'total', 'NoAns_exact', 'NoAns_f1', 'NoAns_total']
    opinions_columns = list(map(lambda x: 'op_' + x, aspect_columns))

    df_metric = pd.DataFrame(data, columns=['subtype'] + aspect_columns + opinions_columns)

    df_metric.sort_values('f1', ascending=True, inplace=True)
    if write_mlflow:
        write_to_mlflow(df_metric, experiment_name, runs_name)

    return df_metric


def eval_polar_result(runs_name, experiment_name=None, write_mlflow=False, trans_subtype=None):
    result_dir = f'/data/projects/bert_pytorch/{experiment_name}_out/{runs_name}'
    polar_metric= []

    for dirname in os.listdir(result_dir):
        subtype = dirname.replace('.', '/').replace('_'," ")
        if subtype not in trans_subtype:
            continue
        print(subtype)
        json_path = os.path.join(result_dir, dirname, 'eval_results.txt')
        with open(json_path, 'r') as cc:
            lines = cc.read().splitlines()
            f1, prec, recall, support = [json.loads(item.split('=')[-1]) for item in lines[1:]]
            for idx, m in enumerate(zip(f1, prec, recall, support)):
                polar_metric.append((
                    m[1], m[2], m[0], m[3], subtype, polarity_map[idx]
                ))
    df_polar_m = pd.DataFrame(polar_metric, columns=['precision', 'recall', 'f1', 'sample_num', 'subtype', 'polarity'])
    df_polar_m['weight_f1'] = df_polar_m.apply(lambda x:x['sample_num']*x['f1'],axis=1)

    f1_pos = df_polar_m[df_polar_m.polarity==5]['weight_f1'].sum()/df_polar_m[df_polar_m.polarity==5]['sample_num'].sum()
    f1_neg = df_polar_m[df_polar_m.polarity==1]['weight_f1'].sum()/df_polar_m[df_polar_m.polarity==1]['sample_num'].sum()

    if write_mlflow:
        write_polar_to_mlflow(f1_pos, f1_neg, experiment_name, runs_name)
    return f1_pos, f1_neg


def write_to_mlflow(df_metric, experiment_name, runs_name):
    experiments = client.list_experiments()  # returns a list of mlflow.entities.Experiment
    experiment = list(filter(lambda x: x.name == experiment_name, experiments))
    target_run = client.search_runs(experiment[0].experiment_id, f"tag.mlflow.runName='{runs_name}' and tag.mlflow.user='{HOST_USER}'")
    if len(target_run) > 0:
        for _, line in df_metric.iterrows():
            sub = line['subtype']
            f1 = line['f1']
            op_f1 = line['op_f1']
            client.log_metric(target_run[0].info.run_id, f'{sub}_f1', f1)
            client.log_metric(target_run[0].info.run_id, f'{sub}_opf1', op_f1)
            print('write metric succeed...')


def write_polar_to_mlflow(f1_pos, f1_neg, experiment_name, runs_name):
    experiments = client.list_experiments()  # returns a list of mlflow.entities.Experiment
    experiment = list(filter(lambda x: x.name == experiment_name, experiments))
    target_run = client.search_runs(experiment[0].experiment_id, f"tag.mlflow.runName='{runs_name}' tag.mlflow.user='{HOST_USER}'")

    if len(target_run) > 0:
        client.log_metric(target_run[0].info.run_id, 'pos_polar_aver', f1_pos)
        client.log_metric(target_run[0].info.run_id, 'neg_polar_aver', f1_neg)
        print('write metric succeed...')


if __name__ == '__main__':
    SUBTYPE_DICT = json.load(open(os.path.join(os.path.dirname(__file__), 'SUBTYPE.json'), 'r'))

    experiment_name = 'skincare_patch'
    data_path = f'/data/projects/bert_pytorch/{experiment_name}'

    subdict = 'skincare'
    trans_subtype = SUBTYPE_DICT[subdict]

    sample_ratios = [1]

    for sample_ratio in sample_ratios:

        raio_str = str(sample_ratio).replace('.', '')
        runs_name = f'{subdict}_{raio_str}'
        runs_name = 'all_in_one_ground'
        print('write eval result:', runs_name)
        r1 = eval_aspect_result(runs_name, data_path=data_path, experiment_name=experiment_name, write_mlflow=True, trans_subtype=trans_subtype, subdict=subdict)
        r2 = eval_polar_result(runs_name, experiment_name=experiment_name, write_mlflow=True, trans_subtype=trans_subtype)

