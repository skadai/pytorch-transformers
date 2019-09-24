# -*- coding: utf-8 -*-

# @File    : test_sample_ratio.py
# @Date    : 2019-09-05
# @Author  : skym
# 用来测试训练量对模型指标的影响

import os

sample_ratios = [1]
# sample_ratios = [0.8]


subdict = 'chocolate'
task_name = 'chocolate'

data_dir = f'/data/projects/bert_pytorch/{task_name}_out/'
corpus_dir = f'/data/projects/bert_pytorch/{task_name}'

subtypes = [ dirname for dirname in os.listdir(corpus_dir) if os.path.isdir(os.path.join(corpus_dir, dirname))]


for sample_ratio in sample_ratios:
    raio_str = str(sample_ratio).replace('.', '')
    # dirname = f'{subdict}_{raio_str}'
    dirname = 'all_in_one/checkpoint-4800'
    if not os.path.exists(os.path.join(data_dir, dirname)):
        os.makedirs(os.path.join(data_dir, dirname))
    # command = f'./run_ecom_senti.sh {dirname} {task_name}  {subdict}  {sample_ratio}'
    # print(command)
    # os.system(command)

    for subtype in subtypes:
        command = f'./run_ecom_senti_eval.sh  {subtype}   {task_name}  {dirname} {subdict}'
        print(command)
        os.system(command)

        command = f'./run_ecom_senti_polar_eval.sh  {subtype}   {task_name}  {dirname} {subdict}'
        print(command)
        os.system(command)
