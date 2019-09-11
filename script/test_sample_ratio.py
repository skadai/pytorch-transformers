# -*- coding: utf-8 -*-

# @File    : test_sample_ratio.py
# @Date    : 2019-09-05
# @Author  : skym

import os

sample_ratios = [1]
# sample_ratios = [0.8]

subtypes = ['Smell', 'Fat_Granule', 'Irritation', 'Whitening', 'Greasy', 'Moisturization']
subdict = 'skincare'
task_name = 'skincare_patch'

data_dir = f'/data/projects/bert_pytorch/{task_name}_out/'


for sample_ratio in sample_ratios:
    raio_str = str(sample_ratio).replace('.', '')
    # dirname = f'{subdict}_{raio_str}'
    dirname = 'all_in_one_ground/checkpoint-4200'
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
