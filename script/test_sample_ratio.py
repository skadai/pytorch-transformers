# -*- coding: utf-8 -*-

# @File    : test_sample_ratio.py
# @Date    : 2019-09-05
# @Author  : skym

import os

sample_ratios = [0.3, 0.35, 0.38, 0.42, 0.46, 0.55, 0.58]

subtype = 'Whitening'
subdict = subtype.lower()
subdict = 'skincare'
task_name = 'skincare_whiten'

data_dir = f'/data/projects/bert_pytorch/{task_name}_out/'


for sample_ratio in sample_ratios:
    raio_str = str(sample_ratio).replace('.','')
    dirname =f'{subdict}_{raio_str}'
    if not os.path.exists(os.path.join(data_dir, dirname)):
        os.makedirs(os.path.join(data_dir, dirname))
    command = f'./run_skincare_v2.sh {dirname} {task_name}  {subdict}  {sample_ratio}'
    print(command)
    os.system(command)

    command = f'./run_skincare_v2_eval.sh  {subtype}   {task_name}  {dirname} {subdict}'
    print(command)
    os.system(command)

    command = f'./run_skincare_v2_polar_eval.sh  {subtype}   {task_name}  {dirname} {subdict}'
    print(command)
    os.system(command)
