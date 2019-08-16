# -*- coding: utf-8 -*-

# @File    : train_ecom_op.py
# @Date    : 2019-08-16
# @Author  : skym
# @Desp: 遍历subtype训练阅读理解模型同时识别 特征词 情感词

import os

TRANS_SUBTYPE = {'Brand Equity': '品牌资产',
 'Loyalty': '品牌忠诚度',
 'New User': '品牌新用户',
 'WOM': '品牌口碑',
 'Fake Concern': '假货',
 'Inventory': '库存',
 'Expiration Date': '保质期',
 'Logistics Speed': '快递送货速度',
 'Pick-up Speed': '快递发货速度',
 'Wrong Delivery': '快递错发漏发',
 'Logistics Fee': '快递费用',
 'Logistics Service': '快递服务',
 'Logistics Company': '快递公司',
 'Logistics Package': '快递包装',
 'Logistics Damage': '快递破损',
 'Package Cleanliness': '包装清洁度',
 'Package Design': '包装设计',
 'Package Integrity': '包装完整度',
 'Package Material': '包装材质',
 'Package Printing': '包装印刷',
 'Package General': '包装概览',
 'Price Satisfaction': '价格满意度',
 'Price Sensitivity': '价格敏感度',
 'Promotion': '促销',
 'Shop/Customer Service': '店铺或客服服务',
 'Return Exchange': '退换货服务'
}


data_dir = '/data/projects/bert_pytorch/ecom_aspect_bak'
for dirname in os.listdir(data_dir):
    if dirname in TRANS_SUBTYPE:
        command = f'./run_ecom_op.sh {dirname}'
        print(f'******正在训练 {dirname}')
        os.system(command)

