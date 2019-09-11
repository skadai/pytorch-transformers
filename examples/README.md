## 模型训练说明

### 约定命名

**task_name**: 一次模型训练任务名，对应两个文件夹 `<task_name>` 和 `<task_name>_out `前者存储训练数据，后者存放历次运行的结果模型文件和相关评估结果

**run_name**: 一次模型训练可以有多次运行, 分别对应`<task_name>_out`目录下的`<run_name>`文件

**subdict**: 当前训练任务对应的subtype定义，训练前手动添加到`SUBTYPE.json`，参考已有格式

**注: 当前两个GPU服务器的本项目数据目录在 `/data/projects/bert_pytorch`**

### 准备数据集

1. 抽取不同subtype的标注json文件作为正样本
2. 增加排除词作为负样本
3. 增加易混淆的subtype标注语料作为负样本
4. 正负样本组合，划分各subtype的训练集/(验证集)/测试集

### 模型训练

1. 推荐训练参数

   | 参数             | 推荐值 | 参数          | 推荐值      |
   | ---------------- | ------ | ------------- | ----------- |
   | train_batch_size | 16     | weight_decay  | 0.01        |
   | eval_batch_size  | 32     | Adam_epsilon  | 1e-6        |
   | num_train_epochs | 5      | Learning_rate | [2e-5,5e-5] |

   说明

   - epoch:  训练脚本目前没有early stop功能，但是通过设置合理的save step步长保存checkpoint，然后评估模型选择较优的模型，根据经验**epoch大于5后测试集指标增长缓慢甚至出现负增长**

2. 目前训练脚本模型启动了mlflow 功能，会将模型训练参数以及training loss等信息写入mlflow server，mlflow server可通过 http://172.16.24.32(33):9001访问

3. 训练脚本使用说明

   ```shell
   # sample_ratio 是训练集数据采样一定比例, 不传时默认1
   # subdict 为预先写入SUBTYPE.json的品类subtype 词典名, 不传默认general
   ./run_ecom_senti.sh {run_name} {task_name}  {subdict}  {sample_ratio}
   ```

### 模型评估

1. 模型评估分为两部分：
   - 特征词，情感词采用阅读理解模型的评估指标 exact,f1
   - 情感极性采用分类模型的评估指标 precision,recall,f1
   
2. 评估脚本使用说明
   - 特征词情感词评估
     
     ```shell
     # 跑模型评估(script目录)
     ./run_ecom_senti_eval.sh  {subtype}   {task_name}  {run_name} {subdict}
     # 计算指标 (examples目录)
     python evaluate_ecom_asop.py -st <aspect|op> -sd <subdict> -rn <run_name> -tn <task_name> filename
     ```
     
   - 情感极性评估
     ```shell
     # 模型评估(script目录)
     ./run_ecom_senti_polar_eval.sh  {subtype}   {task_name}  {run_name} {subdict}
     ```
     
     终端会输正/负/中三个极性的评估结果, 且保存到输出目录的eval_results.txt文件内
   
   **也可使用eval_sample_ratio.py脚本，将评估结果写入mlflow server**

### 模型封装

- jupyter notebook临时调用，参考模板`examples/ecom_senti.ipynb`

- 如需提供api服务， 使用时，将`exampls/flask_example.py`中, 模型路径修改为你的，然后启动gunicorn即可

  - 修改model_path
  
  ```python
  if __name__ == "__main__":
      model_path = <你的模型路径,需要包含config.json/vocab.txt/pytorch_model.bin三个文件>
      model = Model(model_path=model_path, target_device=<指定GPU/CPU，例如'cuda:0'>)
  
      # start child thread as worker
      streamer = ThreadedStreamer(model.predict, batch_size=8, max_latency=0.1)
  
      # spawn child process as worker
      # streamer = Streamer(model.predict, batch_size=64, max_latency=0.1)
  
    app.run(host='0.0.0.0', port=5005, debug=True)
  ```
  
  - 启动gunicorn
  
    ```shell
    gunicorn --config gunicorn_conf.py wsgi:app
    ```
  
  - 客户端调用，文本以字符串列表传入，支持多条文本(最大不超过16)
  
    ```shell
    curl -X POST \
      http://172.16.24.33:5005/stream \
      -H 'Cache-Control: no-cache' \
      -H 'Content-Type: application/json' \
      -d '{
          "texts":   [
     "还是网上买东西实惠，宝贝收到了，我超喜欢，呵呵，服务态度也不错，很有心的店家，以后常光顾。",
     "欧莱雅的洗面奶一直在用，挺好的，干爽不紧绷，挺好的，京东的服务也很不错，会一直回购"]
    
    }'
    ```
  
    可以看到正常返回
  
    ```json
    [
      {
        "opinions": [],
        "text": "还是网上买东西实惠,宝贝收到了,我超喜欢,呵呵,服务态度也不错,很有心的店家,以后常光顾,"
      },
      {
        "opinions": [
          {
            "aspectSubtype": "Greasy",
            "aspectTerm": "干爽",
            "opinionTerm": "干爽",
            "polarity": 5
          },
          {
            "aspectSubtype": "Moisturization",
            "aspectTerm": "不紧绷",
            "opinionTerm": "不紧绷",
            "polarity": 5
          }
        ],
        "text": "欧莱雅的洗面奶一直在用,挺好的,干爽不紧绷,挺好的,京东的服务也很不错,会一直回购"
      }
    ]
    ```