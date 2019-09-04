#!/usr/bin/env bash
# 依次接收2个参数 subtype, task_name
# 重要的超参数
# learning_rate 2e-5 - 5e-5
# epoch 2-6
# traning-size 16
# max-seq-length 256 (和training-size配合)


RUNS_NAME=$1
TASK_NAME=$2

export GLUE_DIR=/data/projects/bert_pytorch/
export CUDA_VISIBLE_DEVICES=0
export SQUAD_DIR=$GLUE_DIR/${TASK_NAME}/$RUNS_NAME


python ../examples/run_skincare_v2.py \
    --model_type  multi_v3 \
    --model_name_or_path $GLUE_DIR/pytorch_model.bin \
    --config_name $GLUE_DIR/$TASK_NAME/config.json \
    --tokenizer_name  $GLUE_DIR/vocab.txt \
    --do_train \
    --do_lower_case \
    --multi_subtype_dir $GLUE_DIR/${TASK_NAME} \
    --ecom_subtype $RUNS_NAME  \
    --train_file $SQUAD_DIR/train.json \
    --predict_file $SQUAD_DIR/dev.json \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --adam_epsilon 1e-6 \
    --max_seq_length 256 \
    --max_answer_length 20 \
    --save_steps 100 \
    --weight_decay 0.01 \
    --doc_stride 128 \
    --output_dir $GLUE_DIR/${TASK_NAME}_out/$RUNS_NAME \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=16   \
    --subtype_dict smell \
    --train_sample_ratio 0.6 \
    --runs_name ${RUNS_NAME} \
    --task_name ${TASK_NAME} \
    --version_2_with_negative \
    --overwrite_output_dir \
