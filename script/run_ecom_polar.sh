#!/usr/bin/env bash


export GLUE_DIR=/data/projects/bert_pytorch/
export CUDA_VISIBLE_DEVICES=1
export TASK_NAME=ecom
export TASK_NOTE=label_3_polar
export MODEL_DIR=$GLUE_DIR/${TASK_NAME}_output/$TASK_NOTE


python ../examples/run_glue.py \
    --model_type bert \
    --model_name_or_path $GLUE_DIR/pytorch_model.bin \
    --config_name $GLUE_DIR/config.json \
    --tokenizer_name  $GLUE_DIR/vocab.txt \
    --task_name $TASK_NAME \
    --save_steps 2000 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME/$TASK_NOTE \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir $GLUE_DIR/${TASK_NAME}_output/$TASK_NOTE
