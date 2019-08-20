#!/usr/bin/env bash


SUBTYPE=$1

export GLUE_DIR=/data/projects/bert_pytorch/
export CUDA_VISIBLE_DEVICES=1
export TASK_NAME=ecom_aspect_mtl_polar
export SQUAD_DIR=$GLUE_DIR/${TASK_NAME}/$SUBTYPE
export MODEL_DIR=$GLUE_DIR/${TASK_NAME}_out/all_in_one

python ../examples/run_squad_polar.py \
    --model_type bert \
    --model_name_or_path $MODEL_DIR/pytorch_model.bin \
    --config_name $MODEL_DIR/config.json \
    --tokenizer_name  $MODEL_DIR/vocab.txt \
    --do_eval \
    --do_lower_case \
    --multi_subtype_dir $GLUE_DIR/${TASK_NAME} \
    --ecom_subtype $SUBTYPE  \
    --train_file $SQUAD_DIR/train.json \
    --predict_file $SQUAD_DIR/dev.json \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --save_steps 1000 \
    --doc_stride 128 \
    --output_dir $GLUE_DIR/${TASK_NAME}_out/$SUBTYPE \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=16   \
    --version_2_with_negative \
    --overwrite_output_dir \
    --overwrite_cache

