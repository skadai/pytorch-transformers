#!/usr/bin/env bash
# 依次接收三个参数 subtype, task_name, model_dirname

SUBTYPE=$1
TASK_NAME=$2
SUBDICT=$4

export GLUE_DIR=/data/projects/bert_pytorch/
export CUDA_VISIBLE_DEVICES=0
export SQUAD_DIR=$GLUE_DIR/${TASK_NAME}/$SUBTYPE
export MODEL_DIR=$GLUE_DIR/${TASK_NAME}_out/$3

python ../examples/run_skincare_v2.py \
    --model_type multi_v3 \
    --model_name_or_path $MODEL_DIR/pytorch_model.bin \
    --config_name $GLUE_DIR/$TASK_NAME/config.json \
    --tokenizer_name  $GLUE_DIR/vocab.txt \
    --do_eval \
    --do_lower_case \
    --multi_subtype_dir $GLUE_DIR/${TASK_NAME} \
    --ecom_subtype $SUBTYPE  \
    --train_file $SQUAD_DIR/train.json \
    --predict_file $SQUAD_DIR/dev.json \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --max_seq_length 256 \
    --save_steps 1000 \
    --doc_stride 128 \
    --output_dir $MODEL_DIR/$SUBTYPE \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=16   \
    --version_2_with_negative \
    --subtype_dict ${SUBDICT} \
    --runs_name ${SUBTYPE} \
    --task_name ${TASK_NAME} \
    --overwrite_output_dir \
