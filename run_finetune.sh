#!/bin/bash

FINETUNE_STEPS=10000
TOTAL_STEPS=1110000
TPU_TYPE=v3-8
BUCKET_NAME=t5-tutorial-storage
PRETRAINED_DIR=gs://t5-tutorial-storage/pretrained


MODEL_SIZE=t5.1.1.small_ssm
python3 finetune_t5_cbqa.py --model_size ${MODEL_SIZE} \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--base_dir gs://${BUCKET_NAME} \
	--pretrained_dir ${PRETRAINED_DIR} \
	--tpu_type ${TPU_TYPE} 2>&1 | tee log_${TPU_TYPE}_${MODEL_SIZE}_${TOTAL_STEPS}.txt

MODEL_SIZE=t5.1.1.xl_ssm
python3 finetune_t5_cbqa.py --model_size ${MODEL_SIZE} \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--base_dir gs://${BUCKET_NAME} \
	--pretrained_dir ${PRETRAINED_DIR} \
	--tpu_type ${TPU_TYPE} 2>&1 | tee log_${TPU_TYPE}_${MODEL_SIZE}_${TOTAL_STEPS}.txt

MODEL_SIZE=t5.1.1.xxl_ssm
python3 finetune_t5_cbqa.py --model_size ${MODEL_SIZE} \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--base_dir gs://${BUCKET_NAME} \
	--pretrained_dir ${PRETRAINED_DIR} \
	--tpu_type ${TPU_TYPE} 2>&1 | tee log_${TPU_TYPE}_${MODEL_SIZE}_${TOTAL_STEPS}.txt
