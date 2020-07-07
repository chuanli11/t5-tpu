#!/bin/bash

FINETUNE_STEPS=10000
TOTAL_STEPS=1110000
BUCKET_NAME=t5-tutorial-storage
MODEL_SIZE=t5.1.1.xxl_ssm_tqa

python3 eval_t5_cbqa.py --model_size $MODEL_SIZE \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--data_dir gs://${BUCKET_NAME} \
	--model_dir gs://${BUCKET_NAME}/reference/${MODEL_SIZE} \
	--tpu_type v2-8 2>&1 | tee eval_reference_${MODEL_SIZE}_${TOTAL_STEPS}.txt



