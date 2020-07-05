#!/bin/bash

FINETUNE_STEPS=10000
TOTAL_STEPS=1110000

python3 finetune_t5_cbqa.py --model_size t5.1.1.small_ssm \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v2-8 2>&1 | tee log_v2-8_t5.1.1.small_ssm_${TOTAL_STEPS}.txt

python3 finetune_t5_cbqa.py --model_size t5.1.1.xl_ssm \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v2-8 2>&1 | tee log_v2-8_t5.1.1.xl_ssm_${TOTAL_STEPS}.txt

python3 finetune_t5_cbqa.py --model_size t5.1.1.xxl_ssm \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v2-8 2>&1 | tee log_v2-8_t5.1.1.xxl_ssm_${TOTAL_STEPS}.txt


