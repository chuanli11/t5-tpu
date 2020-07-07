#!/bin/bash

FINETUNE_STEPS=10000
TOTAL_STEPS=1110000

python3 eval_t5_cbqa.py --model_size reference_t5.1.1.xl_ssm_nq \
	--finetune_steps $FINETUNE_STEPS \
	--save_per_steps $FINETUNE_STEPS \
	--data_dir gs://caramel-spot-280923 \
	--model_dir gs://caramel-spot-280923/models-v2-8/reference_t5.1.1.xl_ssm_nq \
	--tpu_type v2-8 2>&1 | tee eval_reference_v2-8_t5.1.1.xl_ssm_${TOTAL_STEPS}.txt



