python3 finetune_t5_cbqa.py --model_size t5.1.1.small_ssm \
	--finetune_steps 1000 \
	--save_per_steps 1000 \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v2-8 2>&1 | tee log_t5.1.1.small_ssm-v2-8.txt

#python3 finetune_t5_cbqa.py --model_size t5.1.1.small \
#	--finetune_steps 1000 \
#	--save_per_steps 1000 \
#	--base_dir gs://caramel-spot-280923 \
#	--tpu_type v2-8 2>&1 | tee log_t5.1.1.small-v2-8.txt
#
#python3 finetune_t5_cbqa.py --model_size small \
#	--finetune_steps 1000 \
#	--save_per_steps 1000 \
#	--base_dir gs://caramel-spot-280923 \
#	--tpu_type v2-8 2>&1 | tee log_small-v2-8.txt
#
#python3 finetune_t5_cbqa.py --model_size base \
#	--finetune_steps 1000 \
#	--save_per_steps 1000 \
#	--base_dir gs://caramel-spot-280923 \
#	--tpu_type v2-8 2>&1 | tee log_base-v2-8.txt
#
#
#python3 finetune_t5_cbqa.py --model_size large \
#	--finetune_steps 1000 \
#	--save_per_steps 1000 \
#	--base_dir gs://caramel-spot-280923 \
#	--tpu_type v2-8 2>&1 | tee log_large-v2-8.txt
#
#
#python3 finetune_t5_cbqa.py --model_size 3B \
#	--finetune_steps 1000 \
#	--save_per_steps 1000 \
#	--base_dir gs://caramel-spot-280923 \
#	--tpu_type v2-8 2>&1 | tee log_3B-v2-8.txt
#

