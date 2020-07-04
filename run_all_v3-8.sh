python3 finetune_t5_cbqa.py --model_size small \
	--finetune_steps 1000 \
	--save_per_steps 1000 \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v3-8 2>&1 | tee log_small-v3-8.txt

python3 finetune_t5_cbqa.py --model_size base \
	--finetune_steps 1000 \
	--save_per_steps 1000 \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v3-8 2>&1 | tee log_base-v3-8.txt


python3 finetune_t5_cbqa.py --model_size large \
	--finetune_steps 1000 \
	--save_per_steps 1000 \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v3-8 2>&1 | tee log_large-v3-8.txt


python3 finetune_t5_cbqa.py --model_size 3B \
	--finetune_steps 1000 \
	--save_per_steps 1000 \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v3-8 2>&1 | tee log_3B-v3-8.txt


python3 finetune_t5_cbqa.py --model_size 11B \
	--finetune_steps 25000 \
	--save_per_steps 25000 \
	--base_dir gs://caramel-spot-280923 \
	--tpu_type v3-8 2>&1 | tee log_11B-v3-8.txt
