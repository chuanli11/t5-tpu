python3 finetune_t5_cbqa.py --model_size small \
	--finetune_steps 25000 \
	--save_per_steps 25000 \
	--base_dir gs://caramel-spot-280923 2>&1 | tee log_small.txt

python3 finetune_t5_cbqa.py --model_size base \
	--finetune_steps 25000 \
	--save_per_steps 25000 \
	--base_dir gs://caramel-spot-280923 2>&1 | tee log_base.txt

python3 finetune_t5_cbqa.py --model_size large \
	--finetune_steps 25000 \
	--save_per_steps 25000 \
	--base_dir gs://caramel-spot-280923 2>&1 | tee log_large.txt

python3 finetune_t5_cbqa.py --model_size 3B \
	--finetune_steps 25000 \
	--save_per_steps 25000 \
	--base_dir gs://caramel-spot-280923 2>&1 | tee log_3B.txt
