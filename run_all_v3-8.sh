python3 finetune_t5_cbqa.py --model_size 11B \
	--finetune_steps 25000 \
	--save_per_steps 25000 \
	--base_dir gs://caramel-spot-280923 2>&1 | tee log_11B.txt

