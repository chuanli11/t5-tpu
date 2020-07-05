# t5-tpu

```
# Set up Google Cloud Storage Bucket
export PROJECT_ID=caramel-spot-280923
export STORAGE_ZONE=us-central
export BUCKET=gs://${PROJECT_ID}
gsutil mb -p ${PROJECT_ID} -c standard -l ${STORAGE_ZONE} -b on ${BUCKET}


# On local machine
export PROJECT_ID=caramel-spot-280923
export TPU_ZONE=us-central1-b
export BUCKET=gs://${PROJECT_ID}
export TPU_NAME=t5-${PROJECT_ID}

# Choose TPU instance based on model
# 11B   : v3-8
# Others: v2-8
ctpu up   --name=$TPU_NAME   --project=$PROJECT_ID  --zone=$ZONE   --tpu-size=v2-8


# SSH into TPU instance
# Do not create virtualenv since pip installed tensorflow-gpu does not work well with TPU
pip3 install t5


git clone https://github.com/chuanli11/t5-tpu.git
cd t5-tpu

python3 finetune_t5_cbqa.py
```


#### Bug Fix

```
# Some operative_config.gin file have use_custom_ops set to True. This cause "tensor2tensor.data_generators.ops.pack_sequences_ops not found error"

vim /home/ubuntu/.local/lib/python3.7/site-packages/mesh_tensorflow/transformer/dataset.py

# Add this to line 522
use_custom_ops=False
```
