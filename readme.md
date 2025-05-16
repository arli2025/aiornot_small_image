Repository contains models and sample for inferencing models on small images. 

Models are stored in dvc, first do 
```
dvc pull
```

How to start serving with triton server
```
bash make_sym_links.sh
sudo docker run --rm --gpus=1 -p8000:8000 -p8001:8001 -p8002:8002 -v"$(pwd)"/models/:/models/ nvcr.io/nvidia/tritonserver:25.02-py3 tritonserver --model-repository=/models
```

rine_vit is taken from here https://github.com/mever-team/rine
resnet was is based on codebase from here https://github.com/aiornotinc/aiornot_base
Both models were trained on same data

inference example
```
python inference_triton.py 1cf0ea0f8d067336cd5961bc5fa7b685dec433799d8eb08258f052ebfc1f5107.jpg 
```

hors2 server with A10G gives ~34 it/s or ~30 ms 
trt fp16 gives ~50 it/s or ~20 ms 

precision_recall_curve.html contains several curves, including resnet and ensembled variant comparing with baseline.
