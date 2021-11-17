# VLSP 2021 - viecap4h challenge

This project is a fork from rmokady's [CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption)
with some modification from gptteam to run with viecap4h challenge.

Our contributions are:

- [x] Using pretrained Language Model: Vietnamese GPT-2 as decoder from @imthanhlv, with CLIP Vit-B16 as text
and image encoder.
- [x] Propose a novel method to improve image captioning performance, especially for low-resources dataset,
by joint training with billigual text dataset (iwslt15 en-vi) using a share text-image encoder.


## How to run

1. Install require packages and download datasets, pretrained embedding
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
./sh down_viecap.sh
```

2. [Optional] Computing image embedding data and translation embedding, 
you can take a look if you want to adapt to your custom dataset, 
but for viecap we already provide theses embedding files in `down_viecap.sh`

```
python encode_image.py
python encode_text.py
```

3. Training, we highly recommend using V38 free from Kaggle to train
```
accelerate launch --config_file ./config_tpu.yml train.py  # For v38, ~1h
# accelerate launch --config_file ./config_tpu.yml train.py --batch-size=32  # For v28, ~2h
# accelerate launch --config_file ./config_p100.yml train.py --batch-size=16  # For P100, ~1d
```

4. Inference (run on cuda devices)
```
python inference.py
```
