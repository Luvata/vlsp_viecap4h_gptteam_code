# VLSP 2021 - viecap4h challenge

This project is a fork from rmokady's [CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption)
with some modification from gptteam to run with viecap4h challenge.

Our contributions are:

- [x] Using pretrained Language Model: Vietnamese GPT-2 as decoder from @imthanhlv, with CLIP Vit-B16 as text
and image encoder.
- [x] Propose a novel method to improve image captioning performance, especially for low-resources dataset,
by joint training with billigual text dataset (iwslt15 en-vi) using a share text-image encoder.


## How to run

1. Install require packages and download datasets
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
./sh down_viecap.sh
```

2. Prepare image captioning data (on cuda device)

```
python encode_image.py
```

3. Prepare iwslt data (on cuda device)

```
python encode_text.py
```
4. Optional: Configure huggingface's accelerate for CUDA/TPU

```
accelerate config
```

5. Train: Training for 20 epochs in 30 mins using a single TPU V3-8.

```
python train.py
```

6. Inference (run on cuda devices)

```
python inference.py
```

