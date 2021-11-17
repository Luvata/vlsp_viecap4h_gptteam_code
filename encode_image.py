import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from transformers import AutoTokenizer
import json
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


BATCH_SIZE = 64
MAX_TGT_SEQ_LEN = 128
VOCAB_SIZE = 50256


device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
vtokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
vtokenizer.pad_token = "<pad>"


def tokenize(text):
    return vtokenizer.encode(
        text, max_length=MAX_TGT_SEQ_LEN, truncation=True, padding="max_length"
    )
    # Follow huggingface jax clm script for making gpt tokenizers is not correct,
    # since tokenizer has 50265 tokens, whereas wte has only 50256
    # so all sentence with token id > VOCAB_SIZE must be filtered out


data = json.load(open("./viecap/train_captions.json"))
train, val = train_test_split(data, test_size=0.05, random_state=42)
public_test = json.load(open("viecap/sample_submission.json", "r"))
private_test = json.load(open("viecap/private_sample_sub.json", "r"))


print(
    "Train size:",
    len(train),
    "validation size:",
    len(val),
    "public test size:",
    len(public_test),
    "private test size:",
    len(private_test),
)
clip_model, preprocess = clip.load("ViT-B/16", device=device)


def create_dataset(text_image_pairs, image_path, save_path):
    """
    Calculate CLIP embeddings for each image and save them to disk
    Args:
        text_image_pairs: list of tuples (text, image_path)
        image_path: path to the images
        save_path: path to save the embeddings
    """
    clip_embedding = []
    tgt = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(text_image_pairs, batch_size=BATCH_SIZE)):
            images_path = [image_path + i for i in batch["id"]]
            images = torch.stack([preprocess(Image.open(i)) for i in images_path]).to(
                device
            )
            embeddings = clip_model.encode_image(images).cpu()
            if "captions" in batch:
                for embedding, captions, img_path in zip(
                    embeddings, batch["captions"], images_path
                ):
                    for caption in captions.split("\n"):
                        vt = tokenize(caption)
                        assert all(
                            [id <= VOCAB_SIZE for id in vt]
                        ), f"Must skip sentence with token ids > {VOCAB_SIZE}"
                        clip_embedding.append(embedding)
                        tgt.append(torch.LongTensor(vt))
                        ids.append(img_path)
    clip_embedding = torch.stack(clip_embedding)
    tgt = torch.stack(tgt)
    torch.save(
        {"clip_embedding": clip_embedding, "target": tgt, "id": ids},
        save_path,
    )
    print("Done")


create_dataset(train, "viecap/images_train/", "train_img_b16.pt")
create_dataset(val, "viecap/images_train/", "val_img_b16.pt")
create_dataset(public_test, "viecap/images_public_test/", "test_public_b16.pt")
create_dataset(private_test, "viecap/images_private_test/", "test_private_b16.pt")
