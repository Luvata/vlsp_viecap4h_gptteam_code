import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from transformers import AutoTokenizer
from datasets import load_dataset


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


dataset = load_dataset("mt_eng_vietnamese", "iwslt2015-en-vi")
en_sentences = [
    i["en"]
    for i in dataset["train"]["translation"]
    if (len(i["en"].strip()) > 0 and len(i["vi"].strip()) > 0)
]
vi_sentences = [
    i["vi"]
    for i in dataset["train"]["translation"]
    if (len(i["en"].strip()) > 0 and len(i["vi"].strip()) > 0)
]
en_token = []
vi_token = []
skip = []

# filter
# (1) English sentence with len > 77, clip.tokenize will throw error
# (2) Vietnamese sentence with id > VOCAB_SIZE


for en, vi in tqdm(zip(en_sentences, vi_sentences)):
    try:
        et = clip.tokenize([en])  # torch
        vt = tokenize(vi)  # array
        assert all([id <= VOCAB_SIZE for id in vt])
        en_token.append(et)
        vi_token.append(vt)

    except:
        skip.append(en)

print("Skip total", len(skip))


def collate_fn(batch):
    be = [i[0] for i in batch]
    bv = [i[1] for i in batch]
    return torch.cat(be), torch.tensor(bv)


dataloader = DataLoader(
    list(zip(en_token, vi_token)),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True,
)

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_embedding = []
tgt = []


with torch.no_grad():
    for et, vt in tqdm(dataloader):
        e_embed = clip_model.encode_text(et.to(device)).cpu()
        clip_embedding.append(e_embed)
        tgt.append(vt)


clip_embedding = torch.cat(clip_embedding)
tgt = torch.cat(tgt)
torch.save({"clip_embedding": clip_embedding, "target": tgt}, "text_b16.pt")
