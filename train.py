import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import (
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm
import os
import sys
from typing import Tuple, Optional
from accelerate import Accelerator


class ClipDataset(Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.captions_tokens[item],
            self.masks[item],
            self.prefixes[item],
        )

    def __init__(self, data_path: str, prefix_length: int):
        """
        Args:
            data_path: path to train.pkl, result of parse_viecap.py
            prefix_length:
        """
        self.prefix_length = prefix_length
        self.max_seq_len = 128
        dt = torch.load(data_path)
        sys.stdout.flush()
        self.captions_tokens = dt["target"]
        self.captions_tokens[self.captions_tokens.eq(1)] = 0
        self.prefixes = dt["clip_embedding"].float()
        self.masks = []
        for tokens in self.captions_tokens:
            # 5 is token <pad> in tokenizer
            mask = (tokens.greater(0)).long()
            mask = torch.cat((torch.ones(prefix_length), mask))
            self.masks.append(mask)


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        """Project clip output to embedding of first prefix_length tokens"""
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                # added some dropout here
                layers.append(nn.Dropout(p=0.2))
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate prefix tokens, shape Bxprefix_length"""
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int = 10, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("imthanhlv/gpt2news")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP(
            (
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            )
        )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def training_function(config):
    os.makedirs("./checkpoints", exist_ok=True)
    set_seed(config["seed"])
    accelerator = Accelerator()
    accelerator.print(config)

    epochs = config["epochs"]
    output_prefix = "nmt"
    output_dir = "checkpoints"
    prefix_length = 10

    train_dataset = ConcatDataset(
        (
            ClipDataset("./text_b16.pt", prefix_length),
            ClipDataset("./train_img_b16.pt", prefix_length),
            ClipDataset("./train_img_b16.pt", prefix_length),
            ClipDataset("./train_img_b16.pt", prefix_length),
        )
    )

    val_dataset = ClipDataset("./val_img_b16.pt", prefix_length)

    accelerator.print(len(train_dataset), len(val_dataset))
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )

    model = ClipCaptionModel(prefix_length)
    model = model.to(accelerator.device)
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5000,
        num_training_steps=epochs * len(train_dataloader),
    )

    for epoch in range(epochs):
        accelerator.print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(
            total=len(train_dataloader),
            desc=output_prefix,
            disable=not accelerator.is_local_main_process,
        )
        # Train phase
        model.train()
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, prefix_length - 1 : -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0
            )
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item()})
            progress.update()

        progress.close()
        if epoch % config["save_every"] == 0 or epoch == epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                unwrapped_model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

        val_loss = []

        if epoch % config["val_every"] == 0 or epoch == epochs - 1:
            accelerator.print("Running evaluate")
            model.eval()
            for step, (tokens, mask, prefix) in enumerate(val_dataloader):
                with torch.no_grad():
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, prefix_length - 1 : -1]
                    loss = nnf.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        tokens.flatten(),
                        ignore_index=0,
                    )
                    val_loss.append(accelerator.gather(loss))
            accelerator.print(epoch, ">>>>>>>>", torch.cat(val_loss).mean())
    return model


def main():
    config = {
        "epochs": 20,
        "lr": 2e-5,
        "seed": 42,
        "batch_size": 64,
        "prefix_length": 10,
        "save_every": 1,
        "val_every": 1,
    }
    training_function(config)


if __name__ == "__main__":
    main()
