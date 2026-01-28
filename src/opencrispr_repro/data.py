import functools

import numpy as np
import pandas as pd
import torch
from composer.utils import dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizerBase

from .schema import FinetuneAPI


class SeqDataset(Dataset):
    def __init__(self, csv_fname: str, sequence_col: str, label_col: str | None = None):
        super().__init__()
        df = pd.read_csv(csv_fname, sep="\t" if csv_fname.endswith(".tsv") else ",")
        self.seqs = df[sequence_col].tolist()
        if label_col is not None:
            self.labels = df[label_col].tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        d = dict(sequence=self.seqs[idx])
        if self.labels is not None:
            d = dict(label=self.labels[idx])
        return d


def progen2_collate_fn(items: list[dict], tokenizer: PreTrainedTokenizerBase):
    seqs = ["1" + item["sequence"] + "2" for item in items]
    seqs = [seq if np.random() < 0.5 else seq[::-1] for seq in seqs]
    batch = tokenizer(seqs, return_tensors="pt", padding=True)
    return {
        "input_ids": batch["input_ids"],
        "labels": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }


def esm2_collate_fn(items: list[dict], tokenizer: PreTrainedTokenizerBase):
    seqs = [item["sequence"] for item in items]
    seqs = [seq if np.random() < 0.5 else seq[::-1] for seq in seqs]
    batch = tokenizer(seqs, return_tensors="pt", padding=True)
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": torch.tensor([item["label"] for item in items])
    }

def get_dataloader(config: FinetuneAPI, dataset: SeqDataset, tokenizer: PreTrainedTokenizerBase):
    model_name = config.model.name
    if model_name == "progen2":
        collate_fn = functools.partial(progen2_collate_fn, tokenizer=tokenizer)
    elif model_name == "esm2":
        collate_fn = functools.partial(esm2_collate_fn, tokenizer=tokenizer)
    else:
        raise ValueError("config.model.name must be progen2 or esm2")

    return DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_global_rank(),
            seed=0,
            drop_last=False,
        ),
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
    )
