import os
import pickle
import click
import yaml
import torch
import pandas as pd
import hashlib

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from grna_modeling.model.gRNAModel import gRNAModel, gRNAModelInput
from grna_modeling.utility.protein import NucleicAcidBatch, ProteinBatch
from grna_modeling.utility.esm import ESM2
from grna_modeling.utility import vocabulary


class gRNADatasetWithIdx(Dataset):
    """Dataset that includes indices for mapping to embeddings."""
    def __init__(self, rna_seqs, protein_seqs):
        self.rna_seqs = rna_seqs
        self.protein_seqs = protein_seqs

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        return {
            "rna": self.rna_seqs[idx],
            "protein": self.protein_seqs[idx],
            "idx": idx,
        }


def build_rna_targets(df: pd.DataFrame, columns_cfg: dict, tracr_first: bool = True):
    """Build RNA and protein sequences, ensuring lowercase RNA sequences."""
    protein_col = columns_cfg["protein"]
    tracr_col = columns_cfg.get("tracr")
    crispr_col = columns_cfg.get("crispr")

    if crispr_col not in df.columns:
        raise ValueError(f"Missing crispr column: {crispr_col}")
    if protein_col not in df.columns:
        raise ValueError(f"Missing protein column: {protein_col}")

    # tracr sequences, lowercase
    if tracr_col in df.columns:
        tracr_seqs = df[tracr_col].fillna("").astype(str).str.lower().tolist()
    else:
        tracr_seqs = [""] * len(df)

    # crispr sequences, lowercase
    crispr_seqs = df[crispr_col].fillna("").astype(str).str.lower().tolist()

    rna_seqs = vocabulary.concat_rna_sequences(
        tracr_seqs,
        crispr_seqs,
        tracr_first=tracr_first
    )
    protein_seqs = df[protein_col].astype(str).tolist()

    return rna_seqs, protein_seqs


def make_collate_fn(esm_model, save_embeddings=False, embeddings_dir=None, split="train"):
    """
    Collate function that:
    - Uses content-based hashing for caching
    - Logs cache hits/misses
    - Pads protein embeddings
    """
    if save_embeddings:
        embeddings_dir = os.path.abspath(embeddings_dir)
        os.makedirs(embeddings_dir, exist_ok=True)

    def hash_seq(seq: str) -> str:
        return hashlib.sha1(seq.strip().encode()).hexdigest()

    def collate_fn(batch):
        rna_batch = NucleicAcidBatch.from_sequences([b["rna"] for b in batch])
        protein_seqs = [b["protein"] for b in batch]
        protein_embs_list = []

        hits, misses = 0, 0

        for b in batch:
            protein = b["protein"]

            if save_embeddings:
                key = hash_seq(protein)
                emb_path = os.path.join(embeddings_dir, f"{key}.pkl")

                if os.path.exists(emb_path):
                    with open(emb_path, "rb") as f:
                        emb = pickle.load(f)
                    hits += 1
                else:
                    emb = esm_model.embed(protein)
                    with open(emb_path, "wb") as f:
                        pickle.dump(emb, f)
                    misses += 1
            else:
                emb = esm_model.embed(protein)

            protein_embs_list.append(emb)

        if save_embeddings:
            print(f"[{split}] cache hits: {hits}, misses: {misses}")

        # Pad embeddings along sequence length
        max_len = max(e.size(0) for e in protein_embs_list)
        d_s_protein = protein_embs_list[0].size(1)

        padded_embs = []
        for e in protein_embs_list:
            pad_len = max_len - e.size(0)
            if pad_len > 0:
                e = torch.cat(
                    [e, torch.zeros(pad_len, d_s_protein, device=e.device)],
                    dim=0
                )
            padded_embs.append(e)

        protein_embs = torch.stack(padded_embs, dim=0).to(rna_batch.S.device)

        return gRNAModelInput(
            rna_batch=rna_batch,
            protein_batch=ProteinBatch.from_sequences(protein_seqs),
            protein_embs=protein_embs,
            S_label=rna_batch.S,
        )

    return collate_fn


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--save-embeddings", is_flag=True, default=False)
@click.option("--embeddings-dir", type=click.Path(), default="./embeddings")
def main(config, save_embeddings, embeddings_dir):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model = ESM2(cfg["esm_model"], device=device)

    dataset_cfg = cfg["dataset"]
    columns_cfg = dataset_cfg["columns"]
    tracr_first = dataset_cfg.get("tracr_first", True)

    # --- Train ---
    df_train = pd.read_csv(dataset_cfg["train_csv"])
    rna_train, protein_train = build_rna_targets(df_train, columns_cfg, tracr_first)

    train_dataset = gRNADatasetWithIdx(rna_train, protein_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(
            esm_model,
            save_embeddings,
            embeddings_dir,
            split="train"
        ),
    )

    # --- Validation ---
    df_val = pd.read_csv(dataset_cfg["val_csv"])
    rna_val, protein_val = build_rna_targets(df_val, columns_cfg, tracr_first)

    val_dataset = gRNADatasetWithIdx(rna_val, protein_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(
            esm_model,
            save_embeddings,
            embeddings_dir,
            split="val"
        ),
    )

    # --- Model ---
    model = gRNAModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="grna_model-{epoch:02d}-{step:05d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=-1,
        strategy="auto",
        precision=16,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=cfg["optimizer"]["acc_batches"],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()