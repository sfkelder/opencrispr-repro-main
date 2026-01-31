import yaml
import click
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from einops import repeat

from pytorch_lightning.callbacks import ModelCheckpoint
from grna_modeling.model.gRNAModel import gRNAModel, gRNAModelInput
from grna_modeling.utility.protein import NucleicAcidBatch, ProteinBatch
from grna_modeling.utility.esm import ESM2

class gRNADataset(Dataset):
    def __init__(self, rna_seqs, protein_seqs, esm_model: ESM2, device="cuda"):
        self.rna_seqs = rna_seqs
        self.protein_seqs = protein_seqs
        self.esm_model = esm_model
        self.device = device

        # Precompute embeddings
        self.protein_embs = []
        for seq in protein_seqs:
            emb = esm_model.embed(seq).to(device) 
            self.protein_embs.append(emb)

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        return {
            "rna": self.rna_seqs[idx],
            "protein": self.protein_seqs[idx],
            "protein_emb": self.protein_embs[idx],
        }

def collate_fn(batch):
    # RNA batch
    rna_batch = NucleicAcidBatch.from_sequences([b["rna"] for b in batch])

    # Protein batch
    protein_batch = ProteinBatch.from_sequences([b["protein"] for b in batch])

    # Pad protein embeddings to max length in batch
    protein_embs_list = [b["protein_emb"] for b in batch]
    max_len = max(e.size(0) for e in protein_embs_list)
    d_s_protein = protein_embs_list[0].size(1)

    padded_embs = []
    for e in protein_embs_list:
        pad_len = max_len - e.size(0)
        if pad_len > 0:
            e = torch.cat([e, torch.zeros(pad_len, d_s_protein).to(e.device)], dim=0)
        padded_embs.append(e)

    protein_embs = torch.stack(padded_embs, dim=0)  # (B, L_max, d_s_protein)

    return gRNAModelInput(
        rna_batch=rna_batch,
        protein_batch=protein_batch,
        protein_embs=protein_embs,
        S_label=rna_batch.S,
    )

@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to YAML config file")
def main(config):
    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ESM model for embeddings
    esm_model = ESM2(cfg["esm_model"], device=device)

    dataset_cfg = cfg["dataset"]
    rna_col = dataset_cfg["columns"]["rna"]
    protein_col = dataset_cfg["columns"]["protein"]

    # Train
    df_train = pd.read_csv(dataset_cfg["train_csv"])
    rna_train = df_train[rna_col].tolist()
    protein_train = df_train[protein_col].tolist()
    train_dataset = gRNADataset(rna_train, protein_train, esm_model, device=device)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)

    # Validation
    df_val = pd.read_csv(dataset_cfg["val_csv"])
    rna_val = df_val[rna_col].tolist()
    protein_val = df_val[protein_col].tolist()
    val_dataset = gRNADataset(rna_val, protein_val, esm_model, device=device)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = gRNAModel(cfg)

    # Callbacks
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
            strategy="ddp",                    
            precision=16,                       
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=cfg["optimizer"]["acc_batches"], 
            log_every_n_steps=50,              
            check_val_every_n_epoch=1,        
            enable_progress_bar=True,          
        )

    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
