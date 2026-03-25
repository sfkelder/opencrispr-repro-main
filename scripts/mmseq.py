import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import torch
import esm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
from filelock import FileLock  # added for safe appending

# Reproducible subsampling
np.random.seed(42)
random.seed(42)

############################################
# ARGUMENTS
############################################

parser = argparse.ArgumentParser()

parser.add_argument("--mode", choices=["prepare", "evaluate"], required=True)
parser.add_argument("--train", help="training dataset")
parser.add_argument("--gen_dir", help="directory of generated sets")
parser.add_argument("--format", choices=["fasta", "csv"], default="fasta")
parser.add_argument("--seq_col", default="sequence")
parser.add_argument("--id_col", default="id")
parser.add_argument("--reference_dir", default="train_reference")
parser.add_argument("--results_dir", default="results")
parser.add_argument("--tmp_dir", required=True)


# Embeddings
parser.add_argument("--save_embeddings", action="store_true",
                    help="Save generated embeddings (only in evaluate mode)")
parser.add_argument("--embeddings_dir", default="embeddings",
                    help="Directory to save embeddings")

# Pseudo-perplexity parameters
parser.add_argument("--max_ppl_seqs", type=int, default=200,
                    help="Maximum number of sequences to use for pseudo-perplexity")
parser.add_argument("--truncate_length", type=int, default=400,
                    help="Maximum sequence length for perplexity computation")
parser.add_argument("--mask_batch_size", type=int, default=128,
                    help="Number of masked positions per batch for perplexity")

args = parser.parse_args()

os.makedirs(args.reference_dir, exist_ok=True)
os.makedirs(args.results_dir, exist_ok=True)
if args.mode == "evaluate" and args.save_embeddings:
    os.makedirs(args.embeddings_dir, exist_ok=True)

TMP_DIR = args.tmp_dir
os.makedirs(TMP_DIR, exist_ok=True)

############################################
# DEVICE
############################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################
# LOAD SEQUENCES
############################################

def load_sequences(file):
    seqs = []
    if args.format == "fasta":
        for r in SeqIO.parse(file, "fasta"):
            seqs.append((r.id, str(r.seq)))
    elif args.format == "csv":
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            seqs.append((str(row[args.id_col]), row[args.seq_col]))
    return seqs

############################################
# WRITE FASTA
############################################

def write_fasta(seqs, path):
    with open(path, "w") as f:
        for id, seq in seqs:
            f.write(f">{id}\n{seq}\n")

############################################
# LOAD ESM MODEL
############################################

def load_model():
    print("Loading ESM model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

############################################
# COMPUTE EMBEDDINGS
############################################

def compute_embeddings(seqs, model, batch_converter, batch_size=32):
    embeddings = {}
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            results = model(toks, repr_layers=[33])
            reps = results["representations"][33]
        for j, label in enumerate(labels):
            emb = reps[j, 1:len(strs[j])+1].mean(0)
            embeddings[label] = emb.cpu().numpy()
    return embeddings

############################################
# PERPLEXITY
############################################

def compute_perplexity(seqs, model, batch_converter):
    mask_idx = model.alphabet.mask_idx
    perplexities = []
    for label, seq in tqdm(seqs, desc="Perplexity"):
        batch = [(label, seq)]
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(device)
        seq_len = len(seq)
        log_probs = []
        positions = list(range(1, seq_len+1))
        for i in range(0, seq_len, args.mask_batch_size):
            batch_positions = positions[i:i+args.mask_batch_size]
            masked_batch = toks.repeat(len(batch_positions), 1)
            for j, pos in enumerate(batch_positions):
                masked_batch[j, pos] = mask_idx
            with torch.no_grad():
                outputs = model(masked_batch)
            logits = outputs["logits"]
            log_soft = torch.log_softmax(logits, dim=-1)
            for j, pos in enumerate(batch_positions):
                true_token = toks[0, pos]
                log_prob = log_soft[j, pos, true_token]
                log_probs.append(log_prob.item())
        ppl = np.exp(-np.mean(log_probs))
        perplexities.append(ppl)
    return np.array(perplexities)

############################################
# PREPARE TRAINING REFERENCE
############################################

def prepare_reference():
    print("Loading training sequences...")
    train_seqs = load_sequences(args.train)

    fasta_tmp = os.path.join(args.reference_dir, "train.fasta")
    write_fasta(train_seqs, fasta_tmp)

    model, batch_converter = load_model()

    print("Computing training embeddings...")
    train_emb = compute_embeddings(train_seqs, model, batch_converter)

    ids = list(train_emb.keys())
    matrix = np.vstack(list(train_emb.values()))

    # Always save training embeddings
    np.save(os.path.join(args.reference_dir, "train_embeddings.npy"), matrix)
    pd.DataFrame({"id": ids}).to_csv(
        os.path.join(args.reference_dir, "train_ids.csv"),
        index=False
    )

    subprocess.run([
        "mmseqs",
        "createdb",
        fasta_tmp,
        os.path.join(args.reference_dir, "trainDB")
    ], check=True)

    print("Training reference ready.")

############################################
# LOAD REFERENCE
############################################

def load_reference():
    train_matrix = np.load(os.path.join(args.reference_dir, "train_embeddings.npy"))
    ids = pd.read_csv(os.path.join(args.reference_dir, "train_ids.csv"))["id"].tolist()
    return ids, train_matrix

############################################
# MMSEQS SEARCH
############################################

def run_mmseqs(gen_fasta, prefix):
    train_db = os.path.join(args.reference_dir, "trainDB")
    gen_db = f"{prefix}_genDB"
    res_db = f"{prefix}_resDB"

    subprocess.run(["mmseqs", "createdb", gen_fasta, gen_db], check=True)
    subprocess.run([
        "mmseqs",
        "search",
        gen_db,
        train_db,
        res_db,
        TMP_DIR,
        "-s", "7.5",
        "--min-seq-id", "0",
        "--threads", "16"
    ], check=True)

    tsv = f"{prefix}_results.tsv"
    subprocess.run([
        "mmseqs",
        "convertalis",
        gen_db,
        train_db,
        res_db,
        tsv
    ], check=True)
    return tsv

############################################
# PARSE MMSEQS
############################################

def parse_mmseqs(tsv, queries):
    cols = [
        "query","target","pident","alnlen","mismatch","gapopen",
        "qstart","qend","tstart","tend","evalue","bits"
    ]
    df = pd.read_csv(tsv, sep="\t", names=cols)
    best = df.sort_values("bits", ascending=False).drop_duplicates("query")
    all_queries = pd.DataFrame({"query": queries})
    best = all_queries.merge(best, on="query", how="left")
    return best

############################################
# EMBEDDING METRICS
############################################

def embedding_metrics(train_matrix, gen_matrix):
    sim = cosine_similarity(gen_matrix, train_matrix)
    nearest = sim.max(axis=1)
    nbrs = NearestNeighbors(n_neighbors=5, metric="cosine").fit(train_matrix)
    dists, _ = nbrs.kneighbors(gen_matrix)
    nn_distance = dists.mean(axis=1)
    return nearest, nn_distance

############################################
# GENERATION DIVERSITY
############################################

def generation_diversity(gen_matrix):
    sim = cosine_similarity(gen_matrix, gen_matrix)
    np.fill_diagonal(sim, np.nan)
    return np.nanmean(sim)

############################################
# TSNE PLOT
############################################

def plot_tsne_embeddings(train_matrix, gen_matrix, name, max_train=1500):
    print("Computing t-SNE visualization...")
    if len(train_matrix) > max_train:
        idx = np.random.choice(len(train_matrix), max_train, replace=False)
        train_subset = train_matrix[idx]
    else:
        train_subset = train_matrix

    X = np.vstack([train_subset, gen_matrix])
    labels = np.array(["Train"] * len(train_subset) + ["Generated"] * len(gen_matrix))

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8,8))
    mask_train = labels == "Train"
    mask_gen = labels == "Generated"

    plt.scatter(X_tsne[mask_train,0], X_tsne[mask_train,1], s=6, alpha=0.5, c="gray", label="Training")
    plt.scatter(X_tsne[mask_gen,0], X_tsne[mask_gen,1], s=8, alpha=0.8, c="red", label="Generated")

    plt.legend()
    plt.title(f"t-SNE Embedding Space: {name}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f"{name}_tsne.png"), dpi=300)
    plt.close()

############################################
# IDENTITY PLOT
############################################

def plot_identity(df, name):
    plt.figure()
    sns.histplot(df.pident.fillna(0), bins=50)
    no_hits = df.pident.isna().sum()
    total = len(df)
    plt.xlabel("Sequence identity (%)")
    plt.title(f"{name} (no hits: {no_hits}/{total})")
    plt.savefig(os.path.join(args.results_dir, f"{name}_identity.png"))
    plt.close()

############################################
# EVALUATE
############################################

def evaluate():
    train_ids, train_matrix = load_reference()
    model, batch_converter = load_model()
    summary = []

    if os.path.isfile(args.gen_dir):
        files = [os.path.basename(args.gen_dir)]
        base_dir = os.path.dirname(args.gen_dir)
    else:
        files = os.listdir(args.gen_dir)
        base_dir = args.gen_dir

    for file in files:
        path = os.path.join(base_dir, file)
        if not (path.endswith(".fasta") or path.endswith(".csv")):
            continue

        name = os.path.splitext(file)[0]  # safer filename handling
        print("\nProcessing:", name)

        gen_seqs = load_sequences(path)
        gen_fasta = os.path.join(TMP_DIR, "gen_tmp.fasta")
        write_fasta(gen_seqs, gen_fasta)

        tsv = run_mmseqs(gen_fasta, os.path.join(args.results_dir, name))
        query_ids = [x[0] for x in gen_seqs]
        hits = parse_mmseqs(tsv, query_ids)
        plot_identity(hits, name)

        gen_emb = compute_embeddings(gen_seqs, model, batch_converter)
        gen_matrix = np.vstack([gen_emb[id] for id, _ in gen_seqs])

        if args.save_embeddings:
            emb_path = os.path.join(args.embeddings_dir, f"{name}_embeddings.npy")
            np.save(emb_path, gen_matrix)
            id_path = os.path.join(args.embeddings_dir, f"{name}_ids.csv")
            pd.DataFrame({"id": [id for id, _ in gen_seqs]}).to_csv(id_path, index=False)

        plot_tsne_embeddings(train_matrix, gen_matrix, name)

        nn_sim, nn_distance = embedding_metrics(train_matrix, gen_matrix)
        gen_similarity = generation_diversity(gen_matrix)
        total_count = len(gen_seqs)

        gen_seqs_ppl = random.sample(gen_seqs, min(args.max_ppl_seqs, len(gen_seqs)))
        gen_seqs_ppl = [(id, seq[:args.truncate_length]) for id, seq in gen_seqs_ppl]

        print("Computing pseudo-perplexity...")
        ppl = compute_perplexity(gen_seqs_ppl, model, batch_converter)

        pident_hits = hits.pident.dropna()
        no_hit_fraction = hits.pident.isna().mean()

        buckets = {
            "90-100": ((pident_hits >= 0.9) & (pident_hits <= 1)).mean() if len(pident_hits) > 0 else 0.0,
            "70-90": ((pident_hits >= 0.7) & (pident_hits < 0.9)).mean() if len(pident_hits) > 0 else 0.0,
            "50-70": ((pident_hits >= 0.5) & (pident_hits < 0.7)).mean() if len(pident_hits) > 0 else 0.0,
            "30-50": ((pident_hits >= 0.3) & (pident_hits < 0.5)).mean() if len(pident_hits) > 0 else 0.0,
            "<30": ((pident_hits < 0.3)).mean() if len(pident_hits) > 0 else 0.0
        }

        metrics = {
            "model": name,
            "total_count": total_count,
            "mean_identity": pident_hits.mean() if len(pident_hits) > 0 else 0.0,
            "mean_nn_similarity": nn_sim.mean(),
            "mean_nn_distance": nn_distance.mean(),
            "mean_gen_similarity": gen_similarity,
            "mean_perplexity": ppl.mean(),
            "no_hit": no_hit_fraction,
            **buckets
        }

        summary.append(metrics)

    # Save summary with file locking to safely append
    summary_file = os.path.join(args.results_dir, "summary.csv")
    lock_file = summary_file + ".lock"
    summary_df = pd.DataFrame(summary)

    lock = FileLock(lock_file)
    with lock:
        if os.path.exists(summary_file):
            summary_df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_file, mode='w', header=True, index=False)

    print(summary_df)

############################################
# MAIN
############################################

if args.mode == "prepare":
    if args.train is None:
        raise ValueError("Provide --train for prepare mode")
    prepare_reference()
elif args.mode == "evaluate":
    if args.gen_dir is None:
        raise ValueError("Provide --gen_dir for evaluation")
    evaluate()
