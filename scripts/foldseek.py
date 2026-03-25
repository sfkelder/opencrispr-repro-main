import argparse
import subprocess
import pandas as pd
from pathlib import Path

############################################
# ARGPARSE SETTINGS
############################################
parser = argparse.ArgumentParser(description="Run Foldseek and classify CRISPR proteins.")

# Paths
parser.add_argument("--query_dir", required=True, help="Directory with PDB query files")
parser.add_argument("--db", required=True, help="Foldseek database path")
parser.add_argument("--tmp_dir", required=True, help="Temporary directory for Foldseek")
parser.add_argument("--out_dir", required=True, help="Output directory for results")
parser.add_argument("--job_idx", default=None, help="Job index (for parallel jobs)")


# Foldseek thresholds
parser.add_argument("--tm_threshold", type=float, default=0.45, help="TM-score threshold")
parser.add_argument("--prob_threshold", type=float, default=0.8, help="Probability threshold")
parser.add_argument("--rmsd_threshold", type=float, default=6, help="RMSD threshold")
parser.add_argument("--alnlen_threshold", type=int, default=70, help="Alignment length threshold")

args = parser.parse_args()

############################################
# OUTPUT FILE SETUP
############################################
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

suffix = f"_{args.job_idx}" if args.job_idx else ""

out_file = out_dir / f"foldseek_raw{suffix}.tsv"
summary_file = out_dir / f"foldseek_summary{suffix}.tsv"

############################################
# COLUMN NAMES
############################################
columns = [
    "query","target","fident","alnlen","mismatch","gapopen",
    "qstart","qend","tstart","tend","evalue","bits","rmsd",
    "prob","alntmscore"
]

############################################
# DOMAIN DEFINITIONS
############################################
RuvC_cas9 = ['d5axwa1','d4oo8a1','d4cmpa1','d4ogca1','d4cmpb1','d4oo8d1']
RuvC_cas12a = ['d5id6a1']

HNH_cas9 = ['d5axwa2','d4ogca2','d4cmpa2','d4oo8a2','d4cmpb3']

REC_cas9 = ['d4ogca3','d5axwa3','d4oo8a3','d4cmpa3','d4cmpb2','d4oo8d2']

PI_cas9 = ['d4oo8a4','d4cmpa4','d5axwa4','d4ogca4','d4cmpb4','d4oo8d3']

NUC_cas12a = ['d5id6a2']

RuvC = RuvC_cas9 + RuvC_cas12a

domains = {
    "RuvC": RuvC,
    "HNH": HNH_cas9,
    "NUC": NUC_cas12a,
    "REC": REC_cas9,
    "PI": PI_cas9
}

############################################
# RUN FOLDSEEK
############################################
print("Running Foldseek...")
subprocess.run([
    "foldseek","easy-search",
    args.query_dir,
    args.db,
    str(out_file),
    args.tmp_dir,
    "--format-output",
    ",".join(columns)
], check=True)

############################################
# LOAD RESULTS
############################################
df = pd.read_csv(out_file, sep="\t", names=columns)

############################################
# APPLY STRUCTURAL FILTERS
############################################
df = df[
    (df["alntmscore"] > args.tm_threshold) &
    (df["prob"] > args.prob_threshold) &
    (df["rmsd"] < args.rmsd_threshold) &
    (df["alnlen"] > args.alnlen_threshold)
]

############################################
# ASSIGN DOMAINS
############################################
def assign_domain(target):
    for domain, ids in domains.items():
        if target in ids:
            return domain
    return None

df["domain"] = df["target"].apply(assign_domain)
df = df[df["domain"].notnull()]

############################################
# KEEP BEST HIT PER QUERY + DOMAIN
############################################
best_hits = (
    df.sort_values("alntmscore", ascending=False)
      .groupby(["query","domain"])
      .first()
      .reset_index()
)

############################################
# BUILD DOMAIN MATRIX
############################################
matrix = best_hits.pivot(index="query", columns="domain", values="alntmscore")
matrix = matrix.fillna(0)

# Ensure all domains exist
for d in domains.keys():
    if d not in matrix.columns:
        matrix[d] = 0

############################################
# CONVERT TO BOOLEAN PRESENCE
############################################
presence = matrix > 0

############################################
# CLASSIFY PROTEINS
############################################
def classify(row):
    if row["HNH"]:
        return "Cas9_candidate"
    elif row["NUC"] or (row["RuvC"] and row["REC"]):
        return "Cas12_candidate"
    elif row["RuvC"]:
        return "RuvC_nuclease"
    else:
        return "unknown"

presence["classification"] = presence.apply(classify, axis=1)

# Merge TM-scores with classification
result = matrix.copy()
result["classification"] = presence["classification"]

############################################
# SAVE RESULTS
############################################
result.to_csv(summary_file, sep="\t")

print(f"Finished.")
print(f"Foldseek results: {out_file}")
print(f"Summary results: {summary_file}")
