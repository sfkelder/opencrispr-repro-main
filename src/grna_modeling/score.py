import click
import pandas as pd
from pathlib import Path
from typing import Dict, List

from grna_modeling.runner import gRNAModelRunner
from grna_modeling.utility.protein import Protein, NucleicAcidBatch


@click.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True),help="Input CSV containing protein + RNA columns.")
@click.option("--protein-col", default='protein', type=str,help="Column name for protein sequence.")
@click.option("--tracr-col", default='tracrRNA', type=str, help="Column name for tracrRNA.")
@click.option("--crispr-col", default='crRNA', type=str, help="Column name for crRNA.")
@click.option("--output", default='./', type=click.Path(exists=True),help="Output CSV path.")
@click.option("--rna-batch-size", default=16, type=int,help="Number of RNA pairs to score per batch (per protein).")
@click.option("--ckpt-path", 'ckpt_path', required=True, type=click.Path(exists=True), help="Number of RNA pairs to score per batch (per protein).")
def main(input_csv, protein_col, tracr_col, crispr_col, output, rna_batch_size, ckpt_path):
    """
    Score tracrRNA + crRNA pairs against proteins from a CSV file.

    The script:
    - Groups rows by protein
    - Embeds each protein once
    - Batches RNA scoring within each protein
    - Appends normalized log-likelihood scores to the CSV
    """

    # ------------------------
    # Load data
    # ------------------------
    df = pd.read_csv(input_csv)

    for col in [protein_col, tracr_col, crispr_col]:
        if col not in df.columns:
            raise click.ClickException(f"Missing column: {col}")

    # internal working columns
    df["__protein__"] = df[protein_col].astype(str)
    df["__tracr__"] = df[tracr_col].astype(str)
    df["__crispr__"] = df[crispr_col].astype(str)

    # ------------------------
    # Load model runner
    # ------------------------
    click.echo("Loading gRNA model runner...")
    runner = gRNAModelRunner(ckpt_path=ckpt_path)

    # storage for scores
    all_scores: Dict[int, float] = {}

    # ------------------------
    # Group by protein
    # ------------------------
    grouped = df.groupby("__protein__", sort=False)

    for protein_seq, group in grouped:

        click.echo(f"Scoring protein group with {len(group)} rows")

        # build Protein object once
        protein = Protein.from_sequence(protein_seq)

        indices = group.index.tolist()
        tracrs = group["__tracr__"].tolist()
        crisprs = group["__crispr__"].tolist()

        # ------------------------
        # Batch RNAs safely
        # ------------------------
        for start in range(0, len(indices), rna_batch_size):
            end = start + rna_batch_size

            idx_batch = indices[start:end]
            tracr_batch_seqs = tracrs[start:end]
            crispr_batch_seqs = crisprs[start:end]

            tracr_batch = NucleicAcidBatch.from_sequences(tracr_batch_seqs)
            crispr_batch = NucleicAcidBatch.from_sequences(crispr_batch_seqs)

            scores = runner.score(
                tracr_batch=tracr_batch,
                crispr_batch=crispr_batch,
                protein_batch=protein,
            )

            for idx, score in zip(idx_batch, scores):
                all_scores[idx] = float(score)

    # ------------------------
    # Attach scores
    # ------------------------
    df["score"] = pd.Series(all_scores).sort_index()

    # remove internal columns
    df.drop(columns=["__protein__", "__tracr__", "__crispr__"], inplace=True)

    # ------------------------
    # Save output
    # ------------------------
    df.to_csv(f"{output}/grna_predictions.csv", index=False)
    click.echo(f"Saved scored CSV to {output}")


if __name__ == "__main__":
    main()
