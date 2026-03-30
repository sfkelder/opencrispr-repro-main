import click
import pandas as pd
from pathlib import Path
from typing import Dict

from grna_modeling.runner import gRNAModelRunner
from grna_modeling.utility.protein import Protein, NucleicAcidBatch
from grna_modeling.utility import vocabulary


@click.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True),
              help="Input CSV containing protein + RNA columns.")
@click.option("--protein-col", default='protein', type=str,
              help="Column name for protein sequence.")
@click.option("--tracr-col", default='tracrRNA', type=str,
              help="Column name for tracrRNA.")
@click.option("--crispr-col", default='crRNA', type=str,
              help="Column name for crRNA.")
@click.option("--output", default='./', type=click.Path(exists=True),
              help="Output CSV path.")
@click.option("--rna-batch-size", default=16, type=int,
              help="Number of RNA pairs to score per batch (per protein).")
@click.option("--ckpt-path", 'ckpt_path', required=True, type=click.Path(exists=True),
              help="Path to model checkpoint.")
def main(input_csv, protein_col, tracr_col, crispr_col,
         output, rna_batch_size, ckpt_path):
    """
    Score tracrRNA + crRNA pairs against proteins from a CSV file.
    """

    # ------------------------
    # Load data
    # ------------------------
    df = pd.read_csv(input_csv)

    for col in [protein_col, tracr_col, crispr_col]:
        if col not in df.columns:
            raise click.ClickException(f"Missing column: {col}")

    df["__protein__"] = df[protein_col].astype(str)
    df["__tracr__"] = df[tracr_col].fillna("").astype(str)
    df["__crispr__"] = df[crispr_col].fillna("").astype(str)

    # ------------------------
    # Load model runner
    # ------------------------
    click.echo("Loading gRNA model runner...")
    runner = gRNAModelRunner(ckpt_path=ckpt_path)

    all_scores: Dict[int, float] = {}

    # ------------------------
    # Group by protein
    # ------------------------
    grouped = df.groupby("__protein__", sort=False)

    for protein_seq, group in grouped:

        click.echo(f"Scoring protein group with {len(group)} rows")

        protein = Protein.from_sequence(protein_seq)

        indices = group.index.tolist()
        tracrs = group["__tracr__"].tolist()
        crisprs = group["__crispr__"].tolist()

        # ------------------------
        # Batch RNAs
        # ------------------------
        for start in range(0, len(indices), rna_batch_size):
            end = start + rna_batch_size

            idx_batch = indices[start:end]
            tracr_batch_seqs = tracrs[start:end]
            crispr_batch_seqs = crisprs[start:end]

            # 🔧 FIX: handle empty sequences like training (add sentinels)
            tracr_batch_seqs = [
                s if s != "" else vocabulary.TRACR_START_SENT + vocabulary.TRACR_END_SENT
                for s in tracr_batch_seqs
            ]

            crispr_batch_seqs = [
                s if s != "" else vocabulary.CRISPR_START_SENT + vocabulary.CRISPR_END_SENT
                for s in crispr_batch_seqs
            ]

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
    df["score"] = df.index.map(all_scores)

    df.drop(columns=["__protein__", "__tracr__", "__crispr__"], inplace=True)

    # ------------------------
    # Save output
    # ------------------------
    output_path = Path(output) / "grna_predictions.csv"
    df.to_csv(output_path, index=False)

    click.echo(f"Saved scored CSV to {output_path}")


if __name__ == "__main__":
    main()