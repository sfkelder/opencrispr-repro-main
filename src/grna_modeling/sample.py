import click
import pandas as pd
from pathlib import Path
from typing import List
import re
import hashlib

from grna_modeling.runner import gRNAModelRunner
from grna_modeling.utility.protein import Protein, ProteinBatch, vocabulary
from grna_modeling.utility.data_loaders import (
    load_protein_from_string,
    load_protein_from_csv,
    load_protein_from_fasta,
)


def protein_from_batch(batch, i: int) -> Protein:
    pad = vocabulary.pad_token
    valid = batch.S[i] != pad

    C = batch.C[i][valid]
    S = batch.S[i][valid]

    return Protein(C, S, metadata=batch.metadata[i])


@click.command()

# Single sequence settings
@click.option("--sequence", help="Single protein sequence string.")
@click.option("--sequence-id", "sequence_id", default="generated_sequence", help="Protein sequence name")

# CSV settings
@click.option("--csv", "csv_file", type=click.Path(exists=True), help="CSV file with protein sequences.")
@click.option("--sequence-col", "sequence_col", default="sequence", help="Column with protein sequences.")
@click.option("--id-col", "id_col", default=None, help="Column with protein name/id.")

# Fasta settings
@click.option("--fasta", "fasta_file", type=click.Path(exists=True), help="FASTA file with protein sequences.")

# Generation settings
@click.option("--num-samples", "num_samples", default=5, type=int, help="Number of gRNAs to sample per protein.")
@click.option("--temperature", default=1.0, type=float, help="Sampling temperature")
@click.option("--batch-size", "batch_size", default=1, type=int, help="Batch size for sampling")
@click.option("--max-len", "max_len", default=300, type=int, help="Maximum length of generated sequences")
@click.option("--silent", is_flag=True, help="Disable progress output")
@click.option("--ckpt-path", "ckpt_path", type=click.Path(exists=True), required=True, help="Path to model checkpoint")

# Output
@click.option("--output", default="./", type=str, help="Output directory")
def main(
    sequence,
    sequence_id,
    csv_file,
    fasta_file,
    num_samples,
    temperature,
    batch_size,
    max_len,
    output,
    silent,
    sequence_col,
    id_col,
    ckpt_path,
):
    """
    Sample gRNAs from protein sequences and write results to CSV.
    """

    # Validate exactly one input source
    inputs = [sequence, csv_file, fasta_file]
    if sum(x is not None for x in inputs) != 1:
        raise click.UsageError("Provide exactly one of --sequence, --csv, or --fasta.")

    runner = gRNAModelRunner(ckpt_path=ckpt_path)
    results = []

    # Load proteins
    proteins: List[Protein] = []

    if sequence:
        proteins.append(load_protein_from_string(sequence))

    elif csv_file:
        protein_batch = load_protein_from_csv(csv_file, sequence_col, id_col)
        proteins.extend([protein_from_batch(protein_batch, i) for i in range(protein_batch.size)])

    elif fasta_file:
        protein_batch = load_protein_from_fasta(fasta_file)
        proteins.extend([protein_from_batch(protein_batch, i) for i in range(protein_batch.size)])

    # Process proteins
    for idx, protein in enumerate(proteins, 1):

        if sequence:
            seq_id = sequence_id
            protein_seq = sequence
        else:
            seq_id = protein.metadata.get("id") or f"protein_{idx}"
            protein_seq = protein.metadata.get("sequence") or "".join(protein.S)

        samples = runner.sample(
            protein,
            num_samples=num_samples,
            temperature=temperature,
            batch_size=batch_size,
            max_len=max_len,
            silent=silent,
        )

        for (tracr, crispr) in samples:
            # Generate a hash-based unique ID
            hash_input = f"{seq_id}_{tracr}_{crispr}".encode("utf-8")
            grna_id = hashlib.sha256(hash_input).hexdigest()[:12]  # first 12 chars

            results.append({
                "grna_id": grna_id,
                "protein_id": seq_id,
                "protein": protein_seq,
                "tracrRNA": tracr,
                "crRNA": crispr,
            })

    # Determine output filename
    if sequence:
        base_name = sequence_id
    elif csv_file:
        base_name = Path(csv_file).stem
    elif fasta_file:
        base_name = Path(fasta_file).stem

    # Clean filename
    base_name = re.sub(r"[^\w]+", "_", base_name).strip("_")

    # Ensure output directory exists
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{base_name}_grna_predictions.csv"

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    click.echo(f"Saved {len(results)} gRNA samples to {output_file}")


if __name__ == "__main__":
    main()