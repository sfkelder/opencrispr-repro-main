import click
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

from grna_modeling.runner import gRNAModelRunner
from grna_modeling.utility.protein import Protein, ProteinBatch
from grna_modeling.utility.data_loaders import (
    load_protein_from_string,
    load_protein_from_csv,
    load_protein_from_fasta,
)

from grna_modeling.utility.protein import Protein, vocabulary

def protein_from_batch(batch, i: int) -> Protein:
    pad = vocabulary.pad_token
    valid = batch.S[i] != pad

    C = batch.C[i][valid]
    S = batch.S[i][valid]

    return Protein(C, S, metadata=batch.metadata[i])

@click.command()

# Single sequence settings
@click.option("--sequence", help="Single protein sequence string.")
@click.option("--sequence-id", 'sequence_id', default='generated_sequence', help="Protein sequence name")

# CSV settings
@click.option("--csv", "csv_file", help="CSV file with protein sequences.")
@click.option("--sequence-col", 'sequence_col', default="sequence", help="Column with protein sequences.")
@click.option("--id-col", "id_col", default=None, help="Column with protein name/id.")

# Fasta settings
@click.option("--fasta", "fasta_file", help="FASTA file with protein sequences.")

# Generation settings
@click.option("--num-samples", 'num_samples', default=5, type=int, help="Number of gRNAs to sample per protein.")
@click.option("--temperature", default=1.0, type=float, help="Sampling temperature, by default 1")
@click.option("--batch-size", 'batch_size', default=1, type=int, help=" Batch size for sampling, by default 1")
@click.option("--max-len", 'max_len', default=300, type=int, help="Maximum length of the generated sequences, by default 300")
@click.option("--silent", default=True, help="Whether to show progress bar, by default True")
@click.option("--ckpt-path", "ckpt_path", type=click.Path(exists=True), required=True, help="Path to model ckpt file")

# Output dir
@click.option("--output", default='./', type=str, help="Output CSV file path, default to current working dir")
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
    ckpt_path
    ):
    """
    Sample gRNAs from a protein or multiple proteins, and write results to CSV.
    """
    runner = gRNAModelRunner(ckpt_path=ckpt_path)
    results = []

    # Load protein(s)
    proteins: List[Protein] = []

    if sequence:
        proteins.append(load_protein_from_string(sequence))
    elif csv_file:
        protein_batch = load_protein_from_csv(csv_file, sequence_col, id_col)
        proteins.extend([protein_from_batch(protein_batch, i) for i in range(protein_batch.size)])
    elif fasta_file:
        protein_batch = load_protein_from_fasta(fasta_file)
        proteins.extend([protein_from_batch(protein_batch, i) for i in range(protein_batch.size)])
    else:
        raise click.UsageError("Please provide --sequence, --csv, or --fasta input.")

    # Loop over each protein sequence
    for idx, protein in enumerate(proteins, 1):
        if sequence:
          seq_id = sequence_id
          protein_seq = sequence
        else: 
          seq_id = protein.metadata.get("id", f"protein_{idx}")
          protein_seq = protein.metadata.get("sequence")
        
        samples = runner.sample(protein, num_samples=num_samples, temperature=temperature, batch_size=batch_size, max_len=max_len, silent=silent)

        # Append to results
        for (tracr, crispr) in samples:
            results.append({
                "protein_id": seq_id,
                "protein": protein_seq,
                "tracrRNA": tracr,
                "crRNA": crispr
            })

    # Write to CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{output}/grna_predictions.csv", index=False)
    click.echo(f"Saved {len(results)} gRNA samples to {output}")


if __name__ == "__main__":
    main()
