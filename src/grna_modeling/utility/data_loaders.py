from typing import List, Optional, Union
import pandas as pd
from Bio import SeqIO

from grna_modeling.utility.protein import Protein, ProteinBatch, NucleicAcidBatch


# -----------------------------
# Protein loaders
# -----------------------------
def load_protein_from_string(sequence: str) -> Protein:
    """Load a single protein from a string sequence."""
    return Protein.from_sequence(sequence)

def load_protein_from_fasta(fasta_file: str, record_id: Optional[str] = None) -> ProteinBatch:
    """Load protein(s) from a FASTA file. Returns a ProteinBatch."""
    return ProteinBatch.from_fasta(fasta_file, record_id)

def load_protein_from_csv(
    csv_file: str,
    sequence_col: str = "sequence",
    id_col: Optional[str] = None,
) -> ProteinBatch:
    """Load protein sequences from a CSV file. Returns a ProteinBatch."""

    df = pd.read_csv(csv_file)

    proteins = []
    for i, row in df.iterrows():
        seq = str(row[sequence_col])
        protein = Protein.from_sequence(seq)

        protein.metadata["sequence"] = seq
        protein.metadata["id"] = (
            str(row[id_col]) if id_col and id_col in df.columns else f"row_{i}"
        )

        proteins.append(protein)

    return ProteinBatch.from_proteins(proteins)