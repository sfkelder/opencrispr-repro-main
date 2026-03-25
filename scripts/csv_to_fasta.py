import os
import argparse
import pandas as pd
import re
from tqdm import tqdm


def make_safe_filename(name):
    """Convert string to a safe filename."""
    name = str(name)
    return re.sub(r"[^\w\-.]", "_", name)


def wrap_sequence(seq, width=60):
    """Wrap sequence to fixed line width (FASTA standard)."""
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def process_csv(
    csv_path,
    output_root,
    sequence_column,
    id_column=None,
    nrows=None,
    single_file=False,
    trim_left=0,
    trim_right=0,
):
    file = os.path.basename(csv_path)
    folder_name = os.path.splitext(file)[0]

    df = pd.read_csv(csv_path, nrows=nrows, encoding="utf-8")

    if sequence_column not in df.columns:
        raise ValueError(f"{sequence_column} not found in {csv_path}")

    use_id_column = id_column and id_column in df.columns

    # 🔹 Case 1: write everything to a single FASTA file
    if single_file:
        output_path = os.path.join(output_root, f"{folder_name}.fasta")

        seen_ids = set()

        with open(output_path, "w") as out_f:
            for row in tqdm(
                df.itertuples(index=True),
                total=len(df),
                desc=f"Processing {file}",
            ):
                row_dict = row._asdict()
                seq_val = row_dict[sequence_column]

                if pd.isna(seq_val):
                    continue

                seq = str(seq_val)

                # ✂️ Apply trimming
                if trim_left > 0:
                    seq = seq[trim_left:]
                if trim_right > 0:
                    seq = seq[:-trim_right] if trim_right < len(seq) else ""

                if not seq:
                    continue

                if use_id_column:
                    seq_id = str(row_dict[id_column])
                else:
                    seq_id = f"seq_{row.Index + 1}"

               #  if seq_id in seen_ids:
               #     seq_id = f"{seq_id}_{row.Index}"
               #  seen_ids.add(seq_id)

                out_f.write(f">{seq_id}\n{wrap_sequence(seq)}\n")

        print(f"Processed {csv_path} → {output_path}")

    # 🔹 Case 2: one FASTA per sequence
    else:
        csv_output_folder = os.path.join(output_root, folder_name)
        os.makedirs(csv_output_folder, exist_ok=True)

        for row in tqdm(
            df.itertuples(index=True),
            total=len(df),
            desc=f"Processing {file}",
        ):
            row_dict = row._asdict()
            seq_val = row_dict[sequence_column]

            if pd.isna(seq_val):
                continue

            seq = str(seq_val)

            # ✂️ Apply trimming
            if trim_left > 0:
                seq = seq[trim_left:]
            if trim_right > 0:
                seq = seq[:-trim_right] if trim_right < len(seq) else ""

            if not seq:
                continue

            if use_id_column:
                seq_id = str(row_dict[id_column])
            else:
                seq_id = f"seq_{row.Index + 1}"

            safe_id = make_safe_filename(seq_id)[:100]
            fasta_path = os.path.join(csv_output_folder, f"{safe_id}.fasta")

            # Prevent overwriting duplicate IDs
            counter = 1
            original_path = fasta_path
            while os.path.exists(fasta_path):
                fasta_path = os.path.join(
                    csv_output_folder,
                    f"{safe_id}_{counter}.fasta"
                )
                counter += 1

            with open(fasta_path, "w") as f:
                f.write(f">{seq_id}\n{wrap_sequence(seq)}\n")

        print(f"Processed {csv_path} → {csv_output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert sequences from CSV files to FASTA files."
    )

    parser.add_argument(
        "-i", "--input_path",
        help="CSV file or directory containing CSV files",
        required=True,
    )

    parser.add_argument(
        "-o", "--output_folder",
        default=None,
        help="Output folder (default: same as input location)",
    )

    parser.add_argument(
        "-s", "--seq_col",
        default="sequence",
        help="Column containing sequences (default: sequence)",
    )

    parser.add_argument(
        "--id_col",
        default=None,
        help="Column containing sequence IDs, if not provided row ID is used",
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Read only first N rows (useful for testing)",
    )

    parser.add_argument(
        "--single_file",
        action="store_true",
        help="Write all sequences to a single FASTA file instead of separate files",
    )

    # ✂️ NEW ARGUMENTS
    parser.add_argument(
        "--trim_left",
        type=int,
        default=0,
        help="Number of characters to trim from the start of the sequence",
    )

    parser.add_argument(
        "--trim_right",
        type=int,
        default=0,
        help="Number of characters to trim from the end of the sequence",
    )

    args = parser.parse_args()

    input_path = args.input_path

    if args.output_folder:
        output_folder = args.output_folder
    else:
        output_folder = (
            input_path
            if os.path.isdir(input_path)
            else os.path.dirname(input_path) or "."
        )

    os.makedirs(output_folder, exist_ok=True)

    # Case 1: single CSV
    if os.path.isfile(input_path):

        if not input_path.endswith(".csv"):
            raise ValueError("Input file must be a CSV")

        process_csv(
            input_path,
            output_folder,
            args.seq_col,
            args.id_col,
            args.nrows,
            args.single_file,
            args.trim_left,
            args.trim_right,
        )

    # Case 2: directory of CSVs
    elif os.path.isdir(input_path):

        csv_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".csv")
        ]

        if not csv_files:
            raise ValueError("No CSV files found in directory")

        for csv_path in csv_files:
            process_csv(
                csv_path,
                output_folder,
                args.seq_col,
                args.id_col,
                args.nrows,
                args.single_file,
                args.trim_left,
                args.trim_right,
            )

    else:
        raise ValueError("Input path must be a file or directory")


if __name__ == "__main__":
    main()
