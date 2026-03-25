import pandas as pd
from pathlib import Path
import argparse
import re
from tqdm import tqdm

tqdm.pandas()  # enables progress_apply

def clean_row(seq):
    """
    Clean a single sequence:
    - Remove sequences with repeated 1s or 2s
    - Remove all remaining 1s and 2s
    - Return None if the sequence becomes empty or invalid
    """
    seq = str(seq)
    if re.search(r"(?:1{2,}|2{2,})", seq):
        return None
    seq = re.sub(r"[12]", "", seq)
    if seq == "":
        return None
    return seq

def clean_dataframe(df, seq_col):
    # Apply cleaning with a progress bar
    df[seq_col] = df[seq_col].progress_apply(clean_row)
    # Drop invalid/empty sequences
    df = df.dropna(subset=[seq_col])
    return df

def process_file(csv_file: Path, output_folder: Path, seq_col: str):
    print(f"Processing: {csv_file.name}")

    df = pd.read_csv(csv_file)

    if seq_col not in df.columns:
        print(f"Column '{seq_col}' not found in {csv_file.name}")
        return

    df_clean = clean_dataframe(df, seq_col)

    output_file = output_folder / csv_file.name
    df_clean.to_csv(output_file, index=False)

    print(f"Saved cleaned file → {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Clean CSV sequences by removing rows with repeated 1s/2s and stripping remaining 1s and 2s."
    )

    parser.add_argument(
        "-i", "--input_path",
        required=True,
        help="Input CSV file or folder containing CSV files"
    )

    parser.add_argument(
        "-o", "--output_folder",
        required=True,
        help="Output folder for cleaned CSV files"
    )

    parser.add_argument(
        "-s", "--seq_col",
        default="sequence",
        help="Column name containing sequences (default: sequence)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_folder)
    seq_col = args.seq_col

    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        process_file(input_path, output_path, seq_col)

    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))

        if not csv_files:
            print("No CSV files found in folder.")
        else:
            for csv_file in tqdm(csv_files, desc="Processing CSV files"):
                process_file(csv_file, output_path, seq_col)

    else:
        print("Invalid input path.")

if __name__ == "__main__":
    main()
