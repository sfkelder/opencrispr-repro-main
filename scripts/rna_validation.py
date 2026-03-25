from Bio import SeqIO
import os
import csv
import glob
from collections import defaultdict

# -------------------------------
# LOAD SEQUENCES
# -------------------------------
def load_sequences(fasta_file):
    sequences = defaultdict(list)
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id].append(str(record.seq))
    return dict(sequences)

def load_sequences_from_csv(csv_file, id_col, seq_col):
    sequences = defaultdict(list)
    rows = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row[id_col]
            seq = row[seq_col]
            sequences[seq_id].append(seq)
            rows.append(row)
    return dict(sequences), rows, reader.fieldnames

# -------------------------------
# PARSE BASEPAIRS FILE
# -------------------------------
def parse_basepairs_file(file_path, prob_threshold=0.5):
    name = os.path.basename(file_path).split("_basepairs")[0]
    pairs = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                i, j, p = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                continue
            if p >= prob_threshold:
                pairs.append((i, j))
    return name, pairs

# -------------------------------
# FIND STEMS
# -------------------------------
def find_stems(pairs, min_stem_length=3):
    pairs = sorted(pairs)
    used = set()
    stems = []
    pair_set = set(pairs)
    for (i, j) in pairs:
        if (i, j) in used:
            continue
        stem = [(i, j)]
        used.add((i, j))
        k = 1
        while (i + k, j - k) in pair_set:
            stem.append((i + k, j - k))
            used.add((i + k, j - k))
            k += 1
        if len(stem) >= min_stem_length:
            stems.append(stem)
    return stems

# -------------------------------
# VALIDATE REPEAT
# -------------------------------
def validate_repeat(
    sequence,
    stems,
    min_len,
    max_len,
    min_valid_stem_length,
    max_valid_stem_length,
    min_loop_length,
    max_loop_length,
    max_competing_stems,
    near_5prime_fraction
):
    seq_len = len(sequence)
    length_ok = min_len <= seq_len <= max_len

    # Longest stem
    if stems:
        longest_stem = max(stems, key=len)
        stem_length_ok = min_valid_stem_length <= len(longest_stem) <= max_valid_stem_length

        # Loop length: distance between end of 5' arm and start of 3' arm
        loop_length = (min(j for i,j in longest_stem) - max(i for i,j in longest_stem)) - 1
        loop_ok = min_loop_length <= loop_length <= max_loop_length

        first_i = min(i for i,j in longest_stem)
        stem_near_5prime = first_i <= near_5prime_fraction * seq_len
    else:
        longest_stem = []
        stem_length_ok = False
        loop_ok = False
        stem_near_5prime = False

    # Competing stems
    strong_stems = [s for s in stems if len(s) >= min_valid_stem_length]
    no_competing_stems = len(strong_stems) <= max_competing_stems

    checks = {
        "length_ok": length_ok,
        "stem_length_ok": stem_length_ok,
        "loop_ok": loop_ok,
        "stem_near_5prime": stem_near_5prime,
        "no_competing_stems": no_competing_stems,
    }

    valid = all(checks.values())
    return valid, checks

# -------------------------------
# PROCESS FOLDER
# -------------------------------
def process_folder(
    folder_path,
    fasta_file=None,
    csv_file=None,
    id_col=None,
    seq_col=None,
    output_csv="validation_results.csv",
    min_len=18,
    max_len=24,
    min_stem_length=3,
    max_valid_stem_length=7,
    prob_threshold=0.5,
    min_valid_stem_length=5,
    min_loop_length=4,
    max_loop_length=8,
    max_competing_stems=1,
    near_5prime_fraction=0.25
):
    # Load sequences
    if csv_file:
        seq_dict, original_rows, original_fields = load_sequences_from_csv(csv_file, id_col, seq_col)
        use_csv = True
    elif fasta_file:
        seq_dict = load_sequences(fasta_file)
        use_csv = False
        original_rows = []
        original_fields = []
    else:
        raise ValueError("Provide either --fasta_file or --csv_file")

    files = glob.glob(os.path.join(folder_path, "*basepairs*"))
    if not files:
        print("No basepairs files found.")
        return

    results = []
    validation_fields = [
        "valid/invalid",
        "length_ok",
        "stem_length_ok",
        "loop_ok",
        "stem_near_5prime",
        "no_competing_stems",
    ]

    for file_path in files:
        try:
            name, pairs = parse_basepairs_file(file_path, prob_threshold=prob_threshold)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

        seq_list = seq_dict.get(name)
        if seq_list is None:
            print(f"Warning: sequence not found for {name}")
            continue
        if len(seq_list) > 1:
            print(f"Warning: multiple sequences for {name}, using first")
        sequence = seq_list[0]

        try:
            stems = find_stems(pairs, min_stem_length=min_stem_length)
            valid, checks = validate_repeat(
                sequence,
                stems,
                min_len,
                max_len,
                min_valid_stem_length,
                max_valid_stem_length,
                min_loop_length,
                max_loop_length,
                max_competing_stems,
                near_5prime_fraction
            )
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

        result = {id_col if csv_file else "name": name,
                  "sequence": sequence,
                  "valid/invalid": "VALID" if valid else "INVALID"}
        for field in validation_fields[1:]:
            result[field] = "PASS" if checks.get(field, False) else "FAIL"
        results.append(result)

    # Output
    if use_csv:
        result_map = {r[id_col]: r for r in results}
        updated_rows = []
        for row in original_rows:
            seq_id = row[id_col]
            res = result_map.get(seq_id)
            if res:
                for field in validation_fields:
                    row[field] = res.get(field, "")
            else:
                for field in validation_fields:
                    row[field] = ""
            updated_rows.append(row)
        fieldnames = original_fields + validation_fields
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
    else:
        fieldnames = ["name","sequence"] + validation_fields
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    total = len(results)
    valid_count = sum(1 for r in results if r["valid/invalid"] == "VALID")
    print(f"\nValidation completed. Results saved to: {output_csv}")
    print(f"Valid sequences: {valid_count}/{total}")

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate synthetic Cas crRNA repeats with all rules")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--fasta_file", help="Input FASTA file")
    parser.add_argument("--csv_file", help="Input CSV file")
    parser.add_argument("--id_col", help="Column name for sequence ID (CSV)")
    parser.add_argument("--seq_col", help="Column name for sequence (CSV)")
    parser.add_argument("-o","--output", default="validation_results.csv")

    # Validation parameters
    parser.add_argument("--min_len", type=int, default=18)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--min_stem_length", type=int, default=3)
    parser.add_argument("--min_valid_stem_length", type=int, default=5)
    parser.add_argument("--max_valid_stem_length", type=int, default=7)
    parser.add_argument("--min_loop_length", type=int, default=4)
    parser.add_argument("--max_loop_length", type=int, default=8)
    parser.add_argument("--max_competing_stems", type=int, default=1)
    parser.add_argument("--near_5prime_fraction", type=float, default=0.25)
    parser.add_argument("--prob_threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.csv_file and (not args.id_col or not args.seq_col):
        parser.error("--csv_file requires --id_col and --seq_col")

    process_folder(
        folder_path=args.input_path,
        fasta_file=args.fasta_file,
        csv_file=args.csv_file,
        id_col=args.id_col,
        seq_col=args.seq_col,
        output_csv=args.output,
        min_len=args.min_len,
        max_len=args.max_len,
        min_stem_length=args.min_stem_length,
        min_valid_stem_length=args.min_valid_stem_length,
        max_valid_stem_length=args.max_valid_stem_length,
        min_loop_length=args.min_loop_length,
        max_loop_length=args.max_loop_length,
        max_competing_stems=args.max_competing_stems,
        near_5prime_fraction=args.near_5prime_fraction,
        prob_threshold=args.prob_threshold
    )
