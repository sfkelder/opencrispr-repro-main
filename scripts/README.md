## Usage
This repository includes several scripts for data proccessing to allow for the creation of training and generation piplines. Please follow the setup setps in the main [`README`](../README.md).

### clean_generations
For each sequence in the specified column:

1. Removes invalid sequences:
Any sequence containing repeated "1" or "2" tokens (e.g., "11", "222")
2. Cleans valid sequences:
Removes all remaining "1" and "2" characters
3. Filters results:
Drops sequences that become empty after cleaning

```
python clean_generations.py --input_path PATH_TO_INPUT \
                          --output_folder PATH_TO_OUTPUT \
                          [--seq_col COLUMN_NAME]
```

| Argument                | Type              | Default      | Description                                                     |
| ----------------------- | ----------------- | ------------ | --------------------------------------------------------------- |
| `-i`, `--input_path`    | file or directory | —            | **Required.** Input CSV file or directory containing CSV files. |
| `-o`, `--output_folder` | directory         | —            | **Required.** Output directory for cleaned CSV files.           |
| `-s`, `--seq_col`       | string            | `"sequence"` | Column name containing sequences to clean.                      |

**Output**
Cleaned files are saved with the same filename and format in the specified output directory.

> [!WARNING]
> If models are improperley trained, they could generate seuences with repeated BOS and
> EOS. This script filters out all sequences with repeated BOS and EOS tokens.  

### csv_to_fasta
This script converts protein or nucleotide sequences stored in CSV files into FASTA format, supporting both:

 - Single combined FASTA file
 - One FASTA file per sequence

It also supports sequence trimming, custom IDs, and batch processing of multiple CSV files.


```
python csv_to_fasta.py --input_path PATH \
                       [--output_folder PATH] \
                       [--seq_col COLUMN] \
                       [--id_col COLUMN] \
                       [--nrows N] \
                       [--single_file] \
                       [--trim_left N] \
                       [--trim_right N]
```

| Argument                | Type              | Default        | Description                                                            |
| ----------------------- | ----------------- | -------------- | ---------------------------------------------------------------------- |
| `-i`, `--input_path`    | file or directory | —              | **Required.** Input CSV file or directory containing CSV files.        |
| `-o`, `--output_folder` | directory         | input location | Output directory for FASTA files.                                      |
| `-s`, `--seq_col`       | string            | `"sequence"`   | Column containing sequences.                                           |
| `--id_col`              | string            | `None`         | Column containing sequence IDs. If not provided, row indices are used. |
| `--nrows`               | int               | `None`         | Process only the first *N* rows (useful for testing).                  |
| `--single_file`         | flag              | `False`        | Write all sequences into a single FASTA file.                          |
| `--trim_left`           | int               | `0`            | Number of characters to remove from the start of each sequence.        |
| `--trim_right`          | int               | `0`            | Number of characters to remove from the end of each sequence.          |

**Output**
1. `--single_file` provides one fasta file with all sequences.
2. Defaults to ceating file per sequence containing only one sequence. 

### foldseek
Before using this file, check setup in main [`README`](../README.md).
This script runs Foldseek structural searches on protein structures (PDB files) and classifies them into CRISPR-associated protein types (e.g., Cas9, Cas12) based on domain-level structural matches.

```
python foldseek.py \
  --query_dir PATH_TO_PDBS \
  --db PATH_TO_FOLDSEEK_DB \
  --tmp_dir PATH_TO_TMP \
  --out_dir PATH_TO_OUTPUT \
  [--job_idx IDX] \
  [--tm_threshold FLOAT] \
  [--prob_threshold FLOAT] \
  [--rmsd_threshold FLOAT] \
  [--alnlen_threshold INT]
```

**Required**
| Argument      | Type      | Description                                          |
| ------------- | --------- | ---------------------------------------------------- |
| `--query_dir` | directory | Folder containing query PDB files.                   |
| `--db`        | path      | Foldseek database path.                              |
| `--tmp_dir`   | directory | Temporary directory for Foldseek intermediate files. |
| `--out_dir`   | directory | Output directory for results.                        |

**Optional**
| Argument             | Type   | Default | Description                                                 |
| -------------------- | ------ | ------- | ----------------------------------------------------------- |
| `--job_idx`          | string | `None`  | Job index for parallel runs (appended to output filenames). |
| `--tm_threshold`     | float  | `0.45`  | Minimum TM-score threshold.                                 |
| `--prob_threshold`   | float  | `0.8`   | Minimum Foldseek probability.                               |
| `--rmsd_threshold`   | float  | `6`     | Maximum RMSD threshold.                                     |
| `--alnlen_threshold` | int    | `70`    | Minimum alignment length.                                   |

**Output**
| Column                            | Description             |
| --------------------------------- | ----------------------- |
| `query`                           | Input protein structure |
| `RuvC`, `HNH`, `NUC`, `REC`, `PI` | TM-scores per domain    |
| `classification`                  | Predicted protein class |

**Clasification logic**
| Condition                   | Classification    |
| --------------------------- | ----------------- |
| HNH domain present          | `Cas9_candidate`  |
| NUC present OR (RuvC + REC) | `Cas12_candidate` |
| Only RuvC present           | `RuvC_nuclease`   |
| None detected               | `unknown`         |

> [!IMPORTANT]
> Domains are matched by domainIDs defined inside `foldseek.py`. These are currently only for Cas12a and Cas9. Check this [Google Colab](https://colab.research.google.com/drive/1D4BHCwhdIhPdvVwIVba0SXwDCSYBQI81?usp=sharing) for ID conversion.

### mmseq
Before using this file, check setup in main [`README`](../README.md).
This script evaluates generated protein sequences against a training dataset using:

 - Sequence similarity (MMseqs2)
 - Embedding similarity (ESM2)
 - Diversity metrics
 - Pseudo-perplexity (ESM-based)
 - Visualization (t-SNE, identity distributions)

It supports a two-step workflow:
```
prepare → compute training embeddings → build MMseqs DB
evaluate → compare generated sequences → compute metrics → visualize
```

1. Prepare reference embeddings
    ```
    python mmseq.py \
        --mode prepare \
        --train PATH_TO_TRAIN_DATA \
        --format fasta \
        --tmp_dir ./tmp
    ```

2. Evaluate generated sequences
    ```
    python mmseq.py \
        --mode evaluate \
        --gen_dir PATH_TO_GENERATED \
        --reference_dir ./train_reference \
        --results_dir ./results \
        --tmp_dir ./tmp \
        [--save_embeddings]
    ```


**Core arguments**
| Argument    | Type                   | Description                    |
| ----------- | ---------------------- | ------------------------------ |
| `--mode`    | `prepare` / `evaluate` | Select pipeline stage          |
| `--tmp_dir` | directory              | Temporary directory for MMseqs |


**Prepare mode**
| Argument  | Description                             |
| --------- | --------------------------------------- |
| `--train` | Path to training dataset (FASTA or CSV) |


**Evaluation mode**
| Argument          | Description                                      |
| ----------------- | ------------------------------------------------ |
| `--gen_dir`       | Directory or file containing generated sequences |
| `--reference_dir` | Directory with prepared reference data (Database)|
| `--results_dir`   | Output directory for evaluation results          |

**Input formating**
| Argument    | Default    | Description                     |
| ----------- | ---------- | ------------------------------- |
| `--format`  | `fasta`    | Input format (`fasta` or `csv`) |
| `--seq_col` | `sequence` | Sequence column (CSV only)      |
| `--id_col`  | `id`       | ID column (CSV only)            |


**Embedding options**
| Argument            | Description                            |
| ------------------- | -------------------------------------- |
| `--save_embeddings` | Save embeddings of generated sequences |
| `--embeddings_dir`  | Directory for saved embeddings         |


**Perplexity options**
| Argument            | Default | Description                       |
| ------------------- | ------- | --------------------------------- |
| `--max_ppl_seqs`    | `200`   | Max sequences used for perplexity |
| `--truncate_length` | `400`   | Max sequence length               |
| `--mask_batch_size` | `128`   | Masked tokens per batch           |

> [!TIP]
> For large sequences, perplexity calculation can take a very long time and requires significant memory. A rough estimate of perplexity can be obtained by using only a portion (e.g., 1/4) of the full sequence.
>
> This can be controlled with --truncate_length, which reduces resource usage.
>
> When working with many sequences, limiting the number used for the calculation can further speed up processing. This can be controlled with --max_ppl_seqs.

**Outputs**
Most important files are `trainDB` and `summary.csv`. The sumary file contains a summary of the models peformance. Running several training runs appends data to the summary file. 

```
train_reference/
├── train_embeddings.npy
├── train_ids.csv
└── trainDB/               # MMseqs database
```

```
results/
├── summary.csv            # Readable summary of calculated metrics
├── <model>_identity.png.  # Identity plot of embeding predictions vs database
├── <model>_tsne.png.      # tsne of embeddings predictions vs database
├── <model>_results.tsv.   # Default output of mmseq2
```

**Metrics explanation**
| Metric                | Description                                     |
| --------------------- | ----------------------------------------------- |
| `mean_identity`       | Sequence identity vs training set (MMseqs)      |
| `mean_nn_similarity`  | Cosine similarity to nearest training embedding |
| `mean_nn_distance`    | Average distance to nearest neighbors           |
| `mean_gen_similarity` | Diversity within generated sequences            |
| `mean_perplexity`     | Sequence likelihood under ESM                   |
| `no_hit`              | Fraction with no MMseqs match                   |

Identity Buckets
 - 90–100%: Near duplicates
 - 70–90%: High similarity
 - 50–70%: Moderate similarity
 - 30–50%: Remote similarity
 - <30%: Novel sequences

Visualizations
1. Identity Distribution
    - Histogram of sequence similarity to training set
    - Highlights novelty vs memorization
2. t-SNE Embedding Space
    - Projects embeddings into 2D
    - Compares training vs generated distributions

> [!IMPORTANT]
> Important: Invalid Tokens
>
> Generated sequences may contain invalid amino acids (X, O, Z).
> These are not supported by ESM and can break evaluation steps.

### rna_validation
This script validates synthetic CRISPR RNA (crRNA) repeat sequences based on secondary structure constraints derived from base-pair probability files (e.g., ViennaRNA output).

It ensures that generated repeats follow biologically plausible structural rules such as:
 - Stem formation
 - Loop size constraints
 - Stem positioning
 - Absence of competing structures

```
basepairs files → extract stems → apply structural rules → classify VALID / INVALID
```

```
python rna_validation.py \
  --input_path PATH_TO_BASEPAIRS_FILES \
  --csv_file sequences.csv \
  --id_col id \
  --seq_col sequence \
  --output validation_results.csv
```

**Options**
| Parameter      | Type   | Default                  | Description                              |
| -------------- | ------ | ------------------------ | ---------------------------------------- |
| `--input_path` | path   | **required**             | Directory containing `*basepairs*` files |
| `--csv_file`   | path   | `None`                   | Input CSV file with sequences            |
| `--fasta_file` | path   | `None`                   | Input FASTA file with sequences          |
| `--id_col`     | string | `None`                   | Column name for sequence IDs (CSV only)  |
| `--seq_col`    | string | `None`                   | Column name for sequences (CSV only)     |
| `--output`     | path   | `validation_results.csv` | Output CSV file                          |


**Validation critera**
| Rule                 | Description                          |
| -------------------- | ------------------------------------ |
| `length_ok`          | Sequence length within allowed range |
| `stem_length_ok`     | Longest stem within valid length     |
| `loop_ok`            | Loop size between stem arms is valid |
| `stem_near_5prime`   | Stem located near 5′ end             |
| `no_competing_stems` | Limited number of alternative stems  |


**Validation Parameters**
| Parameter                 | Type  | Default | Description                                |
| ------------------------- | ----- | ------- | ------------------------------------------ |
| `--min_len`               | int   | `18`    | Minimum sequence length                    |
| `--max_len`               | int   | `40`    | Maximum sequence length                    |
| `--min_stem_length`       | int   | `3`     | Minimum stem length to be considered       |
| `--min_valid_stem_length` | int   | `5`     | Minimum valid stem length                  |
| `--max_valid_stem_length` | int   | `7`     | Maximum valid stem length                  |
| `--min_loop_length`       | int   | `4`     | Minimum loop size                          |
| `--max_loop_length`       | int   | `8`     | Maximum loop size                          |
| `--max_competing_stems`   | int   | `1`     | Maximum allowed competing stems            |
| `--near_5prime_fraction`  | float | `0.25`  | Fraction of sequence defining 5′ proximity |
| `--prob_threshold`        | float | `0.5`   | Minimum basepair probability threshold     |


**Output**
If CSV input, columns are concatenated to the original CSV, otherwise a new CSV is created.

| Column               | Description            |
| -------------------- | ---------------------- |
| `valid/invalid`      | Overall classification |
| `length_ok`          | PASS / FAIL            |
| `stem_length_ok`     | PASS / FAIL            |
| `loop_ok`            | PASS / FAIL            |
| `stem_near_5prime`   | PASS / FAIL            |
| `no_competing_stems` | PASS / FAIL            |

> [!IMPORTANT]
> Passing validation does not guarantee biological functionality.
> This script enforces structural heuristics, not full biological validation.