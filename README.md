## Setup OpenCrispr
To get started with the program, follow the steps below:

### 1. Create a Conda environment
First, create a new environment with Python 3.12:

```
conda create -n opencrispr-env python=3.12 -y
conda activate opencrispr-env
```

### 2. Clone the repository
Download the project from GitHub:

```
git clone https://github.com/sfkelder/opencrispr-repro-main.git
```

### 3. Install dependencies
Navigate into the project folder and install the required packages:

```
cd opencrispr-repro-main
pip install .
```

> [!NOTE]
> If the installation fails, you can use the included requirements.txt file, which 
> contains the exact dependency versions used for this project.

## Setup Pipeline
This repository includes Python files for further data processing, located in the [`scripts`](./scripts) folder.  

Running these files requires some additional setup. It is recommended to use a separate environment to avoid conflicts with OpenCrispr and other projects.

> [!IMPORTANT]
> Example pipeline files are located in the [`example_pipelines`](./example_pipelines) 
> folder. Make sure to change the paths inside the files to your local corresponding 
> file paths.

### 1. Create a Conda environment
First, create a new environment with Python 3.12:

```
conda create -n pipeline-env python=3.12 -y
conda activate pipeline-env
```

### 2. Install dependencies

Conda packages:
```
conda install -c conda-forge -c bioconda foldseek mmseqs2 viennarna -y
```

Correct torch and cuda verions:
```
python -m pip install torch==2.10.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

Pip dependencies:
```
python -m pip install pandas tqdm biopython numpy scikit-learn matplotlib seaborn filelock fair-esm accelerate
```

### 3. Download ESMFold
The pipeline requires ESMFold as a dependency. 

```
git clone https://github.com/sara-nl/ESMFold_Snellius
```

> [!WARNING]
> In the current setup, ESMFold may require a different CUDA version than the one 
> previously installed.
>
> On an HPC system, this can be resolved by loading the appropriate module, as 
> illustrated in [`train_pipeline.sh`](./example_pipelines/train_pipeline.sh). 
> Alternatively, a separate conda environment can be created specifically for ESMFold 
> to isolate its dependencies from other software.
>
> This approach ensures that version conflicts are avoided and that both the pipeline 
> and ESMFold can run reliably.


## Configuration files
The Protein- and crRNA generation models require configuration files to run. Aditionaly protein prediction requires another seperate configuration file.

### Protein model configuration
The protein model needs configuration for training and generation. This configuration is done using a `.yml` style setup. Below is an example configuration.

```
# Folder where checkpoints and model outputs will be saved
save_folder: ./checkpoints

data:
  # Column containing protein sequences
  sequence_col: protein
  # Paths to training and validation CSV files
  train_data_path: "./opencrispr-repro-main/data/crispr_train_test.csv"
  val_data_path: "./opencrispr-repro-main/data/crispr_val_test.csv"
  # Label column (if applicable, otherwise null)
  label_col: null
  # Batch_size specifies how many samples are fed through the model per training step.
  # If set to high the model could run out of memmory.
  batch_size: 4
  num_workers: 2

model:
  # Always keep name as progen2
  name: progen2
  # This determines the size of the model that is used, see Progen2 git for more 
  # information
  path: base

training:
  # Training parameters
  learning_rate: 0.0001
  weight_decay: 0.001
  warmup_steps: 1000
  train_steps: 10000
  total_lr_decay_factor: 0.2
  gradient_clipping_threshold: 1.0

# Interval (in steps) at which to save checkpoints
save_interval_steps: 1000
```
> [!CAUTION]
>  **Caution: Invalid Amino Acid Tokens (X, O, Z)**
>
> The original ProGen2 model may generate non-standard amino acid tokens such as `X`, `O`, or `Z`.  
> These tokens are **not supported by ESM-based models** and may cause failures in downstream preprocessing or embedding steps.
>
> This issue can occur if the model is not properly trained or fine-tuned (e.g., due to an excessively low learning rate or insufficient convergence).


### Protein generation configuration
Protein generation requires a configuration file in `.yml` format. Below is an example of a minimal setup:

```
# Minimal setup
context:
  # Name of this generation run
  name: some_name
  # Protein sequence to start generation from. 
  # Use "1" for unconditional generation
  seq: "1"
```

```
# Optional extensions
num_samples: 1000      # Number of protein sequences to generate
batch_size: 10         # Number of sequences per batch
temperature: 1         # Sampling temperature for diversity
top_p: 1               # Nucleus sampling probability
top_k: 0               # Top-K sampling cutoff
```

> [!Note]
> 1. The `seq` value must always be enclosed in double quotes `" "`.
> 2. The `seq` value can be any length but must contain only valid protein residues.
> 3. The first character must be either `"1"` or `"2"`:
>    - `"1"` → start generation from the N-terminus
>    - `"2"` → start generation from the C-terminus
> 4. For unconditional generation, use `"1"`.

### crRNA model configuration
The crRNA model requires a configuration file in `.yml` format. Below is an example file.

```
# -------------------------
# Model configuration
# -------------------------

# Always keep these settings inline with the esm_model. 
esm_model: "facebook/esm2_t6_8M_UR50D" 
d_s: 128
d_s_protein: 320
n_enc_layers: 1
n_dec_layers: 3
enc_self_attn:
  n_heads: 8
dec_self_attn:
  n_heads: 8
dec_cross_attn:
  n_heads: 8

# -------------------------
# Dataset configuration
# -------------------------
dataset:
  # Paths to training and validation CSV files
  train_csv: "./opencrispr-repro-main/data/crispr_train_test.csv"
  val_csv: "./opencrispr-repro-main/data/crispr_val_test.csv"
  columns:
    # Colums used for training
    crispr: "crispr_repeat"
    protein: "protein"
    tracr: "tracr"

# -------------------------
# Training configuration
# -------------------------
batch_size: 16
epochs: 12

optimizer:
  initial_lr: 0.0002
  warmup: 800
  weight_decay: 0.001
  acc_batches: 2   # gradient accumulation

# -------------------------
# Optional: device
# -------------------------
device: "cuda"
```

> [!WARNING]
> The columns crispr, protein and tracr must always be provided. If the CRISPR-Cas type
> does not include a tracr please provide a column name with empty values.

## Usage

### Protein model training
```
opencrispr-train --config path/to/config.yml [--eval-only]
```

| Argument      | Type   | Default | Description                                                                                 |
| ------------- | ------ | ------- | ------------------------------------------------------------------------------------------- |
| `--config`    | string | —       | **Required.** Path to a JSON or YAML configuration file following the `FinetuneAPI` schema. |
| `--eval-only` | flag   | `False` | Optional. Run evaluation only without training.                                             |

> [!NOTE]
> The training script supports Distributed Data Processing (ddp). This can be achieved 
> by using `torchrun` with flags `--standalone` and `--nproc_per_node=1`. Make sure 
> `--nproc_per_node=1` always equals the amount of gpus and processes available. In 
> Batch scripsts `nproc_per_node` = `gres=gpu` = `ntasks`

### Protein model generation
```
opencrispr-generate --model-path PATH_TO_MODEL \
                            --config PATH_TO_CONFIG \
                            [--save-dir PATH_TO_OUTPUT] \
                            [--job-idx JOB_INDEX]
```

| Argument       | Type   | Default | Description                                                                                                      |
| -------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `--model-path` | string | —       | **Required.** Path to the trained model directory.                                                               |
| `--config`     | string | —       | **Required.** Path to a JSON or YAML file containing generation hyperparameters. Must include a `context` field. |
| `--save-dir`   | string | `'./'`  | Optional. Directory to save generated protein sequences (default: current folder).                               |
| `--job-idx`    | string | `None`  | Optional. Index for parallel jobs; appends `_JOBIDX` to output files to avoid overwriting.                       |


### crRNA model training
```
python train_grna.py --config PATH_TO_CONFIG [--save-embeddings] [--embeddings-dir PATH]
```

| Argument            | Type           | Default        | Description                                                                                                          |
| ------------------- | -------------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `--config`          | file path      | —              | **Required.** Path to a YAML configuration file defining dataset paths, model hyperparameters, and training options. |
| `--save-embeddings` | flag           | `False`        | Optional. If set, protein embeddings will be cached to disk for faster subsequent runs.                              |
| `--embeddings-dir`  | directory path | `./embeddings` | Optional. Directory where protein embeddings are stored when `--save-embeddings` is used.                            |


### crRNA model sample
```
grna-modeling-train --ckpt-path PATH_TO_CHECKPOINT [INPUT_OPTIONS] [SAMPLING_OPTIONS] [OUTPUT_OPTIONS]
```


**INPUT_OPTIONS**
| Argument         | Type      | Default                | Description                                                    |
| ---------------- | --------- | ---------------------- | -------------------------------------------------------------- |
| `--sequence`     | string    | —                      | Single protein sequence string. Provide only one input source. |
| `--sequence-id`  | string    | `"generated_sequence"` | Name/ID for the protein sequence (used in output).             |
| `--csv`          | file path | —                      | CSV file containing protein sequences.                         |
| `--sequence-col` | string    | `"sequence"`           | Column in CSV with protein sequences.                          |
| `--id-col`       | string    | `None`                 | Column in CSV with protein IDs/names.                          |
| `--fasta`        | file path | —                      | FASTA file containing protein sequences.                       |

> [!WARNING]
> **Exactly one** of `--sequence`, `--csv`, or `--fasta` must be provided.



**SAMPLING_OPTIONS**
| Argument        | Type  | Default | Description                                            |
| --------------- | ----- | ------- | ------------------------------------------------------ |
| `--num-samples` | int   | `5`     | Number of gRNAs to sample per protein.                 |
| `--temperature` | float | `1.0`   | Sampling temperature controlling randomness/diversity. |
| `--batch-size`  | int   | `1`     | Batch size for gRNA sampling.                          |
| `--max-len`     | int   | `300`   | Maximum length of generated gRNA sequences.            |
| `--silent`      | flag  | `False` | Disable progress output.                               |



**OUTPUT_OPTIONS**
| Argument      | Type      | Default | Description                                       |
| ------------- | --------- | ------- | ------------------------------------------------- |
| `--ckpt-path` | file path | —       | **Required.** Path to trained model checkpoint.   |
| `--output`    | directory | `"./"`  | Directory where generated gRNA CSV will be saved. |

### crRNA model score
```
grna-modeling-score --input PATH_TO_INPUT_CSV \
                      --ckpt-path PATH_TO_CHECKPOINT \
                      [--protein-col COL] \
                      [--tracr-col COL] \
                      [--crispr-col COL] \
                      [--rna-batch-size N] \
                      [--output PATH]
```

| Argument           | Type      | Default      | Description                                                   |
| ------------------ | --------- | ------------ | ------------------------------------------------------------- |
| `--input`          | file path | —            | **Required.** Input CSV containing protein and RNA sequences. |
| `--ckpt-path`      | file path | —            | **Required.** Path to trained model checkpoint.               |
| `--protein-col`    | string    | `"protein"`  | Column containing protein sequences.                          |
| `--tracr-col`      | string    | `"tracrRNA"` | Column containing tracrRNA sequences.                         |
| `--crispr-col`     | string    | `"crRNA"`    | Column containing crRNA sequences.                            |
| `--rna-batch-size` | int       | `16`         | Number of RNA pairs to score per batch (per protein).         |
| `--output`         | directory | `"./"`       | Directory where the scored CSV will be saved.                 |

> [!WARNING]
> The columns crispr, protein and tracr must always be provided. If the CRISPR-Cas type
> does not include a tracr please provide a column name with empty values.

## Citations

**Nijkamp, E., Ruffolo, J., Weinstein, E.N., Naik, N., & Madani, A.** ProGen2: Exploring the Boundaries of Protein Language Models. arXiv preprint arXiv:2206.13517 (2022). https://arxiv.org/abs/2206.13517

**Ruffolo, J.A., Nayfach, S., Gallagher, J. et al.** Design of highly functional genome editors by modelling CRISPR–Cas sequences. Nature 645, 518–525 (2025). https://doi.org/10.1038/s41586-025-09298-z


