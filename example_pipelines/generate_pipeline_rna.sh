#!/bin/bash
#
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

export TOKENIZERS_PARALLELISM=false

module purge
module add tools/miniconda/python3.12/24.9.2

BASE_PATH="/exports/ana-geijsenlab/sfkelder"
CONDA_ENVS="$BASE_PATH/conda_envs"
GEN_CONFIG_NAME="generate_config"

#############################
#     generate proteins     #
#############################
conda activate "$CONDA_ENVS/opencrispr"

timeout 10m opencrispr-generate \
       	--model-path "$BASE_PATH/cas12_training/checkpoints/ckpt_24142097_23/huggingface/ba7270" \
       	--config "$BASE_PATH/$GEN_CONFIG_NAME.yml" \
	--save-dir "$BASE_PATH/test_rna_fold" \
	--job-idx $SLURM_JOB_ID

#############################
#       generate grna       #
#############################

python "$BASE_PATH/clean_generations.py" \
        --input_path "$BASE_PATH/test_rna_fold/generations/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}.csv" \
        --output_folder "$BASE_PATH/test_rna_fold/cleaned_generations"

grna-modeling-sample \
        --ckpt-path  "$BASE_PATH/checkpoints/grna_model-epoch=11-step=11208-val_loss=0.2718.ckpt"  \
        --csv "$BASE_PATH/test_rna_fold/cleaned_generations/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}.csv" \
        --output "$BASE_PATH/test_rna_fold" \
        --id-col context_name

#############################
#     convert to fasta      #
#############################

python "$BASE_PATH/csv_to_fasta.py" \
        --input_path "$BASE_PATH/test_rna_fold/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}_grna_predictions.csv" \
        --output_folder "$BASE_PATH/test_rna_fold/" \
        --seq_col crRNA \
        --single_file \
        --id_col grna_id \
        --trim_left 15

#############################
#     rna validity check    #
#############################
conda activate "$CONDA_ENVS/rnafold"

mkdir -p  "$BASE_PATH/test_rna_fold/probabilities"
cd "$BASE_PATH/test_rna_fold/probabilities"
RNAplfold -o <  "$BASE_PATH/test_rna_fold/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}_grna_predictions.fasta"

python  "$BASE_PATH/rna_validation.py" \
        -o "$BASE_PATH/test_rna_fold/out.csv" \
        --input_path  "$BASE_PATH/test_rna_fold/probabilities" \
        --csv_file "$BASE_PATH/test_rna_fold/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}_grna_predictions.csv" \
        --id_col grna_id \
        --seq_col crRNA

