#!/bin/bash
#
#SBATCH --partition=RENEWgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=48G
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

opencrispr-generate \
       	--model-path "$BASE_PATH/checkpoints_4/huggingface/ba30000" \
       	--config "$BASE_PATH/$GEN_CONFIG_NAME.yml" \
	--save-dir "$BASE_PATH/testfolder" \
	--job-idx $SLURM_JOB_ID 

#############################
#     convert to fasta      #
#############################
python "$BASE_PATH/clean_generations.py" \
	--input_path "$BASE_PATH/testfolder/generations/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}.csv" \
	--output_folder "$BASE_PATH/testfolder/cleaned_generations"

python "$BASE_PATH/csv_to_fasta.py" \
	--input_path "$BASE_PATH/testfolder/cleaned_generations/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}.csv" \
	--output_folder "$BASE_PATH/testfolder/fasta_files"

#############################
#          mmseqs2          #
#############################
conda activate "$CONDA_ENVS/mmseq"

python "$BASE_PATH/mmseq.py"  \
	--mode evaluate \
       	--reference_dir "$BASE_PATH/data/mmseq" \
       	--results_dir "$BASE_PATH/testfolder/mmseq_out" \
       	--gen_dir "$BASE_PATH/testfolder/cleaned_generations/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}.csv" \
	--tmp_dir "$BASE_PATH/testfolder/tmp" \
       	--format csv \
	--id_col context_name

#############################
#          esmfold          #
#############################
module add library/cuda/12.8/gcc.8.5.0
conda activate "$CONDA_ENVS/esmfold"

mkdir -p "$BASE_PATH/testfolder/pdbs"
mkdir -p "$BASE_PATH/testfolder/pdbs/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}"
python "$BASE_PATH/ESMFold_Snellius/esmfold.py" \
       	--fastas_folder "$BASE_PATH/testfolder/fasta_files/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}" \
      	--output_folder "$BASE_PATH/testfolder/pdbs/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}" 

#############################
#          foldseek         #
#############################
conda activate "$CONDA_ENVS/foldseek"

mkdir -p "$BASE_PATH/testfolder/tmp"
python "$BASE_PATH/foldseek.py" \
	--query_dir "$BASE_PATH/testfolder/pdbs/${GEN_CONFIG_NAME}_${SLURM_JOB_ID}" \
	--db "$BASE_PATH/foldseekDB/scope40_db" \
	--tmp_dir "$BASE_PATH/testfolder/tmp" \
	--out_dir "$BASE_PATH/testfolder/foldseek_out" \
	--job_idx $SLURM_JOB_ID
