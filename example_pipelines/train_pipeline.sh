#!/bin/bash
#
#SBATCH --partition=RENEWgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=1-6
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

export TOKENIZERS_PARALLELISM=false

module purge
module add tools/miniconda/python3.12/24.9.2

BASE_PATH="/exports/ana-geijsenlab/sfkelder"
CONDA_ENVS="$BASE_PATH/conda_envs"
GEN_CONFIG_NAME="generate_config"
GEN_FOLDER="cas12_training"

#############################
#           setup           #
#############################
YAML_FILE="$BASE_PATH/opencrispr-repro-main/config_opencrispr.yml"

mkdir -p "$BASE_PATH/$GEN_FOLDER"
mkdir -p "$BASE_PATH/$GEN_FOLDER/configs"
mkdir -p "$BASE_PATH/$GEN_FOLDER/checkpoints"

# Range for learning rate and weight decay
MAX_VAL=0.01
MIN_VAL=0.001

# generate value at intervals to limit selection space
i=0
val=$MAX_VAL
while (( $(echo "$val >= $MIN_VAL" | bc -l) )); do
    values+=($val)

    if (( i % 2 == 0 )); then
        val=$(echo "$val / 2" | bc -l)
    else
        val=$(echo "$val / 5" | bc -l)
    fi

    ((i++))
done

random_choice() {
    local arr=("$@")
    echo "${arr[RANDOM % ${#arr[@]}]}"
}

# Pick hyperparameters for this job
lr=$(random_choice "${values[@]}")
wd=$(random_choice "${values[@]}")

# Temporary YAML for this job
TMP_YAML="$BASE_PATH/$GEN_FOLDER/configs/config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"
cp $YAML_FILE $TMP_YAML
sed -i "s/learning_rate: .*/learning_rate: $lr/" $TMP_YAML
sed -i "s/weight_decay: .*/weight_decay: $wd/" $TMP_YAML
sed -i "s|save_folder: .*|save_folder: ${BASE_PATH}/${GEN_FOLDER}/checkpoints/ckpt_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}|" "$TMP_YAML"

#############################
#     generate proteins     #
#############################
conda activate "$CONDA_ENVS/opencrispr"

torchrun \
       	--standalone \
	--nproc_per_node=1 \
	"$CONDA_ENVS/opencrispr/bin/opencrispr-train" \
       	--config "$BASE_PATH/$GEN_FOLDER/configs/config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"

timeout 3h opencrispr-generate \
       	--model-path "${BASE_PATH}/${GEN_FOLDER}/checkpoints/ckpt_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/huggingface/ba7270" \
       	--config "$BASE_PATH/$GEN_CONFIG_NAME.yml" \
	--save-dir "$BASE_PATH/$GEN_FOLDER" \
	--job-idx "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 

#############################
#     convert to fasta      #
#############################
python "$BASE_PATH/clean_generations.py" \
	--input_path "$BASE_PATH/$GEN_FOLDER/generations/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" \
	--output_folder "$BASE_PATH/$GEN_FOLDER/cleaned_generations"

python "$BASE_PATH/csv_to_fasta.py" \
	--input_path "$BASE_PATH/$GEN_FOLDER/cleaned_generations/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" \
	--output_folder "$BASE_PATH/$GEN_FOLDER/fasta_files" \
	--nrows 20

#############################
#          mmseqs2          #
#############################
conda activate "$CONDA_ENVS/mmseq"

python "$BASE_PATH/mmseq.py"  \
	--mode evaluate \
       	--reference_dir "$BASE_PATH/data/mmseq_cas12a" \
       	--results_dir "$BASE_PATH/$GEN_FOLDER/mmseq_out" \
       	--gen_dir "$BASE_PATH/$GEN_FOLDER/cleaned_generations/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" \
	--tmp_dir "$BASE_PATH/$GEN_FOLDER/tmp" \
       	--format csv \
	--id_col context_name

#############################
#          esmfold          #
#############################
module add library/cuda/12.8/gcc.8.5.0
conda activate "$CONDA_ENVS/esmfold"

mkdir -p "$BASE_PATH/$GEN_FOLDER/pdbs"
mkdir -p "$BASE_PATH/$GEN_FOLDER/pdbs/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
python "$BASE_PATH/ESMFold_Snellius/esmfold.py" \
       	--fastas_folder "$BASE_PATH/$GEN_FOLDER/fasta_files/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
      	--output_folder "$BASE_PATH/$GEN_FOLDER/pdbs/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 

#############################
#          foldseek         #
#############################
conda activate "$CONDA_ENVS/foldseek"

mkdir -p "$BASE_PATH/$GEN_FOLDER/tmp"
python "$BASE_PATH/foldseek.py" \
	--query_dir "$BASE_PATH/$GEN_FOLDER/pdbs/${GEN_CONFIG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
	--db "$BASE_PATH/foldseekDB/scope40_db" \
	--tmp_dir "$BASE_PATH/$GEN_FOLDER/tmp" \
	--out_dir "$BASE_PATH/$GEN_FOLDER/foldseek_out" \
	--job_idx "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
