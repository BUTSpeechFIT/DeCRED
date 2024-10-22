#!/usr/bin/bash
#SBATCH --job-name DeCRED_tokenizer
#SBATCH --nodes=1
#SBATCH --time 2:00:00
#SBATCH --partition qcpu
#SBATCH --output=outputs/decred_base_tokenizer.out

source ./env.sh
cd "${WORK_DIR}" || exit
EXPERIMENT="train_tokenizer"
RECIPE_DIR="${WORK_DIR}/recipes"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"


export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"


args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="500"

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Tokenizer related arguments
  --tokenizer_name="BUT-FIT/DeCRED_uni5000_normalized_tokenizer"
  --vocab_size=5000
  --tokenizer_type="unigram"
  --text_column_name="text"
  --train_split="train"
  --pad_token="([pad])"
  --unk_token="([unk])"
  --bos_token="([bos])"
  --eos_token="([eos])"
  --mask_token="([mask])"
)

python huggingface_asr/src/trainers/train_tokenizer.py "${args[@]}"
