#!/usr/bin/bash
#SBATCH --job-name DeCRED_prep
#SBATCH --nodes=1
#SBATCH --time 2-00:00:00
#SBATCH --partition qcpu
#SBATCH --output=outputs/data_prep.out

source ./env.sh
cd "${WORK_DIR}" || exit
EXPERIMENT="preprocess_english"
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
  --test_splits wsj_test fisher_swbd_dev voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"
)

python huggingface_asr/src/trainers/train_enc_dec_asr.py "${args[@]}" --preprocess_dataset_only
