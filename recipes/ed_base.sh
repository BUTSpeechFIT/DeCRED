#!/usr/bin/bash
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --gpus=48
#SBATCH --cpus-per-task=128
#SBATCH --time 2-00:00:00
#SBATCH --job-name DeCRED_small
#SBATCH --partition qgpu
#SBATCH --output=outputs/decred_base.out
#SBATCH --error=outputs/decred_base.err

# cd <WORK_DIR>
source ./env.sh
RECIPE_DIR="${WORK_DIR}/recipes"
EXPERIMENT="DeCRED_base"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

export WANDB_PROJECT="DeCRED"
export WANDB_RUN_ID="${EXPERIMENT}"

args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"
  --per_device_train_batch_size="80"
  --per_device_eval_batch_size="8"
  --dataloader_num_workers="8"
  --num_train_epochs="400"
  --group_by_length="False"
  --bf16
  --do_train
  --do_evaluate
  --joint_decoding_during_training
  --load_best_model_at_end
  --metric_for_best_model="eval_wer"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1e-3"
  --warmup_steps="40000"
  --early_stopping_patience="10"
  --weight_decay="1e-6"
  --max_grad_norm="0.5"
  --lsm_factor="0.1"
  --mask_unks
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=500
  --greater_is_better="False"
  --save_total_limit="5"
  --track_ctc_loss

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="500"
  --test_splits wsj_test fisher_swbd_dev voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test ami_corpus_test gigaspeech_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="BUT-FIT/DeCRED_uni5000_normalized_tokenizer"
  --feature_extractor_name="BUT-FIT/feature_extractor_80d_16k"
  --base_encoder_model="BUT-FIT/ebranchformer_16l_512h_80x512x2d"
  --base_decoder_model="BUT-FIT/gpt2_512h_8l"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0"
)

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
PORT=13000
srun --export="MASTER=${MASTER},WORK_DIR=${WORK_DIR},PORT=${PORT}" recipes/multinode_job.sh  "${args[@]}"