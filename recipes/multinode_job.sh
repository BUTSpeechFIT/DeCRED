#! /bin/bash
cd "${WORK_DIR}" || exit
source ./env.sh
args=("${@:1}")
torchrun \
  --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
  --nnodes="${SLURM_JOB_NUM_NODES}" \
  --node_rank="${SLURM_PROCID}" \
  --master_addr="${MASTER}" \
   --master_port="${PORT}" \
   huggingface_asr/src/trainers/train_enc_dec_asr.py "${args[@]}"