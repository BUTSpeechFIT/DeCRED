#!/usr/bin/bash
#SBATCH --job-name DeCRED_init
#SBATCH --nodes=1
#SBATCH --time 1:00:00
#SBATCH --partition qcpu
#SBATCH --output=outputs/decred_init.out

source ./env.sh
cd "${WORK_DIR}" || exit

python src/initialize_base_models.py \
   --feature_extractor_name="BUT-FIT/feature_extractor_80d_16k" \
   --small_encoder_name="BUT-FIT/ebranchformer_12l_256h_80x256x2d" \
   --small_decoder_name="BUT-FIT/gpt2_256h_6l" \
   --base_encoder_name="BUT-FIT/ebranchformer_16l_512h_80x512x2d" \
   --base_decoder_name="BUT-FIT/gpt2_512h_8l"
