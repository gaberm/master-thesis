#!/bin/bash

# exp_lst=(paws_x_mbert_fft_oob_none paws_x_mbert_fft_ls_none paws_x_mbert_fft_ts_none paws_x_xlmr_fft_oob_none paws_x_xlmr_fft_ls_none paws_x_xlmr_fft_ts_none)
exp_lst=(paws_x_mbert_fft paws_x_mbert_fft_ls paws_x_xlmr_fft paws_x_xlmr_fft_ls)


for exp in "${exp_lst[@]}"; do
    echo "Training $exp"
    python train.py +exp=train/$exp
done