encdec_ckpt=$1

python latent_encoder.py \
    --ckpt ${encdec_ckpt} \
    --results_path encoded_dataset/boxing \
    --data_path datasets/boxing/ \
    --dataset boxing \
    --img_size 64x64