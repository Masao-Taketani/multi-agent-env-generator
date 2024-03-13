encdec_ckpt=$1

python latent_encoder.py \
    --ckpt ${encdec_ckpt} \
    --results_path encoded_dataset/pong_num_agts_4 \
    --data_path datasets/pong_num_agts_4/ \
    --dataset pong \
    --img_size 64x64