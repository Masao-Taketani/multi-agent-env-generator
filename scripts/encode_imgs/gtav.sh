encdec_ckpt=$1

python latent_encoder.py \
    --ckpt ${encdec_ckpt} \
    --results_path encoded_dataset/gtav \
    --data_path datasets/gtav/ \
    --dataset gtav \
    --img_size 48x80