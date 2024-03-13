python enc_dec_training.py \
    --log_dir gtav_encdec \
    --use_perceptual_loss \
    --batch 64 \
    --data_path datasets/gtav/ \
    --dataset gtav \
    --img_size 48x80 \
    --nfilterDec 48