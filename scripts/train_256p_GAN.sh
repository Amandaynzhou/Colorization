### Using labels only, use only abnormal pic
python train.py --name color_GAN --resize_or_crop 'scale_width_and_crop' --loadSize 256 --fineSize 224 --lr 0.0001 --niter 100 --tf_log --niter_decay 100 --gpu_ids 4 --batchSize 10 --serial_batches --no_instance