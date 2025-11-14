#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--ckpts ../experiments/pretrain/pretrain/point_madi/ckpt-epoch-300.pth \
--root ../../../Dataset/S3DIS/s3disfull/raw/ --learning_rate 0.0002 --epoch 60 --batch_size 32 \
--log_dir ./point_madi