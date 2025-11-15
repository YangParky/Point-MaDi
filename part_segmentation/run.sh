#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 python main.py \
#--ckpts ../experiments/pretrain/pretrain/point_madi/ckpt-epoch-300.pth \
#--root ../../../Dataset/Shapenet/ --learning_rate 0.0002 --epoch 300 --batch_size 16 \
#--log_dir ./point_madi \


CUDA_VISIBLE_DEVICES=3 python main.py --gpu 1 --ckpts ./log/part_seg/BR_DCenter_Q4_DPatch_Q2_seed31937/checkpoints/best_model.pth --test
