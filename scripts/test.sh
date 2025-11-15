#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_modelnet.yaml --test \
--exp_name point_madi --ckpts ./experiments/finetune_modelnet/finetune/point_madi/ckpt-best.pth

#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_scan_objonly.yaml --test \
#--exp_name point_madi --ckpts ./experiments/finetune_scan_objonly/finetune/point_madi/ckpt-best.pth
