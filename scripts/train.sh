#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain/pretrain.yaml --exp_name point_madi
#
#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_scan_objbg.yaml \
#--finetune_model --exp_name point_madi --ckpts ./experiments/pretrain/pretrain/point_madi/ckpt-last.pth
#
#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_scan_objonly.yaml \
#--finetune_model --exp_name point_madi --ckpts ./experiments/pretrain/pretrain/point_madi/ckpt-last.pth
#
#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_scan_hardest.yaml \
#--finetune_model --exp_name point_madi --ckpts ./experiments/pretrain/pretrain/point_madi/ckpt-last.pth
#
#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_modelnet.yaml \
#--finetune_model --exp_name point_madi --ckpts ./experiments/pretrain/pretrain/point_madi/ckpt-last.pth


for i in {0..9}
do
    echo "5 way 10 shot, fold: $i"
    CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/finetune/fewshot.yaml \
    --finetune_model --exp_name point_madi_5way10shot \
    --ckpts ./experiments/pretrain/pretrain/point_madi/ckpt-last.pth --seed $RANDOM \
    --way 5 --shot 10 --fold $i
done