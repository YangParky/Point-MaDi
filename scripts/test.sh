#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune/finetune_modelnet.yaml --exp_name test --finetune