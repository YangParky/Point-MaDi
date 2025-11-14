import torch
import os
from tools import builder
from utils import misc, dist_utils
from utils.logger import *

import cv2
import numpy as np
from collections import defaultdict


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243", #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517", #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", # microphone
    ]

    result_save_path = os.path.join('./experiments/cdl2', "cd_results.txt")
    os.makedirs(target, exist_ok=True)
    cd_dict = defaultdict(list)
    useful_cate_names = {
        "02691156": "airplane",
        "04379243": "table",
        "03790512": "motorbike",
        "03948459": "pistol",
        "03642806": "laptop",
        "03467517": "guitar",
        "03261776": "earphone",
        "03001627": "chair",
        "02958343": "car",
        "04090263": "rifle",
        "03759954": "microphone",
    }

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            # if  taxonomy_ids[0] not in useful_cate:
            #     continue
            if taxonomy_ids[0] == "02691156":
                a, b= 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45
            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            else:
                a, b = 0, 0

            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # noise_points, dense_points, vis_points, mask_center = base_model(points, vis=True)
            # dense_points: predicted points, vis_points: ground truth points, centers: center points
            full, pos_vis_rand, pos_msk_rand, pos_full, gt_pos_vis_rand, gt_pos_msk_rand, center = base_model(points, vis=True)

            # pos_full: B x N x 3, center: B x N x 3 → 逐个样本计算 CD
            for b in range(points.size(0)):
                taxonomy_id = taxonomy_ids[b]
                if taxonomy_id in useful_cate:
                    cd_single = base_model.loss_func(pos_full[b:b+1], center[b:b+1])  # 保留 batch dim
                    cd_dict[taxonomy_id].append(cd_single.item())

            # final_image = []
            # data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            # if not os.path.exists(data_path):
            #     os.makedirs(data_path)
            #
            # points = points.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            # points = misc.get_ptcloud_img(points,a,b)
            # final_image.append(points[150:650,150:675,:])
            #
            # mask_center = mask_center.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'mask_center.txt'), mask_center, delimiter=';')
            # mask_center = misc.get_ptcloud_img(mask_center,a,b)
            # final_image.append(mask_center[150:650,150:675,:])
            
            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)

            # vis_points = vis_points.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            # vis_points = misc.get_ptcloud_img(vis_points,a,b)
            # final_image.append(vis_points[150:650,150:675,:])
            #
            # noise_points = noise_points.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'noise_points.txt'), noise_points, delimiter=';')
            # noise_points = misc.get_ptcloud_img(noise_points,a,b)
            # final_image.append(noise_points[150:650,150:675,:])
            #
            # dense_points = dense_points.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            # dense_points = misc.get_ptcloud_img(dense_points,a,b)
            # final_image.append(dense_points[150:650,150:675,:])
            #
            # img = np.concatenate(final_image, axis=1)
            # img_path = os.path.join(data_path, f'plot.jpg')
            # cv2.imwrite(img_path, img)

            # if idx > 150:
            #     break

        # 输出每个类别的平均 CD
        print("====== Per-Category Chamfer Distance (CD) ======")
        with open(result_save_path, "w") as f:
            f.write("====== Per-Category Chamfer Distance (CD) ======\n")
            for cate, cds in cd_dict.items():
                avg_cd = sum(cds) / len(cds)
                cate_name = useful_cate_names.get(cate)
                line = f"Category {cate} ({cate_name}): Avg CD = {avg_cd:.6f}, N = {len(cds)}\n"
                print(line.strip())
                f.write(line)

        return
