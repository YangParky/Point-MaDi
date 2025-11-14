import os
import time
import argparse

import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from torchvision import transforms

from easydict import EasyDict
from tools import builder
from utils.logger import get_logger
from collections import defaultdict


class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        device = pc.device
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2./3., scale_high=3./2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        device = pc.device
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(device)) + \
                            torch.from_numpy(xyz2).float().to(device)
        return pc


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Point-MaDi denoising process")
    parser.add_argument("--config", type=str, default='/mnt/sda/xxy/Project/Point-MaDi/cfgs/pretrain/pretrain.yaml')
    parser.add_argument("--checkpoint", type=str,
                        default='/mnt/sda/xxy/Project/Point-MaDi/experiments/pretrain/pretrain/Point_MaDi_BR_DCenter_DPatch/ckpt-last.pth')
    parser.add_argument("--output_path", type=str, default="/mnt/sda/xxy/Project/Point-MaDi/vis_util/denoising/")
    parser.add_argument("--samples_per_category", type=int, default=10, help="Target number of samples per category")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--categories", type=str, nargs="+",
                        default=["02691156",  # plane
                                 "04379243",  # table
                                 "03790512",  # motorbike
                                 "03948459",  # pistol
                                 "03642806",  # laptop
                                 "03467517",  # guitar
                                 "03001627",  # chair
                                 "02958343",  # car
                                 "04090263",  # rifle
                                 "03759954",  # microphone
                                 ],
                        help="ShapeNet category IDs (e.g., Airplane, Table, Chair, Laptop)")
    return parser.parse_args()


def get_config():
    config = EasyDict()
    config.dataset = EasyDict()
    config.dataset.val = EasyDict()
    config.dataset.val._base_ = EasyDict()
    config.dataset.val._base_.NAME = "ShapeNet"
    config.dataset.val._base_.DATA_PATH = "/mnt/sda/xxy/Dataset/Shapenet/ShapeNet55-34/ShapeNet-55"
    config.dataset.val._base_.PC_PATH = "/mnt/sda/xxy/Dataset/Shapenet/ShapeNet55-34/shapenet_pc"
    config.dataset.val._base_.N_POINTS = 8192
    config.dataset.val._base_.ratio = 1.0
    config.dataset.val.others = EasyDict()
    config.dataset.val.others.subset = "test"
    config.dataset.val.others.npoints = 1024
    config.dataset.val.others.bs = 2

    config.model = EasyDict()
    config.model.NAME = "Point-MaDi"
    config.model.group_size = 32
    config.model.num_group = 64
    config.model.loss = "cdl2"
    config.model.gamma = 0.025
    config.model.encoder_config = EasyDict({
        "mask_ratio_rand": 0.6,
        "mask_ratio_block": 0.6,
        "trans_dim": 384,
        "encoder_dims": 384,
        "depth": 12,
        "drop_path_rate": 0.1,
        "num_heads": 6
    })
    config.model.diffusion_config = EasyDict({
        "num_steps": 2000,
        "beta_start": 0.0001,
        "beta_end": 0.02
    })
    config.model.decoder_config = EasyDict({
        "depth": 4,
        "drop_path_rate": 0.1,
        "num_heads": 6
    })

    config.npoints = 1024
    config.num_workers = 2
    return config


def get_shapenet_val_loader(args, config, device, batch_size=1, num_points=2048):
    logger = get_logger("visualization")
    config.dataset.val._base_.NAME = "ShapeNet"
    config.dataset.val.others.npoints = num_points
    try:
        _, val_dataloader = builder.dataset_builder(args, config.dataset.val)
    except Exception as e:
        logger.error(f"Failed to build dataset: {str(e)}")
        raise

    val_transforms = transforms.Compose([
        # PointcloudScaleAndTranslate(),
        # PointcloudRotate(),
    ])

    class TransformedDataLoader:
        def __init__(self, dataloader, transforms, categories, device):
            self.dataloader = dataloader
            self.transforms = transforms
            self.categories = categories
            self.device = device
            self._iterator = None

        def __iter__(self):
            self._iterator = iter(self.dataloader)
            return self

        def __next__(self):
            if self._iterator is None:
                raise StopIteration("Iterator not initialized. Call __iter__ first.")
            while True:
                try:
                    taxonomy_ids, model_ids, data = next(self._iterator)
                    if taxonomy_ids[0] in self.categories:
                        points = data.to(self.device)
                        # points = self.transforms(points)
                        return {"points": points, "taxonomy_id": taxonomy_ids[0], "model_id": model_ids[0]}
                except StopIteration:
                    self._iterator = None
                    raise
                except Exception as e:
                    logger.warning(f"Error loading batch: {str(e)}. Skipping...")
                    continue

    return TransformedDataLoader(val_dataloader, val_transforms, args.categories, device)


def collect_samples(val_loader, categories, samples_per_category, logger):
    samples_by_category = defaultdict(list)
    seen_model_ids = defaultdict(set)
    total_samples_needed = len(categories) * samples_per_category
    max_iterations = total_samples_needed * 10
    iteration = 0

    for batch in val_loader:
        try:
            taxonomy_id = batch["taxonomy_id"]
            model_id = batch["model_id"]
            if taxonomy_id in categories and model_id not in seen_model_ids[taxonomy_id]:
                samples_by_category[taxonomy_id].append(batch)
                seen_model_ids[taxonomy_id].add(model_id)
                logger.info(f"Collected sample for category {taxonomy_id}, model {model_id}. "
                            f"Progress: {len(samples_by_category[taxonomy_id])}/{samples_per_category}")
                if all(len(samples_by_category[cat]) >= samples_per_category for cat in categories):
                    break
            iteration += 1
            if iteration >= max_iterations:
                logger.warning("Max iterations reached. Stopping sample collection.")
                break
        except Exception as e:
            logger.warning(f"Error processing batch: {str(e)}. Continuing...")
            iteration += 1

    for cat in categories:
        count = len(samples_by_category[cat])
        if count < samples_per_category:
            logger.warning(f"Category {cat}: Collected {count} samples, needed {samples_per_category}")
        else:
            logger.info(f"Category {cat}: Successfully collected {count} samples")

    return samples_by_category


def visualize_denoising(model, val_loader, output_path, samples_per_category, categories, logger):
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(output_path, exist_ok=True)

    if not categories:
        logger.error("No categories specified. Exiting.")
        return

    logger.info(f"Collecting up to {samples_per_category} samples for categories: {categories}")
    samples_by_category = collect_samples(val_loader, categories, samples_per_category, logger)

    for taxonomy_id in samples_by_category:
        samples = samples_by_category[taxonomy_id]
        logger.info(f"Processing {len(samples)} samples for category {taxonomy_id}")
        for sample_idx, batch in enumerate(samples[:samples_per_category]):
            pts = batch["points"][:1].to(device)
            model_id = batch["model_id"]
            timestamp = int(time.time() * 1000)
            data_path = os.path.join(output_path, f"{taxonomy_id}_{model_id}_{timestamp}")
            os.makedirs(data_path, exist_ok=True)
            logger.info(f"Saving visualizations to {data_path}")

            try:
                outputs = model(pts, noaug=False, vis=True)
                logger.info(f"Model output: {len(outputs)} items")
                for i, out in enumerate(outputs):
                    logger.info(f"Output {i}: type={type(out)}, shape={out.shape if hasattr(out, 'shape') else 'N/A'}")
                if len(outputs) != 7:
                    logger.error(f"Expected 7 outputs, got {len(outputs)}. Skipping sample.")
                    continue
                recon_points, recon_pos_vis, recon_pos_msk, recon_pos_full, gt_pos_vis, gt_pos_msk, gt_center = outputs
            except Exception as e:
                logger.error(f"Error processing sample {taxonomy_id}_{sample_idx} (model {model_id}): {str(e)}")
                continue

            try:
                recon_points = recon_points[0].detach().cpu().numpy()
                recon_pos_vis = recon_pos_vis[0].detach().cpu().numpy()
                recon_pos_msk = recon_pos_msk[0].detach().cpu().numpy()
                recon_pos_full = recon_pos_full[0].detach().cpu().numpy()
                points = pts[0].detach().cpu().numpy()
                gt_pos_vis = gt_pos_vis[0].detach().cpu().numpy()
                gt_pos_msk = gt_pos_msk[0].detach().cpu().numpy()
                gt_center = gt_center[0].detach().cpu().numpy()
                logger.info(f"Shapes: recon_points={recon_points.shape}, recon_pos_vis={recon_pos_vis.shape}, "
                           f"recon_pos_msk={recon_pos_msk.shape}, recon_pos_full={recon_pos_full.shape}, "
                           f"points={points.shape}, gt_pos_vis={gt_pos_vis.shape}, gt_pos_msk={gt_pos_msk.shape}, "
                           f"gt_center={gt_center.shape}")
            except Exception as e:
                logger.error(f"Error converting tensors for sample {taxonomy_id}_{sample_idx}: {str(e)}")
                continue

            visualize_with_open3d(recon_points, recon_pos_vis, recon_pos_msk, recon_pos_full, points, gt_pos_vis, gt_pos_msk, gt_center, data_path, taxonomy_id, sample_idx)
            visualize_with_matplotlib(recon_points, recon_pos_vis, recon_pos_msk, recon_pos_full, points, gt_pos_vis, gt_pos_msk, gt_center, data_path, taxonomy_id, sample_idx)


def visualize_with_open3d(recon_points, recon_pos_vis, recon_pos_msk, recon_pos_full, points, gt_pos_vis, gt_pos_msk, gt_center, data_path, taxonomy_id, sample_idx):
    pcd_recon_points = o3d.geometry.PointCloud()
    pcd_recon_points.points = o3d.utility.Vector3dVector(recon_points)
    pcd_recon_points.paint_uniform_color([1, 0, 0])  # Red

    pcd_recon_pos_vis = o3d.geometry.PointCloud()
    pcd_recon_pos_vis.points = o3d.utility.Vector3dVector(recon_pos_vis)
    pcd_recon_pos_vis.paint_uniform_color([0, 1, 0])  # Green

    pcd_recon_pos_msk = o3d.geometry.PointCloud()
    pcd_recon_pos_msk.points = o3d.utility.Vector3dVector(recon_pos_msk)
    pcd_recon_pos_msk.paint_uniform_color([0, 0.5, 0])  # Dark green

    pcd_recon_pos_full = o3d.geometry.PointCloud()
    pcd_recon_pos_full.points = o3d.utility.Vector3dVector(recon_pos_full)
    pcd_recon_pos_full.paint_uniform_color([1, 0, 1])  # Magenta

    pcd_gt_points = o3d.geometry.PointCloud()
    pcd_gt_points.points = o3d.utility.Vector3dVector(points)
    pcd_gt_points.paint_uniform_color([1, 1, 0])  # Yellow

    pcd_gt_pos_vis = o3d.geometry.PointCloud()
    pcd_gt_pos_vis.points = o3d.utility.Vector3dVector(gt_pos_vis)
    pcd_gt_pos_vis.paint_uniform_color([0, 0, 1])  # Blue

    pcd_gt_pos_msk = o3d.geometry.PointCloud()
    pcd_gt_pos_msk.points = o3d.utility.Vector3dVector(gt_pos_msk)
    pcd_gt_pos_msk.paint_uniform_color([0, 0, 0.5])  # Dark blue

    pcd_gt_center = o3d.geometry.PointCloud()
    pcd_gt_center.points = o3d.utility.Vector3dVector(gt_center)
    pcd_gt_center.paint_uniform_color([0.5, 0.5, 0.5])  # Gray

    o3d.io.write_point_cloud(os.path.join(data_path, f"recon_points_{taxonomy_id}_{sample_idx}.ply"), pcd_recon_points)
    o3d.io.write_point_cloud(os.path.join(data_path, f"recon_pos_vis_{taxonomy_id}_{sample_idx}.ply"), pcd_recon_pos_vis)
    o3d.io.write_point_cloud(os.path.join(data_path, f"recon_pos_msk_{taxonomy_id}_{sample_idx}.ply"), pcd_recon_pos_msk)
    o3d.io.write_point_cloud(os.path.join(data_path, f"recon_pos_full_{taxonomy_id}_{sample_idx}.ply"), pcd_recon_pos_full)
    o3d.io.write_point_cloud(os.path.join(data_path, f"gt_points_{taxonomy_id}_{sample_idx}.ply"), pcd_gt_points)
    o3d.io.write_point_cloud(os.path.join(data_path, f"gt_pos_vis_{taxonomy_id}_{sample_idx}.ply"), pcd_gt_pos_vis)
    o3d.io.write_point_cloud(os.path.join(data_path, f"gt_pos_msk_{taxonomy_id}_{sample_idx}.ply"), pcd_gt_pos_msk)
    o3d.io.write_point_cloud(os.path.join(data_path, f"gt_center_{taxonomy_id}_{sample_idx}.ply"), pcd_gt_center)


def visualize_with_matplotlib(recon_points, recon_pos_vis, recon_pos_msk, recon_pos_full, points, gt_pos_vis, gt_pos_msk, gt_center, data_path, taxonomy_id, sample_idx):
    fig = plt.figure(figsize=(24, 6))

    ax1 = fig.add_subplot(241, projection='3d')
    ax1.scatter(recon_points[:, 0], recon_points[:, 1], recon_points[:, 2], c='r', s=1)
    ax1.set_title("Recon Points")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(242, projection='3d')
    ax2.scatter(recon_pos_vis[:, 0], recon_pos_vis[:, 1], recon_pos_vis[:, 2], c='g', s=10)
    ax2.set_title("Recon Pos Vis")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    ax3 = fig.add_subplot(243, projection='3d')
    ax3.scatter(recon_pos_msk[:, 0], recon_pos_msk[:, 1], recon_pos_msk[:, 2], c='darkgreen', s=10)
    ax3.set_title("Recon Pos Msk")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    ax4 = fig.add_subplot(244, projection='3d')
    ax4.scatter(recon_pos_full[:, 0], recon_pos_full[:, 1], recon_pos_full[:, 2], c='m', s=10)
    ax4.set_title("Recon Pos Full")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")

    ax5 = fig.add_subplot(245, projection='3d')
    ax5.scatter(points[:, 0], points[:, 1], points[:, 2], c='y', s=1)
    ax5.set_title("GT Points")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")

    ax6 = fig.add_subplot(246, projection='3d')
    ax6.scatter(gt_pos_vis[:, 0], gt_pos_vis[:, 1], gt_pos_vis[:, 2], c='b', s=10)
    ax6.set_title("GT Pos Vis")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_zlabel("Z")

    ax7 = fig.add_subplot(247, projection='3d')
    ax7.scatter(gt_pos_msk[:, 0], gt_pos_msk[:, 1], gt_pos_msk[:, 2], c='darkblue', s=10)
    ax7.set_title("GT Pos Msk")
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax7.set_zlabel("Z")

    ax8 = fig.add_subplot(248, projection='3d')
    ax8.scatter(gt_center[:, 0], gt_center[:, 1], gt_center[:, 2], c='gray', s=10)
    ax8.set_title("GT Center")
    ax8.set_xlabel("X")
    ax8.set_ylabel("Y")
    ax8.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"point_cloud_{taxonomy_id}_{sample_idx}.png"))
    plt.close()


def main():
    args = parse_args()
    logger = get_logger("visualization")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config = get_config()
    try:
        model = builder.model_builder(config.model)
        builder.load_model(model, args.checkpoint, logger=logger)
        model.to(device)
        logger.info(f"Model loaded and moved to {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    val_loader = get_shapenet_val_loader(args, config, device, batch_size=1, num_points=2048)
    visualize_denoising(model, val_loader, args.output_path, args.samples_per_category, args.categories, logger)


if __name__ == "__main__":
    main()