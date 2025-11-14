import os
import random
import torch
import numpy as np
import torch.nn as nn
import open3d as o3d

from pointnet2_ops import pointnet2_utils

output_dir = './vis_util/'


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center: B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)  # B G 3
        idx = knn_point(self.group_size, xyz, center)  # B G M

        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mask_Encoder(nn.Module):
    def __init__(self, num_group, group_size, mask_ratio_rand=0.6, mask_ratio_block=0.6, T=2000):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.mask_ratio_rand = mask_ratio_rand
        self.mask_ratio_block = mask_ratio_block
        self.T = T

        self.betas = linear_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio_rand == 0:
            return torch.zeros(center.shape[:2]).bool()
        num_mask = int(self.mask_ratio_rand * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([np.zeros(G - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        return torch.from_numpy(overall_mask).to(torch.bool).to(center.device)

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio_block == 0:
            return torch.zeros(center.shape[:2]).bool()
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio_block * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        return torch.stack(mask_idx).to(center.device)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)
        alphas_cumprod = self.alphas_cumprod.to(x_0.device)
        sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    def forward(self, neighborhood, center, noaug=False):
        bool_masked_pos_rand = self._mask_center_rand(center, noaug=noaug)
        bool_masked_pos_block = self._mask_center_block(center, noaug=noaug)

        B = center.shape[0]
        t = torch.randint(0, self.T, (B,), device=center.device).long()

        vis_center_rand = center[~bool_masked_pos_rand].reshape(B, -1, 3)
        msk_center_rand = center[bool_masked_pos_rand].reshape(B, -1, 3)
        vis_center_block = center[~bool_masked_pos_block].reshape(B, -1, 3)
        msk_center_block = center[bool_masked_pos_block].reshape(B, -1, 3)

        x_t_vis_rand = self.q_sample(vis_center_rand, t)
        x_t_msk_rand = self.q_sample(msk_center_rand, t)
        x_t_vis_block = self.q_sample(vis_center_block, t)
        x_t_msk_block = self.q_sample(msk_center_block, t)

        msk_neighborhood_rand = neighborhood[bool_masked_pos_rand].reshape(B, -1, self.group_size, 3)
        msk_points_rand = msk_neighborhood_rand + msk_center_rand.unsqueeze(2)
        msk_points_rand = msk_points_rand.reshape(B, -1, 3)
        x_t_msk_points_rand = self.q_sample(msk_points_rand, t)

        msk_neighborhood_block = neighborhood[bool_masked_pos_block].reshape(B, -1, self.group_size, 3)
        msk_points_block = msk_neighborhood_block + msk_center_block.unsqueeze(2)
        msk_points_block = msk_points_block.reshape(B, -1, 3)
        x_t_msk_points_block = self.q_sample(msk_points_block, t)

        return (neighborhood, center,
                bool_masked_pos_rand, bool_masked_pos_block,
                x_t_vis_rand, x_t_msk_rand,
                x_t_vis_block, x_t_msk_block,
                x_t_msk_points_rand, x_t_msk_points_block)


class PointCloudProcessor:
    def __init__(self):
        self.permutation = np.arange(100000)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def load_point_cloud(self, ply_path, num_points=2048):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype(np.float32)
        points = self.pc_norm(points)
        if points.shape[0] > num_points:
            points = self.random_sample(points, num_points)
        elif points.shape[0] < num_points:
            indices = np.random.choice(points.shape[0], num_points - points.shape[0])
            points = np.concatenate([points, points[indices]], axis=0)
        return torch.from_numpy(points).float().unsqueeze(0)  # (1, 2048, 3)


def save_point_cloud_ply(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")


def process_point_cloud(pts, num_group=64, group_size=32, mask_ratio_rand=0.6, mask_ratio_block=0.6, T=2000):
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts).float()
    if len(pts.shape) == 2:
        pts = pts.unsqueeze(0)
    pts = pts.cuda() if torch.cuda.is_available() else pts

    save_point_cloud_ply(pts[0], os.path.join(output_dir, "original_point_cloud.ply"))

    grouper = Group(num_group=num_group, group_size=group_size)
    neighborhood, center = grouper(pts)

    mask_encoder = Mask_Encoder(num_group=num_group, group_size=group_size,
                                mask_ratio_rand=mask_ratio_rand, mask_ratio_block=mask_ratio_block, T=T)
    (neighborhood, center, bool_masked_pos_rand, bool_masked_pos_block,
     x_t_vis_rand, x_t_msk_rand, x_t_vis_block, x_t_msk_block,
     x_t_msk_points_rand, x_t_msk_points_block) = mask_encoder(neighborhood, center)

    B, G, M, _ = neighborhood.shape
    batch_size = B
    vis_mask_rand = ~bool_masked_pos_rand
    msk_mask_rand = bool_masked_pos_rand
    vis_center_rand = center[vis_mask_rand].reshape(batch_size, -1, 3)
    msk_center_rand = center[msk_mask_rand].reshape(batch_size, -1, 3)
    vis_neighborhood_rand = neighborhood[vis_mask_rand].reshape(batch_size, -1, M, 3)
    msk_neighborhood_rand = neighborhood[msk_mask_rand].reshape(batch_size, -1, M, 3)
    vis_points_rand = vis_neighborhood_rand + vis_center_rand.unsqueeze(2)
    msk_points_rand = msk_neighborhood_rand + msk_center_rand.unsqueeze(2)
    vis_points_rand = vis_points_rand.reshape(batch_size, -1, 3)
    msk_points_rand = msk_points_rand.reshape(batch_size, -1, 3)

    vis_mask_block = ~bool_masked_pos_block
    msk_mask_block = bool_masked_pos_block
    vis_center_block = center[vis_mask_block].reshape(batch_size, -1, 3)
    msk_center_block = center[msk_mask_block].reshape(batch_size, -1, 3)
    vis_neighborhood_block = neighborhood[vis_mask_block].reshape(batch_size, -1, M, 3)
    msk_neighborhood_block = neighborhood[msk_mask_block].reshape(batch_size, -1, M, 3)
    vis_points_block = vis_neighborhood_block + vis_center_block.unsqueeze(2)
    msk_points_block = msk_neighborhood_block + msk_center_block.unsqueeze(2)
    vis_points_block = vis_points_block.reshape(batch_size, -1, 3)
    msk_points_block = msk_points_block.reshape(batch_size, -1, 3)

    save_point_cloud_ply(vis_points_rand[0], os.path.join(output_dir, "rand_visible_point_cloud.ply"))
    save_point_cloud_ply(msk_points_rand[0], os.path.join(output_dir, "rand_masked_point_cloud.ply"))
    save_point_cloud_ply(vis_center_rand[0], os.path.join(output_dir, "rand_visible_center.ply"))
    save_point_cloud_ply(msk_center_rand[0], os.path.join(output_dir, "rand_masked_center.ply"))
    save_point_cloud_ply(x_t_vis_rand[0], os.path.join(output_dir, "rand_visible_center_noised.ply"))
    save_point_cloud_ply(x_t_msk_rand[0], os.path.join(output_dir, "rand_masked_center_noised.ply"))
    save_point_cloud_ply(x_t_msk_points_rand[0], os.path.join(output_dir, "rand_masked_point_cloud_noised.ply"))

    save_point_cloud_ply(vis_points_block[0], os.path.join(output_dir, "block_visible_point_cloud.ply"))
    save_point_cloud_ply(msk_points_block[0], os.path.join(output_dir, "block_masked_point_cloud.ply"))
    save_point_cloud_ply(vis_center_block[0], os.path.join(output_dir, "block_visible_center.ply"))
    save_point_cloud_ply(msk_center_block[0], os.path.join(output_dir, "block_masked_center.ply"))
    save_point_cloud_ply(x_t_vis_block[0], os.path.join(output_dir, "block_visible_center_noised.ply"))
    save_point_cloud_ply(x_t_msk_block[0], os.path.join(output_dir, "block_masked_center_noised.ply"))
    save_point_cloud_ply(x_t_msk_points_block[0], os.path.join(output_dir, "block_masked_point_cloud_noised.ply"))

    timestep_ranges = [
        (0, 10), (10, 20), (20, 40), (40, 200), (200, 1000), (1000, 2000)
    ]
    for t_min, t_max in timestep_ranges:
        t = torch.randint(t_min, t_max, (batch_size,), device=pts.device).long()

        x_t_vis_rand_t = mask_encoder.q_sample(vis_center_rand, t)
        x_t_msk_rand_t = mask_encoder.q_sample(msk_center_rand, t)
        x_t_msk_points_rand_t = mask_encoder.q_sample(msk_points_rand, t)

        x_t_vis_block_t = mask_encoder.q_sample(vis_center_block, t)
        x_t_msk_block_t = mask_encoder.q_sample(msk_center_block, t)
        x_t_msk_points_block_t = mask_encoder.q_sample(msk_points_block, t)

        range_str = f"t_{t_min}_{t_max}"
        save_point_cloud_ply(x_t_vis_rand_t[0], os.path.join(output_dir, f"rand_visible_center_noised_{range_str}.ply"))
        save_point_cloud_ply(x_t_msk_rand_t[0], os.path.join(output_dir, f"rand_masked_center_noised_{range_str}.ply"))
        save_point_cloud_ply(x_t_msk_points_rand_t[0], os.path.join(output_dir, f"rand_masked_point_cloud_noised_{range_str}.ply"))
        save_point_cloud_ply(x_t_vis_block_t[0], os.path.join(output_dir, f"block_visible_center_noised_{range_str}.ply"))
        save_point_cloud_ply(x_t_msk_block_t[0], os.path.join(output_dir, f"block_masked_center_noised_{range_str}.ply"))
        save_point_cloud_ply(x_t_msk_points_block_t[0], os.path.join(output_dir, f"block_masked_point_cloud_noised_{range_str}.ply"))


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


if __name__ == "__main__":
    ply_path = "/mnt/sda/xxy/Dataset/Shapenet/ShapeNet/02691156/b3323a51c2c1af9937678474be485ca.ply"

    processor = PointCloudProcessor()
    point_cloud = processor.load_point_cloud(ply_path, num_points=2048)

    process_point_cloud(
        point_cloud,
        num_group=64,
        group_size=32,
        mask_ratio_rand=0.6,
        mask_ratio_block=0.6,
        T=2000,
    )