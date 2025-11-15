import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torchvision import transforms

from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import print_log
from utils import knn
from utils import misc
from datasets import data_transforms

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .build import MODELS


train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudScaleAndTranslate(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
    ]
)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.emb_dim = dim
        self.max_period = max_period

        self.freq = nn.Parameter(torch.linspace(0, math.log(max_period), dim // 2))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, ts):
        freq = torch.exp(self.freq).to(ts.device)
        emb = ts[:, None] * freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.mlp(emb)
        return emb


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    # @profile
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group) # B G 3
        idx = knn.knn_point(self.group_size, xyz, center)  # B G M

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        context_dim = context_dim if context_dim is not None else dim

        # Linear layers for q, k, v
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias) # Query from x
        self.k_linear = nn.Linear(context_dim, dim, bias=qkv_bias)  # Key from cond
        self.v_linear = nn.Linear(context_dim, dim, bias=qkv_bias)  # Value from cond

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c=None):
        cond = x if c is None else c  # If cond is None, use x as context (self-attention fallback)

        B, N, C = x.shape
        _, V, _ = cond.shape

        q = self.q_linear(x)  # B, N, C
        k = self.k_linear(cond)  # B, V, C
        v = self.v_linear(cond)  # B, M, C

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        k = k.reshape(B, V, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, V, head_dim
        v = v.reshape(B, V, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, V, head_dim

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, M
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to v
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B, N, C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y=None):
        if y is None:
            # Self attention
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Self attention + Cross attention
        B, N, C = x.shape
        L = y.shape[1]
        x = torch.cat([x, y], dim=1)
        qkv = self.qkv(x).reshape(B, N + L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cross attention
        attn = (q[:, :, N:] @ k[:, :, :].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v[:, :, :]).transpose(1, 2).reshape(B, L, C)
        y = self.proj(y)
        y = self.proj_drop(y)

        # Self attention
        attn = (q[:, :, :N] @ k[:, :, :N].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v[:, :, :N]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, y


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, y=None):  # y is q

        # Can operate as either self-attention or cross-attention.
        if y is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        new_x = self.norm1(x)
        new_y = self.norm1(y)

        new_x, new_y = self.attn(new_x, new_y)
        new_x = x + self.drop_path(new_x)
        new_y = y + self.drop_path(new_y)

        new_x = new_x + self.drop_path(self.mlp(self.norm2(new_x)))
        new_y = new_y + self.drop_path(self.mlp(self.norm2(new_y)))
        return new_x, new_y


class Decoder_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = CrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, cond=None):

        if cond is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(cond)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos, x_mask=None, pos_mask=None):
        if x_mask is None:
            for _, block in enumerate(self.blocks):
                x = block(x + pos)
            return x
        else:
            for _, block in enumerate(self.blocks):
                x, x_mask = block(x + pos, x_mask + pos_mask)
            return x, x_mask


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=256, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.blocks = nn.ModuleList([
            Decoder_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, t_emb=None, return_token_num=None, cond=None, cond_pos=None):

        x = x + t_emb if t_emb is not None else x
        for i, block in enumerate(self.blocks):
            if cond is not None:
                x = block(x + pos, cond=cond + cond_pos)
            else:
                x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel

        return x


class Mask_Encoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio_rand = config.encoder_config.mask_ratio_rand
        self.mask_ratio_block = config.encoder_config.mask_ratio_block
        self.trans_dim = config.encoder_config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_config.encoder_dims
        self.depth = config.encoder_config.depth
        self.drop_path_rate = config.encoder_config.drop_path_rate
        self.num_heads = config.encoder_config.num_heads
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # self.T = config.diffusion_config.num_steps
        self.T2 = config.diffusion_config.num_steps2
        self.betas = torch.linspace(config.diffusion_config.beta_start, config.diffusion_config.beta_end, self.T2)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)
        alphas_cumprod = self.alphas_cumprod.to(x_0.device)
        sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    # @profile
    def forward(self, neighborhood, center, noaug=False):
        bool_masked_pos_rand = self._mask_center_rand(center, noaug=noaug)
        bool_masked_pos_block = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C
        B, G, C = group_input_tokens.size()

        x_vis_rand = group_input_tokens[~bool_masked_pos_rand].reshape(B, -1, C)
        x_mask_rand = group_input_tokens[bool_masked_pos_rand].reshape(B, -1, C)
        vis_center_rand = center[~bool_masked_pos_rand].reshape(B, -1, 3)
        mask_center_rand = center[bool_masked_pos_rand].reshape(B, -1, 3)

        x_vis_block = group_input_tokens[~bool_masked_pos_block].reshape(B, -1, C)
        x_mask_block = group_input_tokens[bool_masked_pos_block].reshape(B, -1, C)
        vis_center_block = center[~bool_masked_pos_block].reshape(B, -1, 3)
        mask_center_block = center[bool_masked_pos_block].reshape(B, -1, 3)

        # Partial positional diffusion (e.g., only on visible or masked tokens) is also optional;
        # it simplifies the task and often performs as well as or even better than full diffusion.
        t = torch.randint(0, self.T2, (B,), device=neighborhood.device).long()
        vis_center_rand = self.q_sample(vis_center_rand, t)
        mask_center_rand = self.q_sample(mask_center_rand, t)
        vis_center_block = self.q_sample(vis_center_block, t)
        mask_center_block = self.q_sample(mask_center_block, t)

        pos_vis_rand = self.pos_embed(vis_center_rand)
        pos_mask_rand = self.pos_embed(mask_center_rand)
        pos_vis_block = self.pos_embed(vis_center_block)
        pos_mask_block = self.pos_embed(mask_center_block)

        x_vis_rand, x_mask_rand = self.blocks(x_vis_rand, pos_vis_rand, x_mask_rand, pos_mask_rand)
        x_vis_block, x_mask_block = self.blocks(x_vis_block, pos_vis_block, x_mask_block, pos_mask_block)

        x_vis_rand = self.norm(x_vis_rand)
        x_mask_rand = self.norm(x_mask_rand)
        x_vis_block = self.norm(x_vis_block)
        x_mask_block = self.norm(x_mask_block)

        return (x_vis_rand, x_mask_rand, bool_masked_pos_rand), (x_vis_block, x_mask_block, bool_masked_pos_block)


@MODELS.register_module()
class Point_MaDi(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MaDi]', logger='Point_MaDi')
        self.config = config
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.gamma = config.gamma
        self.mask_ratio_rand = config.encoder_config.mask_ratio_rand
        self.mask_ratio_block = config.encoder_config.mask_ratio_block
        self.trans_dim = config.encoder_config.trans_dim
        self.encoder_dims = config.encoder_config.encoder_dims
        self.depth = config.encoder_config.depth
        self.num_heads = config.encoder_config.num_heads
        self.drop_path_rate = config.encoder_config.drop_path_rate

        self.T = config.diffusion_config.num_steps
        self.use_time = config.diffusion_config.use_time
        self.betas = torch.linspace(config.diffusion_config.beta_start, config.diffusion_config.beta_end, self.T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.decoder_depth = config.decoder_config.depth
        self.decoder_drop_path_rate = config.decoder_config.drop_path_rate
        self.decoder_num_heads = config.decoder_config.num_heads

        self.MAE_encoder = Mask_Encoder(config)
        self.mask_token = nn.Conv1d(3 * self.group_size, self.trans_dim, 1)
        dpr = [x.item() for x in torch.linspace(0, self.decoder_drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        self.time_emb = TimeEmbedding(self.trans_dim) if self.use_time else None
        self.pred_pos_vis = nn.Sequential(
            nn.Linear(self.trans_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        self.pred_pos_msk = nn.Sequential(
            nn.Linear(self.trans_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == 'cdl1':
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'emd':
            from emd import earth_mover_distance
            self.loss_func = earth_mover_distance().cuda()
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)
        alphas_cumprod = self.alphas_cumprod.to(x_0.device)
        sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    def p_sample_center(self, noisy_center, t, mask, x_vis, x_mask):
        """
        Reverse sampling for center points at timestep t.

        :param noisy_center: Noisy center points at timestep t [B, G, 3]
        :param t: Timestep [1]
        :param mask: Mask indicator [B, G]
        :param x_vis: Latent of visible patches [B, V, C]
        :param pos_vis: Positional embedding of visible centers [B, V, C]
        :param pos_mask: Positional embedding of masked centers [B, M, C]
        :return: Denoised center points at timestep t-1 [B, G, 3]
        """
        B, G, _ = noisy_center.shape
        device = noisy_center.device

        # Get diffusion parameters
        betas_t = self.betas[t].to(device)
        alpha_bar_t = self.alphas_cumprod[t].to(device)
        alpha_bar_t_minus_one = self.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).to(device)
        sqrt_alphas_bar_t_minus_one = torch.sqrt(alpha_bar_t_minus_one).to(device)

        # Prepare inputs
        x_vis = x_vis.to(device)
        x_mask = x_mask.to(device)

        # Predict denoised centers using MAE_encoder outputs
        with torch.no_grad():
            # Use provided x_vis (visible patch latents) and compute masked latents
            x_vis_out = self.pred_pos_vis(x_vis)  # [B, V, 3]
            x_mask_out = self.pred_pos_msk(x_mask)  # [B, M, 3]

            # Reconstruct full center tensor
            center_pred = torch.zeros(B, G, 3, device=device)
            center_pred[~mask] = x_vis_out.view(-1, 3)
            center_pred[mask] = x_mask_out.view(-1, 3)

        # Compute model mean for reverse diffusion
        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)) * noisy_center + \
                     (sqrt_alphas_bar_t_minus_one * betas_t / (1 - alpha_bar_t)) * center_pred

        # Compute noise variance
        sigma_t = torch.sqrt(betas_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)).to(device)

        # Return denoised centers
        if t == 0:
            return model_mean
        else:
            return model_mean + sigma_t * torch.randn_like(noisy_center)

    def p_sample_patch(self, noisy_patch, t, mask, center, x_vis):
        """
        Reverse sampling for masked patches at timestep t.

        :param noisy_patch: Noisy masked patches at timestep t [B, M*group_size, 3]
        :param t: Timestep [1]
        :param mask: Mask indicator [B, G]
        :param center: Center points (denoised or noisy) [B, G, 3]
        :param x_vis: Latent of visible patches [B, V, C]
        :return: Denoised patches at timestep t-1 [B, M*group_size, 3]
        """
        B, _, _ = noisy_patch.shape
        M = mask.sum(dim=1)[0].item()  # Number of masked groups
        device = noisy_patch.device

        # Get diffusion parameters
        betas_t = self.betas[t].to(device)
        alpha_bar_t = self.alphas_cumprod[t].to(device)
        alpha_bar_t_minus_one = self.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).to(device)
        sqrt_alphas_bar_t_minus_one = torch.sqrt(alpha_bar_t_minus_one).to(device)

        # Prepare inputs
        pos_emd_vis = self.MAE_encoder.pos_embed(center[~mask].reshape(B, -1, 3)).to(device)
        pos_emd_msk = self.MAE_encoder.pos_embed(center[mask].reshape(B, -1, 3)).to(device)
        pos_full = torch.cat([pos_emd_vis, pos_emd_msk], dim=1)
        t_emb = self.time_emb(t.to(device))[:, None, :].expand(-1, pos_full.shape[1], -1)

        # Convert noisy patches to mask tokens
        mask_token = self.mask_token(noisy_patch.reshape(B, M, -1).transpose(1, 2)).transpose(1, 2)
        x_full = torch.cat([x_vis.to(device), mask_token], dim=1)

        # Run decoder
        with torch.no_grad():
            x_rec = self.MAE_decoder(x_full, pos_full, t_emb, M)
            x_rec = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)

        # Compute model mean
        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)) * noisy_patch + \
                     (sqrt_alphas_bar_t_minus_one * betas_t / (1 - alpha_bar_t)) * x_rec

        # Compute noise variance
        sigma_t = torch.sqrt(betas_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)).to(device)

        # Return denoised patches
        if t == 0:
            return model_mean
        else:
            return model_mean + sigma_t * torch.randn_like(noisy_patch)

    def sample_centers(self, center_shape, mask, x_vis, x_mask, trace=False):
        """
        Sample center points from Gaussian noise.

        :param center_shape: Shape of center points [B, G, 3]
        :param mask: Mask indicator [B, G]
        :param x_vis: Latent of visible patches [B, V, C]
        :param pos_vis: Positional embedding of visible centers [B, V, C]
        :param pos_mask: Positional embedding of masked centers [B, M, C]
        :param trace: If True, return all denoising steps
        :return: Denoised centers [B, G, 3] or list of steps
        """
        B, G, _ = center_shape
        device = x_vis.device
        noisy_center = torch.randn(center_shape).to(device)
        diffusion_sequence = [noisy_center] if trace else None

        for i in range(self.T - 1, -1, -1):
            t = torch.full((1,), i, device=device)
            noisy_center = self.p_sample_center(noisy_center, t, mask, x_vis, x_mask)
            if trace:
                diffusion_sequence.append(noisy_center)

        if trace:
            return diffusion_sequence
        return noisy_center

    def sample_patches(self, x_vis, mask, center, trace=False, noise_patch=None):
        """
        Sample masked patches from Gaussian noise.

        :param x_vis: Latent of visible patches [B, V, C]
        :param mask: Mask indicator [B, G]
        :param center: Center points [B, G, 3]
        :param trace: If True, return all denoising steps
        :param noise_patch: Pre-defined noise [B, M*group_size, 3]
        :return: Denoised patches [B, M*group_size, 3] or list of steps
        """
        B, V, C = x_vis.shape
        M = mask.sum(dim=1)[0].item()
        device = x_vis.device
        if noise_patch is None:
            noise_patch = torch.randn((B, M * self.group_size, 3)).to(device)

        diffusion_sequence = [noise_patch] if trace else None

        for i in range(self.T - 1, -1, -1):
            t = torch.full((1,), i, device=device)
            noise_patch = self.p_sample_patch(noisy_patch=noise_patch, t=t, mask=mask, center=center, x_vis=x_vis)
            if trace:
                diffusion_sequence.append(noise_patch)

        if trace:
            return diffusion_sequence
        return noise_patch

    def forward(self, pts, noaug=False, vis=False):
        B, N, _ = pts.shape
        G = self.num_group
        t = torch.randint(0, self.T, (B,), device=pts.device).long()

        neighborhood, center = self.group_divider(pts)
        (x_vis_rand, x_mask_rand, mask_rand), (x_vis_block, x_mask_block, mask_block) = self.MAE_encoder(neighborhood, center, noaug)

        pos_vis_rand = self.pred_pos_vis(x_vis_rand)
        pos_msk_rand = self.pred_pos_msk(x_mask_rand)
        pos_vis_block = self.pred_pos_vis(x_vis_block)
        pos_msk_block = self.pred_pos_msk(x_mask_block)

        gt_pos_vis_rand = center[~mask_rand].reshape(B, -1, 3)
        gt_pos_msk_rand = center[mask_rand].reshape(B, -1, 3)
        gt_pos_vis_block = center[~mask_block].reshape(B, -1, 3)
        gt_pos_msk_block = center[mask_block].reshape(B, -1, 3)

        center_loss_rand_vis = F.mse_loss(pos_vis_rand, gt_pos_vis_rand)
        center_loss_rand_msk = F.mse_loss(pos_msk_rand, gt_pos_msk_rand)
        center_loss_block_vis = F.mse_loss(pos_vis_block, gt_pos_vis_block)
        center_loss_block_msk = F.mse_loss(pos_msk_block, gt_pos_msk_block)

        pos_emd_vis_rand = self.MAE_encoder.pos_embed(pos_vis_rand.detach())
        pos_emd_msk_rand = self.MAE_encoder.pos_embed(pos_msk_rand.detach())
        pos_full_rand = torch.cat([pos_emd_vis_rand, pos_emd_msk_rand], dim=1)
        M_rand = pos_emd_msk_rand.shape[1]

        t_emb = self.time_emb(t)[:, None, :].expand(-1, pos_full_rand.shape[1], -1) if self.use_time == 'True' else None
        x_0_rand = neighborhood[mask_rand].reshape(B, M_rand, -1)
        x_t_rand = self.q_sample(x_0_rand, t)
        mask_token_rand = self.mask_token(x_t_rand.transpose(1, 2)).transpose(1, 2)
        x_full_rand = torch.cat([x_vis_rand, mask_token_rand], dim=1)
        x_rec_rand = self.MAE_decoder(x_full_rand, pos_full_rand, t_emb, M_rand)
        x_rec_rand = self.increase_dim(x_rec_rand.transpose(1, 2)).transpose(1, 2).reshape(B * M_rand, -1, 3)
        x_gt_rand = neighborhood[mask_rand].reshape(B * M_rand, -1, 3)
        patch_loss_rand = self.loss_func(x_rec_rand, x_gt_rand)

        pos_emd_vis_block = self.MAE_encoder.pos_embed(pos_vis_block.detach())
        pos_emd_msk_block = self.MAE_encoder.pos_embed(pos_msk_block.detach())
        pos_full_block = torch.cat([pos_emd_vis_block, pos_emd_msk_block], dim=1)
        M_block = pos_emd_msk_block.shape[1]

        x_0_block = neighborhood[mask_block].reshape(B, M_block, -1)
        x_t_block = self.q_sample(x_0_block, t)
        mask_token_block = self.mask_token(x_t_block.transpose(1, 2)).transpose(1, 2)
        x_full_block = torch.cat([x_vis_block, mask_token_block], dim=1)
        x_rec_block = self.MAE_decoder(x_full_block, pos_full_block, t_emb, M_block)
        x_rec_block = self.increase_dim(x_rec_block.transpose(1, 2)).transpose(1, 2).reshape(B * M_block, -1, 3)
        x_gt_block = neighborhood[mask_block].reshape(B* M_block, -1, 3)
        patch_loss_block = self.loss_func(x_rec_block, x_gt_block)

        center_loss = (center_loss_rand_vis + center_loss_rand_msk + center_loss_block_vis + center_loss_block_msk) * 0.25
        patch_loss = (patch_loss_rand + patch_loss_block) * 0.5

        if vis:
            V = G - M_rand
            vis_points = neighborhood[~mask_rand].reshape(B * V, self.group_size, 3)
            full_vis = vis_points + center[~mask_rand].reshape(B * V, 1, 3)
            full_rebuild = x_rec_rand + center[mask_rand].reshape(B * M_rand, 1, 3)

            full = torch.cat([full_vis, full_rebuild], dim=0)
            full = full.reshape(B, self.num_group * self.group_size, 3)
            pos_full = torch.cat([pos_vis_rand, pos_msk_rand], dim=1)
            return full, pos_vis_rand, pos_msk_rand, pos_full, gt_pos_vis_rand, gt_pos_msk_rand, center.reshape(B, -1, 3)
        else:
            return center_loss * self.gamma, patch_loss


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        concat_f = torch.cat([x.mean(1), x.max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)

        return ret