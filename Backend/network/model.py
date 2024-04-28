import torch
import torch.nn.functional as F
from tqdm import tqdm
from network.model_utils import *
from network.unet import UNetModel
from einops import rearrange, repeat
from random import random
from functools import partial
from torch import nn
from torch.special import expm1
from network.pointnet2_cls_msg import get_model
from uni3d.models import uni3d

TRUNCATED_TIME = 0.7


class OccupancyDiffusion(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            use_sketch_condition: bool = True,
            use_text_condition: bool = True,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
            kernel_size: float = 1.0,
            vit_global: bool = False,
            vit_local: bool = True,
            train_prior: bool = True,
            load_uni3d_prior: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        self.eps = eps
        self.verbose = verbose
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.denoise_fn = UNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult, dropout=dropout,
            use_sketch_condition=use_sketch_condition,
            use_text_condition=use_text_condition,
            kernel_size=kernel_size,
            world_dims=3,
            num_heads=num_heads, vit_global=vit_global, vit_local=vit_local,
            attention_resolutions=tuple(attention_ds), with_attention=with_attention,
            verbose=verbose)
        self.vit_global = vit_global
        self.vit_local = vit_local
#        if not train_prior:
#            # self.point_encoder = get_model()
#            if load_uni3d_prior:
#                print("------load_uni3d_prior--------")
#                self.point_encoder = uni3d.create_uni3d('eva_giant_patch14_560.m30m_ft_in22k_in1k', 'pretrain_model/uni3d_pretrain.pt')
#            else:
        self.point_encoder = uni3d.create_uni3d('eva_giant_patch14_560.m30m_ft_in22k_in1k', None)
            

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def training_loss(self, img, sketch, text_feature, projection_matrix, kernel_size=None, train_prior=False, *args, **kwargs):
        batch = img.shape[0]

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        if not train_prior:
            sketch_features = self.point_encoder.encode_pc(sketch)[:, None]
            sketch_features = sketch_features / sketch_features.norm(dim=-1, keepdim=True)
        else:
            sketch_features = sketch
        
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(
                    noised_img, noise_level, sketch_features, text_feature, projection_matrix, kernel_size=kernel_size).detach_()
                
        pred = self.denoise_fn(noised_img, noise_level,
                               sketch_features, text_feature, projection_matrix, self_cond, kernel_size=kernel_size)
        loss = F.mse_loss(pred, img)
        
        # lamb=0.3
        # ds = torch.mean(1+ lamb * torch.sign(img) * torch.sign(img-pred))
        # with open('ds_train_val.txt', 'a') as file:
        #     content_to_append = f"{ds.item()}\n"
        #     file.write(content_to_append)
            
        return loss

    @torch.no_grad()
    def sample_unconditional(self, batch_size=16,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        
        from utils.condition_data import white_image_feature, an_object_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)
        white_c = torch.from_numpy(white_image_feature[1:]).to(
            device).unsqueeze(0).repeat(batch, 1, 1).to(torch.float32)
        object_c = torch.from_numpy(an_object_feature).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)

        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(
                img, noise_cond, white_c, object_c, None, x_start, kernel_size=None)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
                
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )
            img = mean + torch.sqrt(variance) * noise

        return img

    @torch.no_grad()
    def sample_with_sketch(self, sketch_c, projection_matrix=None, kernel_size=None, batch_size=16,
                           steps=50, truncated_index: float = 0.0, sketch_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        from utils.condition_data import white_image_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)
        sketch_condition = torch.from_numpy(sketch_c).to(
            device).unsqueeze(0).repeat(batch, 1, 1).to(torch.float32)
        white_c = torch.from_numpy(white_image_feature).to(
            device).unsqueeze(0).repeat(batch, 1, 1).to(torch.float32)
        _sketch_c = self.point_encoder.encode_pc(sketch_condition)[:, None]
        _sketch_c = _sketch_c / _sketch_c.norm(dim=-1, keepdim=True)
        
        # if self.vit_global and self.vit_local:
        #     _sketch_c = sketch_condition
        #     _sketch_none = white_c
        # elif self.vit_global and not self.vit_local:
        #     _sketch_c = sketch_condition[:, 0:1, :]
        #     _sketch_none = white_c[:, 0:1, :]
        # elif not self.vit_global and self.vit_local:
        #     _sketch_c = sketch_condition[:, 1:, :]
        #     _sketch_none = white_c[:, 1:, :]
        # else:
        #     _sketch_c = None
        #     _sketch_none = None
        _sketch_none = white_c[:, 0:1, :]
        pm = None
        # if projection_matrix is not None:
        #     pm = torch.from_numpy(projection_matrix).to(
        #         device).unsqueeze(0).repeat(batch, 1, 1, 1)
        # else:
        #     pm = None
        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_zero_none = self.denoise_fn(
                img, noise_cond, _sketch_none, None, pm, x_start, kernel_size=kernel_size)
            x_start = x_zero_none + sketch_w * \
                (self.denoise_fn(img, noise_cond, _sketch_c, None,
                 pm, x_start, kernel_size=kernel_size) - x_zero_none)
            
            # x_start = self.denoise_fn(img, noise_cond, _sketch_c, None, pm, x_start, kernel_size=kernel_size)
                                      
            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c

            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + torch.sqrt(variance) * noise

        return img
    
    
    @torch.no_grad()
    def sample_with_img(self, img_feat, projection_matrix=None, kernel_size=None, batch_size=16,
                           steps=50, truncated_index: float = 0.0, sketch_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        from utils.condition_data import white_image_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)
        img_condition = torch.from_numpy(img_feat).to(
            device).repeat(batch, 1, 1).to(torch.float32)
        white_c = torch.from_numpy(white_image_feature).to(
            device).unsqueeze(0).repeat(batch, 1, 1).to(torch.float32)
        _img_c = img_condition
        # if self.vit_global and self.vit_local:
        #     _sketch_c = sketch_condition
        #     _sketch_none = white_c
        # elif self.vit_global and not self.vit_local:
        #     _sketch_c = sketch_condition[:, 0:1, :]
        #     _sketch_none = white_c[:, 0:1, :]
        # elif not self.vit_global and self.vit_local:
        #     _sketch_c = sketch_condition[:, 1:, :]
        #     _sketch_none = white_c[:, 1:, :]
        # else:
        #     _sketch_c = None
        #     _sketch_none = None
        _sketch_none = white_c[:, 0:1, :]
        pm = None
        # if projection_matrix is not None:
        #     pm = torch.from_numpy(projection_matrix).to(
        #         device).unsqueeze(0).repeat(batch, 1, 1, 1)
        # else:
        #     pm = None
        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_zero_none = self.denoise_fn(
                img, noise_cond, _sketch_none, None, pm, x_start, kernel_size=kernel_size)
            x_start = x_zero_none + sketch_w * \
                (self.denoise_fn(img, noise_cond, _img_c, None,
                 pm, x_start, kernel_size=kernel_size) - x_zero_none)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c

            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + torch.sqrt(variance) * noise

        return img

    @torch.no_grad()
    def sample_with_text(self, text_c, batch_size=16,
                         steps=50, truncated_index: float = 0.0, text_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        
        from utils.condition_data import white_image_feature, an_object_feature
            
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)
        white_c = torch.from_numpy(white_image_feature[1:]).to(
            device).unsqueeze(0).repeat(batch, 1, 1).to(torch.float32)
        text_condition = torch.from_numpy(text_c).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)
        object_c = torch.from_numpy(an_object_feature).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)

        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_zero_none = self.denoise_fn(
                img, noise_cond, white_c, object_c, None, x_start, kernel_size=None)
            x_start = x_zero_none + text_w * \
                (self.denoise_fn(img, noise_cond, white_c, text_condition,
                 None, x_start, kernel_size=None) - x_zero_none)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + torch.sqrt(variance) * noise

        return img
    

    @torch.no_grad()
    def sample_with_sketch_zero_shot(self, sketch_c, projection_matrix=None, kernel_size=None, batch_size=16,
                           steps=50, truncated_index: float = 0.0, sketch_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        from utils.condition_data import white_image_feature
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)
        sketch_condition = sketch_c
        _sketch_c = sketch_condition
        white_c = torch.from_numpy(white_image_feature).to(
            device).unsqueeze(0).repeat(batch, 1, 1).to(torch.float32)
        
        # if self.vit_global and self.vit_local:
        #     _sketch_c = sketch_condition
        #     _sketch_none = white_c
        # elif self.vit_global and not self.vit_local:
        #     _sketch_c = sketch_condition[:, 0:1, :]
        #     _sketch_none = white_c[:, 0:1, :]
        # elif not self.vit_global and self.vit_local:
        #     _sketch_c = sketch_condition[:, 1:, :]
        #     _sketch_none = white_c[:, 1:, :]
        # else:
        #     _sketch_c = None
        #     _sketch_none = None
        _sketch_none = white_c[:, 0:1, :]
        pm = None
        # if projection_matrix is not None:
        #     pm = torch.from_numpy(projection_matrix).to(
        #         device).unsqueeze(0).repeat(batch, 1, 1, 1)
        # else:
        #     pm = None
        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_zero_none = self.denoise_fn(
                img, noise_cond, _sketch_none, None, pm, x_start, kernel_size=kernel_size)
            x_start = x_zero_none + sketch_w * \
                (self.denoise_fn(img, noise_cond, _sketch_c, None,
                 pm, x_start, kernel_size=kernel_size) - x_zero_none)
            
            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c

            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + torch.sqrt(variance) * noise

        return img
