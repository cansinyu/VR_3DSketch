import copy
from utils.utils import set_requires_grad
from torch.utils.data import DataLoader
from network.model_utils import EMA
from network.data_loader import occupancy_field_Dataset, occupancy_field_Dataset_7cls_200num, occupancy_field_Dataset_for_prior,occupancy_field_Dataset_cl0, occupancy_field_Dataset_car, occupancy_field_Dataset_4cls_50num
from pathlib import Path
from torch.optim import AdamW,Adam
from utils.utils import update_moving_average, GLOBAL_INDEX, set_global_index
from pytorch_lightning import LightningModule
from network.model import OccupancyDiffusion
import torch.nn as nn
import os
import random



class DiffusionModel(LightningModule):
    def __init__(
        self,
        sdf_folder: str = "",
        sketch_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_sketch_condition: bool = False,
        use_text_condition: bool = True,
        noise_schedule: str = "linear",
        debug: bool = False,
        image_feature_drop_out: float = 0.1,
        view_information_ratio: float = 0.5,
        data_augmentation: bool = False,
        kernel_size: float = 2.0,
        vit_global: bool = False,
        vit_local: bool = True,
        split_dataset: bool = False,
        elevation_zero: bool = False,
        detail_view: bool = False,
        train_prior: bool = False,
        cl: bool = False,
        fix_decoder: bool = True,
        load_uni3d_prior: bool = True,
        sketch_cls: str = "chair",
        cls4_50num: bool = False,
        cls7_200num: bool = False
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = OccupancyDiffusion(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        kernel_size=kernel_size,
                                        dropout=dropout,
                                        use_sketch_condition=use_sketch_condition,
                                        use_text_condition=use_text_condition,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        vit_global=vit_global,
                                        vit_local=vit_local,
                                        verbose=verbose,
                                        train_prior=train_prior,
                                        load_uni3d_prior=load_uni3d_prior)

        self.view_information_ratio = view_information_ratio
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.sdf_folder = sdf_folder
        self.sketch_folder = sketch_folder
        self.data_class = data_class
        self.data_augmentation = data_augmentation
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.image_feature_drop_out = image_feature_drop_out
        self.train_prior = train_prior

        self.vit_global = vit_global
        self.vit_local = vit_local
        self.split_dataset = split_dataset
        self.elevation_zero = elevation_zero
        self.detail_view = detail_view
        self.optimizier = optimizier
        self.reset_parameters()
        if fix_decoder:
            print("--------fix decoder---------")
            set_requires_grad(self.model.denoise_fn, False)
            set_requires_grad(self.ema_model.denoise_fn, False)
        else:
            print("--------unfix decoder---------")
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = 4
        self.cl = cl
        self.sketch_cls = sketch_cls
        self.cls4_50num = cls4_50num
        self.cls7_200num = cls7_200num

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        if self.sketch_cls == "car" :
            _dataset = occupancy_field_Dataset_car(sdf_folder=self.sdf_folder,
                                            sketch_folder=self.sketch_folder,
                                            data_class=self.data_class,
                                            size=self.image_size,
                                            data_augmentation=self.data_augmentation,
                                            feature_drop_out=self.image_feature_drop_out,
                                            vit_global=self.vit_global,
                                            vit_local=self.vit_local,
                                            split_dataset=self.split_dataset,
                                            elevation_zero=self.elevation_zero,
                                            detail_view=self.detail_view,
                                            use_sketch_condition=self.use_sketch_condition,
                                            use_text_condition=self.use_text_condition
                                            )
            dataloader = DataLoader(_dataset,
                                    num_workers=64,
                                    batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
            
        else:
            if self.train_prior:
                _dataset = occupancy_field_Dataset_for_prior(sdf_folder=self.sdf_folder,
                                            sketch_folder=self.sketch_folder,
                                            data_class=self.data_class,
                                            size=self.image_size,
                                            data_augmentation=self.data_augmentation,
                                            feature_drop_out=self.image_feature_drop_out,
                                            vit_global=self.vit_global,
                                            vit_local=self.vit_local,
                                            split_dataset=self.split_dataset,
                                            elevation_zero=self.elevation_zero,
                                            detail_view=self.detail_view,
                                            use_sketch_condition=self.use_sketch_condition,
                                            use_text_condition=self.use_text_condition
                                            )
                dataloader = DataLoader(_dataset,
                                        num_workers=self.num_workers,
                                        batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
            elif (not self.train_prior) and self.cl:
                _dataset = occupancy_field_Dataset_cl0(sdf_folder=self.sdf_folder,
                                                sketch_folder=self.sketch_folder,
                                                data_class=self.data_class,
                                                size=self.image_size,
                                                data_augmentation=self.data_augmentation,
                                                feature_drop_out=self.image_feature_drop_out,
                                                vit_global=self.vit_global,
                                                vit_local=self.vit_local,
                                                split_dataset=self.split_dataset,
                                                elevation_zero=self.elevation_zero,
                                                detail_view=self.detail_view,
                                                use_sketch_condition=self.use_sketch_condition,
                                                use_text_condition=self.use_text_condition
                                                )
                
                dataloader = DataLoader(_dataset,
                                        num_workers=16,
                                        batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
            elif (not self.train_prior) and self.cls4_50num:
                print("----cls4_50num---")
                _dataset = occupancy_field_Dataset_4cls_50num(sdf_folder=self.sdf_folder,
                                                sketch_folder=self.sketch_folder,
                                                data_class=self.data_class,
                                                size=self.image_size,
                                                data_augmentation=self.data_augmentation,
                                                feature_drop_out=self.image_feature_drop_out,
                                                vit_global=self.vit_global,
                                                vit_local=self.vit_local,
                                                split_dataset=self.split_dataset,
                                                elevation_zero=self.elevation_zero,
                                                detail_view=self.detail_view,
                                                use_sketch_condition=self.use_sketch_condition,
                                                use_text_condition=self.use_text_condition
                                                )
                
                dataloader = DataLoader(_dataset,
                                        num_workers=16,
                                        batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
            elif (not self.train_prior) and self.cls7_200num:
                print("----cls7_200num---")
                _dataset = occupancy_field_Dataset_7cls_200num(sdf_folder=self.sdf_folder,
                                                sketch_folder=self.sketch_folder,
                                                data_class=self.data_class,
                                                size=self.image_size,
                                                data_augmentation=self.data_augmentation,
                                                feature_drop_out=self.image_feature_drop_out,
                                                vit_global=self.vit_global,
                                                vit_local=self.vit_local,
                                                split_dataset=self.split_dataset,
                                                elevation_zero=self.elevation_zero,
                                                detail_view=self.detail_view,
                                                use_sketch_condition=self.use_sketch_condition,
                                                use_text_condition=self.use_text_condition
                                                )
                
                dataloader = DataLoader(_dataset,
                                        num_workers=16,
                                        batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
                
            else:
                _dataset = occupancy_field_Dataset(sdf_folder=self.sdf_folder,
                                                sketch_folder=self.sketch_folder,
                                                data_class=self.data_class,
                                                size=self.image_size,
                                                data_augmentation=self.data_augmentation,
                                                feature_drop_out=self.image_feature_drop_out,
                                                vit_global=self.vit_global,
                                                vit_local=self.vit_local,
                                                split_dataset=self.split_dataset,
                                                elevation_zero=self.elevation_zero,
                                                detail_view=self.detail_view,
                                                use_sketch_condition=self.use_sketch_condition,
                                                use_text_condition=self.use_text_condition
                                                )
                dataloader = DataLoader(_dataset,
                                        num_workers=self.num_workers,
                                        batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        sketch_pcd = batch["sketch_pcd"]
        kernel_size = None
        loss = self.model.training_loss(
            occupancy, sketch_pcd, None, None, kernel_size=kernel_size, train_prior=self.train_prior).mean()
        self.log("loss", loss.clone().detach().item(), prog_bar=True)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()
        self.update_EMA()
        

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch)
        return super().on_train_epoch_end()
