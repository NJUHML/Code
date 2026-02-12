import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from train_cuboid_sevir import CuboidSEVIRPLModule
import sys
sys.path.append('/root/code/CM4nowcast/models/EarthFormer_scale_run/earth-forecasting-transformer/src')
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
#from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.visualization.sevir.sevir_vis_seq import save_example_vis_results
from earthformer.metrics.sevir import SEVIRSkillScore
from earthformer.cuboid_transformer.cuboid_transformer_p import CuboidTransformerModel
from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
#from earthformer.utils.apex_ddp import ApexDDPPlugin



def form_model_normal():
    module = CuboidSEVIRPLModule(10000, "/root/code/CM4nowcast/models/EarthFormer_scale_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml", "/fs1/home/yuanhl/yangss/output/Earthformer_384")
    oc_file = "/root/code/CM4nowcast/models/EarthFormer_scale_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml"
    oc_from_file = OmegaConf.load(open(oc_file, "r"))
    oc = module.get_base_config(oc_from_file=oc_from_file)
    model_cfg = OmegaConf.to_object(oc.model)
    num_blocks = len(model_cfg["enc_depth"])
    if isinstance(model_cfg["self_pattern"], str):
        enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
    else:
        enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
    if isinstance(model_cfg["cross_self_pattern"], str):
        dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
    else:
        dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
    if isinstance(model_cfg["cross_pattern"], str):
        dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
    else:
        dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
    return CuboidTransformerModel(
            input_shape=[12,384,384,1],
            target_shape=[12,384,384,1],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

def form_model_normal_128():
    #module = CuboidSEVIRPLModule(10000, "/share/home/veiga2/yangss/pycode/nowcasting/EarthFormer_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml", "/share/home/veiga2/yangss/output/nowcasting/Earthformer_384")
    #oc_file = "/share/home/veiga2/yangss/pycode/nowcasting/EarthFormer_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml"
    module = CuboidSEVIRPLModule(10000, "/save/users/yangss/code/python_code/nowcasting/EarthFormer_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml", "/share/home/veiga2/yangss/output/nowcasting/Earthformer_384")
    oc_file = "/save/users/yangss/code/python_code/nowcasting/EarthFormer_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir/cfg_sevir.yaml"
    oc_from_file = OmegaConf.load(open(oc_file, "r"))
    oc = module.get_base_config(oc_from_file=oc_from_file)
    model_cfg = OmegaConf.to_object(oc.model)
    num_blocks = len(model_cfg["enc_depth"])
    if isinstance(model_cfg["self_pattern"], str):
        enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
    else:
        enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
    if isinstance(model_cfg["cross_self_pattern"], str):
        dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
    else:
        dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
    if isinstance(model_cfg["cross_pattern"], str):
        dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
    else:
        dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
    return CuboidTransformerModel(
            input_shape=[12, 128, 128, 1],
            target_shape=[12, 128, 128, 1],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )



def form_model_my():
    return CuboidTransformerModel(
            input_shape=[12, 384, 384, 1],
            target_shape=[12, 384, 384, 1],
            base_units=128,
            block_units=None,
            scale_alpha=1.0,
            enc_depth=[1, 1],
            dec_depth=[1, 1],
            enc_use_inter_ffn=True,
            dec_use_inter_ffn=True,
            downsample=2,
            downsample_type="patch_merge",
            #enc_attn_patterns=["axial"]*[1, 1],
            #dec_self_attn_patterns=["axial"]*[1, 1],
            #dec_cross_attn_patterns=["cross_1x1"]*[1, 1],
            enc_attn_patterns=["axial", "axial"],
            dec_self_attn_patterns=["axial", "axial"],
            dec_cross_attn_patterns=["cross_1x1", "cross_1x1"],
            dec_cross_last_n_frames=None,
            dec_use_first_self_attn=False,
            num_heads=4,
            attn_drop=0.1,
            proj_drop=0.1,
            ffn_drop=0.1,
            upsample_type="upsample",
            ffn_activation="gelu",
            gated_ffn=False,
            norm_layer="layer_norm",
            # global vectors
            num_global_vectors=8,
            use_dec_self_global=False,
            dec_self_update_global=True,
            use_dec_cross_global=False,
            use_global_vector_ffn=False,
            use_global_self_attn=True,
            separate_global_qkv=True,
            global_dim_ratio=1,
            # initial_downsample
            initial_downsample_type="stack_conv",
            initial_downsample_activation="leaky",
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=3,
            initial_downsample_stack_conv_dim_list=[32, 64, 128],
            initial_downsample_stack_conv_downscale_list=[3, 2, 2],
            initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
            # misc
            padding_type="zeros",
            z_init_method="zeros",
            checkpoint_level=0,
            pos_embed_type="t+h+w",
            use_relative_pos=True,
            self_attn_use_final_proj=True,
            # initialization
            attn_linear_init_mode="0",
            ffn_linear_init_mode="0",
            conv_init_mode="0",
            down_up_linear_init_mode="0",
            norm_init_mode="0",
        )

def form_model_my_128():
    return CuboidTransformerModel(
            input_shape=[12, 128, 128, 1],
            target_shape=[12, 128, 128, 1],
            base_units=128,
            block_units=None,
            scale_alpha=1.0,
            enc_depth=[1, 1],
            dec_depth=[1, 1],
            enc_use_inter_ffn=True,
            dec_use_inter_ffn=True,
            downsample=2,
            downsample_type="patch_merge",
            #enc_attn_patterns=["axial"]*[1, 1],
            #dec_self_attn_patterns=["axial"]*[1, 1],
            #dec_cross_attn_patterns=["cross_1x1"]*[1, 1],
            enc_attn_patterns=["axial", "axial"],
            dec_self_attn_patterns=["axial", "axial"],
            dec_cross_attn_patterns=["cross_1x1", "cross_1x1"],
            dec_cross_last_n_frames=None,
            dec_use_first_self_attn=False,
            num_heads=4,
            attn_drop=0.1,
            proj_drop=0.1,
            ffn_drop=0.1,
            upsample_type="upsample",
            ffn_activation="gelu",
            gated_ffn=False,
            norm_layer="layer_norm",
            # global vectors
            num_global_vectors=8,
            use_dec_self_global=False,
            dec_self_update_global=True,
            use_dec_cross_global=False,
            use_global_vector_ffn=False,
            use_global_self_attn=True,
            separate_global_qkv=True,
            global_dim_ratio=1,
            # initial_downsample
            initial_downsample_type="stack_conv",
            initial_downsample_activation="leaky",
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=3,
            initial_downsample_stack_conv_dim_list=[32, 64, 128],
            initial_downsample_stack_conv_downscale_list=[3, 2, 2],
            initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
            # misc
            padding_type="zeros",
            z_init_method="zeros",
            checkpoint_level=0,
            pos_embed_type="t+h+w",
            use_relative_pos=True,
            self_attn_use_final_proj=True,
            # initialization
            attn_linear_init_mode="0",
            ffn_linear_init_mode="0",
            conv_init_mode="0",
            down_up_linear_init_mode="0",
            norm_init_mode="0",
        )


def form_model_normal_pretrained(checkpoint="/save/users/yangss/output/nowcast/Earthformer_384/earthformer_sevir.pt"):
    model = form_model_normal()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    return model

def form_model_my_pretrained(checkpoint=None):
    model = form_model_my()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    return model

