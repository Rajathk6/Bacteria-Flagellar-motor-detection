"""
Standalone inference script (self-contained).

This file inlines the minimal model and utility implementations that
the original Kaggle inference script imported from the BYU-competition
`src` package so that the script can run without depending on that
package being on PYTHONPATH.

Behavior follows the working Kaggle inference script: create a simple
config, load all .pt weights from a models directory, run
sliding-window inference over .npy volumes in a data directory, and
save half-precision predictions and JSON metadata to an output folder.

Notes:
- Implementations are copied (and lightly simplified) from
  BYU-competition/src to avoid external src imports.
- This script still depends on standard packages (torch, monai, timm,
  numpy, tqdm) being installed in the environment.
"""

from types import SimpleNamespace
import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from monai.inferers import sliding_window_inference

# -----------------------------
# Small utilities copied from src
# -----------------------------

def count_parameters(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Augmentations (only definitions — used by model during init)
# -----------------------------

import random

def rotate(x, mask=None, dims=((-3, -2), (-3, -1), (-2, -1)), p=1.0):
	bs = x.shape[0]
	for d in dims:
		if random.random() < p:
			k = random.randint(0, 3)
			x = torch.rot90(x, k=k, dims=d)
			if mask is not None:
				mask = torch.rot90(mask, k=k, dims=d)

	if mask is not None:
		return x, mask
	else:
		return x


def flip_3d(x, mask=None, dims=(-3, -2, -1), p=0.5):
	axes = [i for i in dims if random.random() < p]
	if axes:
		x = torch.flip(x, dims=axes)
		if mask is not None:
			mask = torch.flip(mask, dims=axes)
	if mask is not None:
		return x, mask
	else:
		return x


def swap_dims(x, mask=None, p=0.5, dims=(-2, -1)):
	if random.random() < p:
		swap_dims = list(dims)
		random.shuffle(swap_dims)
		x = x.transpose(*swap_dims)
		if mask is not None:
			mask = mask.transpose(*swap_dims)

	if mask is not None:
		return x, mask
	else:
		return x


def coarse_dropout_3d(x, mask=None, p=0.5, fill_val=0.0, num_holes=(1, 3), hole_range=(8, 64, 64)):
	if torch.rand(1).item() < p:
		zs, ys, xs = x.shape[-3:]
		num_holes = torch.randint(low=num_holes[0], high=num_holes[1], size=(1,), device="cpu").item()
		z_start = torch.randint(low=0, high=zs - hole_range[0], size=(num_holes,), device="cpu")
		y_start = torch.randint(low=0, high=ys - hole_range[1], size=(num_holes,), device="cpu")
		x_start = torch.randint(low=0, high=xs - hole_range[2], size=(num_holes,), device="cpu")

		z_size = torch.randint(low=2, high=hole_range[0], size=(num_holes,), device="cpu")
		y_size = torch.randint(low=2, high=hole_range[1], size=(num_holes,), device="cpu")
		x_size = torch.randint(low=2, high=hole_range[2], size=(num_holes,), device="cpu")

		for i in range(num_holes):
			x[..., z_start[i]: z_start[i] + z_size[i], y_start[i]: y_start[i] + y_size[i], x_start[i]: x_start[i] + x_size[i]] = fill_val

	if mask is not None:
		return x, mask
	else:
		return x


class Mixup(nn.Module):
	def __init__(self, beta, mixadd=False):
		super().__init__()
		from torch.distributions import Beta
		self.beta_distribution = Beta(beta, beta)
		self.mixadd = mixadd

	def forward(self, X, Y, Z=None):
		b = X.shape[0]
		coeffs = self.beta_distribution.rsample(torch.Size((b,))).to(X.device)
		X_coeffs = coeffs.view((-1,) + (1,) * (X.ndim - 1))
		Y_coeffs = coeffs.view((-1,) + (1,) * (Y.ndim - 1))
		perm = torch.randperm(X.size(0))
		X_perm = X[perm]
		Y_perm = Y[perm]
		X = X_coeffs * X + (1 - X_coeffs) * X_perm
		if self.mixadd:
			Y = (Y + Y_perm).clip(0, 1)
		else:
			Y = Y_coeffs * Y + (1 - Y_coeffs) * Y_perm
		if Z is not None:
			return X, Y, Z
		return X, Y


from torch.distributions import Beta


class CutmixSimple(nn.Module):
	def __init__(self, beta=5.0, dims=(-2, -1)):
		super().__init__()
		assert all(_ < 0 for _ in dims), "dims must be negatively indexed."
		self.beta_distribution = Beta(beta, beta)
		self.dims = dims

	def forward(self, X, Y, Z=None):
		b = X.shape[0]
		cut_idx = self.beta_distribution.sample().item()
		perm = torch.randperm(X.size(0))
		X_perm = X[perm]
		Y_perm = Y[perm]
		axis = random.choice(self.dims)
		cutoff_X = int(cut_idx * X.shape[axis])
		cutoff_Y = int(cut_idx * Y.shape[axis])
		if axis == -1:
			X[..., :cutoff_X] = X_perm[..., :cutoff_X]
			Y[..., :cutoff_Y] = Y_perm[..., :cutoff_Y]
		elif axis == -2:
			X[..., :cutoff_X, :] = X_perm[..., :cutoff_X, :]
			Y[..., :cutoff_Y, :] = Y_perm[..., :cutoff_Y, :]
		else:
			raise ValueError("CutmixSimple: Axis not implemented.")
		return X, Y


# -----------------------------
# Model building blocks copied from BYU-competition/src/models/layers
# -----------------------------

from monai.networks.blocks import UpSample


class ConvBnAct3d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding: int = 0, stride: int = 1,
				 norm_layer: nn.Module = nn.BatchNorm3d, act_layer: nn.Module = nn.ReLU):
		super().__init__()
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
		self.norm = norm_layer(out_channels)
		self.act = act_layer(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		x = self.act(x)
		return x


class DecoderBlock3d(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels, norm_layer: nn.Module = nn.BatchNorm3d,
				 upsample_mode: str = "deconv", scale_factor: int = 2):
		super().__init__()
		self.upsample = UpSample(spatial_dims=3, in_channels=in_channels, out_channels=in_channels,
								 scale_factor=scale_factor, mode=upsample_mode)
		self.conv1 = ConvBnAct3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
								 norm_layer=norm_layer)
		self.conv2 = ConvBnAct3d(out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)

	def forward(self, x, skip: torch.Tensor = None):
		x = self.upsample(x)
		if skip is not None:
			x = torch.cat([x, skip], dim=1)
		x = self.conv1(x)
		x = self.conv2(x)
		return x


class UnetDecoder3d(nn.Module):
	def __init__(self, encoder_channels: tuple[int], skip_channels: tuple[int] = None,
				 decoder_channels: tuple[int] = (256,), scale_factors: tuple[int] = (2,),
				 norm_layer: nn.Module = nn.BatchNorm3d, attention_type: str = None, intermediate_conv: bool = False,
				 upsample_mode: str = "nontrainable"):
		super().__init__()
		self.decoder_channels = decoder_channels
		if skip_channels is None:
			skip_channels = list(encoder_channels[1:]) + [0]
		in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
		self.blocks = nn.ModuleList()
		for i, (ic, sc, dc, sf) in enumerate(zip(in_channels, skip_channels, decoder_channels, scale_factors)):
			self.blocks.append(DecoderBlock3d(ic, sc, dc, norm_layer=norm_layer, upsample_mode=upsample_mode,
											 scale_factor=sf))

	def forward(self, feats: list[torch.Tensor]):
		res = [feats[0]]
		feats = feats[1:]
		for i, b in enumerate(self.blocks):
			skip = feats[i] if i < len(feats) else None
			res.append(b(res[-1], skip=skip))
		return res


class SegmentationHead3d(nn.Module):
	def __init__(self, in_channels, out_channels, scale_factor: tuple[int] = (2, 2, 2)):
		super().__init__()
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
		self.upsample = UpSample(spatial_dims=3, in_channels=out_channels, out_channels=out_channels,
								 scale_factor=scale_factor, mode="nontrainable")

	def forward(self, x):
		x = self.conv(x)
		x = self.upsample(x)
		return x


# -----------------------------
# ResNet3D encoder (copied from BYU-competition/src/models/layers/resnet3d.py)
# -----------------------------

try:
	from timm.layers import DropPath
except Exception:
	class DropPath(nn.Module):
		"""Minimal DropPath fallback used only to keep API compatibility for inference.

		This simple implementation is safe for inference (it becomes identity)
		and provides a basic stochastic depth behavior if model is set to
		training mode.
		"""
		def __init__(self, drop_prob=0.0):
			super().__init__()
			self.drop_prob = float(drop_prob)

		def forward(self, x):
			if (not self.training) or self.drop_prob == 0.0:
				return x
			keep_prob = 1.0 - self.drop_prob
			if keep_prob == 1.0:
				return x
			# generate binary mask
			mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep_prob
			return x * mask / keep_prob

try:
	# prefer torch's checkpoint when available
	from torch.utils.checkpoint import checkpoint
except Exception:
	def checkpoint(module, *args, **kwargs):
		# fallback: simply run the module
		return module(*args, **kwargs)


def conv3x3x3(ic, oc, stride=1):
	return nn.Conv3d(ic, oc, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
	def __init__(self, ic, oc, stride: int = 1, downsample: bool = None, expansion_factor: int = 1,
				 drop_path_rate: float = 0.0, norm_layer: nn.Module = nn.BatchNorm3d, act_layer: nn.Module = nn.ReLU):
		super().__init__()
		self.conv1 = conv3x3x3(ic, oc, stride)
		self.bn1 = norm_layer(oc)
		self.act = act_layer(inplace=True)
		self.conv2 = conv3x3x3(oc, oc)
		self.bn2 = norm_layer(oc)
		self.drop_path = DropPath(drop_prob=drop_path_rate)
		if downsample:
			self.downsample = nn.Sequential(
				nn.Conv3d(ic * expansion_factor, oc, kernel_size=(1, 1, 1), stride=(2, 2, 2), bias=False),
				norm_layer(oc),
			)
		else:
			self.downsample = nn.Identity()

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.drop_path(x)
		residual = self.downsample(residual)
		x += residual
		x = self.act(x)
		return x


class Bottleneck(nn.Module):
	def __init__(self, ic, oc, stride: int = 1, downsample: bool = None, expansion_factor: int = 4,
				 drop_path_rate: float = 0.0, norm_layer: nn.Module = nn.BatchNorm3d, act_layer: nn.Module = nn.ReLU):
		super().__init__()
		self.conv1 = nn.Conv3d(ic * expansion_factor, oc, kernel_size=1, bias=False)
		self.bn1 = norm_layer(oc)
		self.conv2 = nn.Conv3d(oc, oc, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = norm_layer(oc)
		self.conv3 = nn.Conv3d(oc, oc * 4, kernel_size=1, bias=False)
		self.bn3 = norm_layer(oc * 4)
		self.act = act_layer(inplace=True)
		self.drop_path = DropPath(drop_prob=drop_path_rate)
		if downsample is not None:
			stride_tuple = (1, 1, 1) if expansion_factor == 1 else (2, 2, 2)
			self.downsample = nn.Sequential(
				nn.Conv3d(ic * expansion_factor, oc * 4, kernel_size=(1, 1, 1), stride=stride_tuple, bias=False),
				norm_layer(oc * 4),
			)
		else:
			self.downsample = nn.Identity()

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.act(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.drop_path(x)
		residual = self.downsample(residual)
		x += residual
		x = self.act(x)
		return x


def _make_layer(ic, oc, block, n_blocks, stride=1, downsample=False, drop_path_rates=None):
	layers = []
	if downsample:
		layers.append(block(ic=ic, oc=oc, stride=stride, downsample=downsample, drop_path_rate=drop_path_rates[0]))
	else:
		layers.append(block(ic=ic, oc=oc, stride=stride, downsample=downsample, expansion_factor=1, drop_path_rate=drop_path_rates[0]))
	for i in range(1, n_blocks):
		layers.append(block(oc, oc, drop_path_rate=drop_path_rates[i]))
	return nn.Sequential(*layers)


class ResnetEncoder3d(nn.Module):
	def __init__(self, cfg: SimpleNamespace, inference_mode: bool = False, drop_path_rate: float = 0.2,
				 in_stride: tuple[int] = (2, 2, 2), in_dilation: tuple[int] = (1, 1, 1), use_checkpoint: bool = False):
		super().__init__()
		self.cfg = cfg
		self.use_checkpoint = use_checkpoint
		bb = self.cfg.backbone
		backbone_cfg = {
			"r3d18": ([2, 2, 2, 2], BasicBlock),
			"r3d200": ([3, 24, 36, 3], Bottleneck),
		}
		if bb in backbone_cfg:
			layers, block = backbone_cfg[bb]
			wpath = "./data/model_zoo/{}_KM_200ep.pt".format(bb)
		else:
			raise ValueError(f"ResnetEncoder3d backbone: {bb} not implemented.")

		num_blocks = sum(layers)
		flat_drop_path_rates = [drop_path_rate * (i / (num_blocks - 1)) for i in range(num_blocks)]
		drop_path_rates = []
		start = 0
		for b in layers:
			end = start + b
			drop_path_rates.append(flat_drop_path_rates[start:end])
			start = end

		in_padding = tuple(_ * 3 for _ in in_dilation)
		self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(7, 7, 7), stride=in_stride,
							   dilation=in_dilation, padding=in_padding, bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

		self.layer1 = _make_layer(ic=64, oc=64, block=block, n_blocks=layers[0], stride=1, downsample=False,
								  drop_path_rates=drop_path_rates[0])
		self.layer2 = _make_layer(ic=64, oc=128, block=block, n_blocks=layers[1], stride=2, downsample=True,
								  drop_path_rates=drop_path_rates[1])
		self.layer3 = _make_layer(ic=128, oc=256, block=block, n_blocks=layers[2], stride=2, downsample=True,
								  drop_path_rates=drop_path_rates[2])
		self.layer4 = _make_layer(ic=256, oc=512, block=block, n_blocks=layers[3], stride=2, downsample=True,
								  drop_path_rates=drop_path_rates[3])

		if not inference_mode:
			try:
				state = torch.load(wpath, map_location="cpu")
			except Exception:
				state = None

		self._update_input_channels()
		with torch.no_grad():
			out = self.forward_features(torch.randn((1, self.cfg.in_chans, 96, 96, 96)))
			self.channels = [o.shape[1] for o in out]
			del out

	def _update_input_channels(self):
		b = self.conv1
		ic = self.cfg.in_chans
		w = b.weight.sum(dim=1, keepdim=True) / ic
		b.weight = nn.Parameter(w.repeat([1, ic] + [1] * (w.ndim - 2)))
		return

	def _checkpoint_if_enabled(self, module, x):
		return checkpoint(module, x) if self.use_checkpoint else module(x)

	def forward_features(self, x):
		res = []
		x = self._checkpoint_if_enabled(self.conv1, x)
		x = self.bn1(x)
		x = self.relu(x)
		res.append(x)
		x = self.maxpool(x)
		layers = [self.layer1, self.layer2, self.layer3, self.layer4]
		for layer in layers:
			x = self._checkpoint_if_enabled(layer, x)
			res.append(x)
		return res


# -----------------------------
# Small BaseModel (loss initialization is skipped in inference_mode)
# -----------------------------

class BaseModel(nn.Module):
	def __init__(self, cfg: SimpleNamespace, inference_mode: bool = False):
		super().__init__()
		self.cfg = cfg
		self.inference_mode = inference_mode
		self.loss_fn = self._init_loss_fn()

	def _init_loss_fn(self):
		if self.inference_mode:
			return None
		return None


# -----------------------------
# Full model (copied from BYU-competition/src/models/unet3d.py)
# -----------------------------

class Net(BaseModel):
	def __init__(self, cfg: SimpleNamespace, inference_mode: bool = False):
		super().__init__(cfg=cfg, inference_mode=inference_mode)
		self.cfg = cfg
		self.inference_mode = inference_mode

		if not inference_mode and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
			self.last_channels = True
		else:
			self.last_channels = False

		self.mixup = Mixup(cfg.mixup_beta)
		self.cutmix = CutmixSimple()

		self.backbone = ResnetEncoder3d(cfg=cfg, inference_mode=inference_mode, **vars(cfg.encoder_cfg))
		ecs = self.backbone.channels[::-1]

		if self.last_channels:
			self.backbone = self.backbone.to(memory_format=torch.channels_last_3d)

		self.decoder = UnetDecoder3d(encoder_channels=ecs, **vars(cfg.decoder_cfg))
		self.seg_head = SegmentationHead3d(in_channels=self.decoder.decoder_channels[-1], out_channels=cfg.seg_classes)

		if cfg.deep_supervision:
			self.aux_head = SegmentationHead3d(in_channels=ecs[0], out_channels=cfg.seg_classes)

	def proc_flip(self, x_in, dim):
		i = torch.flip(x_in, dim)
		f = self.backbone.forward_features(i)
		f = f[::-1]
		f = f[:len(self.cfg.decoder_cfg.decoder_channels) + 1]
		p = self.seg_head(self.decoder(f)[-1])
		return torch.flip(p, dim)

	def forward(self, batch):
		if self.training:
			raise RuntimeError("Net.forward: training mode not supported in standalone inference script")
		else:
			x = batch.float()
			x = x / 255.0

		if self.last_channels:
			x = x.to(memory_format=torch.channels_last_3d)

		x_in = x
		x_feats = self.backbone.forward_features(x)
		x = x_feats[::-1]
		x = x[:len(self.cfg.decoder_cfg.decoder_channels) + 1]
		x = self.decoder(x)
		x_seg = self.seg_head(x[-1])

		if self.cfg.tta:
			p1 = self.proc_flip(x_in, [2])
			p2 = self.proc_flip(x_in, [3])
			x_seg = torch.mean(torch.stack([x_seg, p1, p2]), dim=0)

		return x_seg


# -----------------------------
# Model / inference helpers
# -----------------------------

def get_roi_weight_map(shape, device, pct=0.30):
	z, h, w = shape
	z_margin = int(z * pct)
	h_margin = int(h * pct)
	w_margin = int(w * pct)
	roi_weight_map = torch.ones((z, h, w), device=device)
	if z_margin > 0:
		roi_weight_map[:z_margin] = 1e-3
		roi_weight_map[-z_margin:] = 1e-3
	if h_margin > 0:
		roi_weight_map[:, :h_margin] = 1e-3
		roi_weight_map[:, -h_margin:] = 1e-3
	if w_margin > 0:
		roi_weight_map[:, :, :w_margin] = 1e-3
		roi_weight_map[:, :, -w_margin:] = 1e-3
	return roi_weight_map


def _extract_state_dict(loaded):
	if isinstance(loaded, dict):
		for key in ("state_dict", "model_state_dict", "model"):
			if key in loaded and isinstance(loaded[key], dict):
				return loaded[key]
		return loaded
	return loaded


def load_model(fpath, device):
	"""Create Net with minimal config required for inference and load weights."""
	cfg = SimpleNamespace()
	cfg.in_chans = 1
	cfg.seg_classes = 1
	cfg.backbone = "r3d200"
	cfg.deep_supervision = True
	cfg.device = device
	cfg.roi_size = (64, 672, 672)
	cfg.kernel_size = 7
	cfg.kernel_sigma = 1.0
	cfg.kernel_type = "smooth"
	cfg.mixup_beta = 1.0
	cfg.mixup_p = 0.0
	cfg.cutmix_p = 0.0
	cfg.tta = False

	cfg.encoder_cfg = SimpleNamespace()
	cfg.encoder_cfg.use_checkpoint = False
	cfg.encoder_cfg.drop_path_rate = 0.0

	cfg.decoder_cfg = SimpleNamespace()
	cfg.decoder_cfg.decoder_channels = (256,)
	cfg.decoder_cfg.attention_type = None
	cfg.decoder_cfg.upsample_mode = "deconv"

	cfg.loss = "dice_ce"

	model = Net(cfg=cfg, inference_mode=True)

	loaded = torch.load(fpath, map_location=device)
	state_dict = _extract_state_dict(loaded)

	try:
		model.load_state_dict(state_dict)
	except Exception:
		model_dict = model.state_dict()
		filtered = {}
		for k, v in state_dict.items():
			if k in model_dict and list(model_dict[k].shape) == list(v.shape):
				filtered[k] = v
			elif k.replace("module.", "") in model_dict and list(model_dict[k].shape) == list(model_dict[k.replace("module.", "")].shape):
				filtered[k.replace("module.", "")] = v
		model_dict.update(filtered)
		model.load_state_dict(model_dict)

	model.to(device)
	model.eval()

	model_cfg = SimpleNamespace()
	model_cfg.roi_size = (128, 512, 512)
	model_cfg.infer_cfg = SimpleNamespace()
	model_cfg.infer_cfg.sw_batch_size = 1
	model_cfg.infer_cfg.overlap = (0.5, 0.5, 0.5)

	return model, model_cfg


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	working_dir = os.path.join(base_dir, "predictions")
	model_dir = os.path.join(base_dir, "BYU-competition", "byu_models")
	data_dir = os.path.join(base_dir, "DATA")
	patch_size = (128, 512, 512)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	os.makedirs(working_dir, exist_ok=True)

	models = []
	if not os.path.isdir(model_dir):
		raise FileNotFoundError(f"Model directory not found: {model_dir}")

	model_paths = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
	print(f"Found {len(model_paths)} model files in: {model_dir}")

	for mpath in tqdm(model_paths, desc="Loading models"):
		fpath = os.path.join(model_dir, mpath)
		model, model_cfg = load_model(fpath, device)
		models.append({"model": model, "cfg": model_cfg})

	if not os.path.isdir(data_dir):
		raise FileNotFoundError(f"Data directory not found: {data_dir}")

	volume_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
	print(f"Found {len(volume_files)} volumes to process in {data_dir}")

	for fname in tqdm(volume_files, desc="Processing volumes"):
		tomo_id = fname.split('.')[0]
		volume_path = os.path.join(data_dir, fname)
		print(f"\nProcessing: {tomo_id}")

		try:
			volume = np.load(volume_path)
			volume_tensor = torch.from_numpy(volume).float()
			volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
			volume_tensor = volume_tensor.to(device)

			roi_weight_map = get_roi_weight_map(patch_size, device)

			preds = None
			with torch.no_grad(), autocast(device_type='cuda', enabled=torch.cuda.is_available()):
				for idx, row in enumerate(models):
					print(f"Running model {idx+1}/{len(models)}")

					# Define predictor function that runs the model once per patch
					def _predictor(x, m=row["model"], out_size=row["cfg"].roi_size):
						out = m(x)
						if isinstance(out, dict):
							out = out.get("logits", out)
						return F.interpolate(out, size=out_size, mode='trilinear', align_corners=False)

					_preds = sliding_window_inference(
						inputs=volume_tensor,
						roi_size=row["cfg"].roi_size,
						sw_batch_size=row["cfg"].infer_cfg.sw_batch_size,
						predictor=_predictor,
						overlap=row["cfg"].infer_cfg.overlap if hasattr(row["cfg"].infer_cfg, 'overlap') else 0.5,
						mode='gaussian'
					)
 
					_preds = _preds[0, 0, ...]
					_preds = torch.sigmoid(_preds)

					if preds is None:
						preds = _preds
					else:
						preds += _preds

					if torch.cuda.is_available():
						torch.cuda.empty_cache()

			preds = preds / len(models)
			preds = preds.half().cpu()
			pred_path = os.path.join(working_dir, f"{tomo_id}_pred.pt")
			torch.save(preds, pred_path)

			meta_path = pred_path.replace('.pt', '.json')
			metadata = {"tomo_id": tomo_id, "z_shape": int(volume.shape[0]), "y_shape": int(volume.shape[1]), "x_shape": int(volume.shape[2])}
			with open(meta_path, 'w') as f:
				json.dump(metadata, f, indent=2)

			print(f"Saved prediction to: {pred_path}")
			print(f"Saved metadata to: {meta_path}")

		except Exception as e:
			print(f"Error processing {tomo_id}: {str(e)}")
			continue


if __name__ == "__main__":
	main()

