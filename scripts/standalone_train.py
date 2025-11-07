import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from types import SimpleNamespace
from monai.networks.blocks import UpSample

try:
    import wandb
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Model Components
# -----------------------------

try:
    from timm.layers import DropPath
except Exception:
    class DropPath(nn.Module):
        """Minimal DropPath implementation for stochastic depth."""
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if (not self.training) or self.drop_prob == 0.0:
                return x
            keep_prob = 1.0 - self.drop_prob
            if keep_prob == 1.0:
                return x
            mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep_prob
            return x * mask / keep_prob

try:
    from torch.utils.checkpoint import checkpoint
except Exception:
    def checkpoint(module, *args, **kwargs):
        return module(*args, **kwargs)

# Augmentation functions
def rotate(x, mask=None, dims=((-3, -2), (-3, -1), (-2, -1)), p=1.0):
    bs = x.shape[0]
    for d in dims:
        if random.random() < p:
            k = random.randint(0, 3)
            x = torch.rot90(x, k=k, dims=d)
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=d)
    return (x, mask) if mask is not None else x

def flip_3d(x, mask=None, dims=(-3, -2, -1), p=0.5):
    axes = [i for i in dims if random.random() < p]
    if axes:
        x = torch.flip(x, dims=axes)
        if mask is not None:
            mask = torch.flip(mask, dims=axes)
    return (x, mask) if mask is not None else x

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
        return (X, Y, Z) if Z is not None else (X, Y)

class ConvBnAct3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU):
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
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=nn.BatchNorm3d,
                 upsample_mode="deconv", scale_factor=2):
        super().__init__()
        self.upsample = UpSample(spatial_dims=3, in_channels=in_channels, out_channels=in_channels,
                                scale_factor=scale_factor, mode=upsample_mode)
        self.conv1 = ConvBnAct3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                               norm_layer=norm_layer)
        self.conv2 = ConvBnAct3d(out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetDecoder3d(nn.Module):
    def __init__(self, encoder_channels, skip_channels=None, decoder_channels=(256,),
                 scale_factors=(2,), norm_layer=nn.BatchNorm3d, upsample_mode="deconv"):
        super().__init__()
        self.decoder_channels = decoder_channels
        if skip_channels is None:
            skip_channels = list(encoder_channels[1:]) + [0]
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()
        for ic, sc, dc, sf in zip(in_channels, skip_channels, decoder_channels, scale_factors):
            self.blocks.append(
                DecoderBlock3d(ic, sc, dc, norm_layer=norm_layer,
                             upsample_mode=upsample_mode, scale_factor=sf)
            )

    def forward(self, feats):
        res = [feats[0]]
        feats = feats[1:]
        for i, block in enumerate(self.blocks):
            skip = feats[i] if i < len(feats) else None
            res.append(block(res[-1], skip=skip))
        return res

class SegmentationHead3d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = UpSample(spatial_dims=3, in_channels=out_channels,
                               out_channels=out_channels, scale_factor=scale_factor,
                               mode="nontrainable")

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

# ResNet3D encoder components
def conv3x3x3(ic, oc, stride=1):
    return nn.Conv3d(ic, oc, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, downsample=None, expansion_factor=1,
                 drop_path_rate=0.0, norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU):
        super().__init__()
        self.conv1 = conv3x3x3(ic, oc, stride)
        self.bn1 = norm_layer(oc)
        self.act = act_layer(inplace=True)
        self.conv2 = conv3x3x3(oc, oc)
        self.bn2 = norm_layer(oc)
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(ic * expansion_factor, oc, kernel_size=1, stride=2, bias=False),
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
    def __init__(self, ic, oc, stride=1, downsample=None, expansion_factor=4,
                 drop_path_rate=0.0, norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU):
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
                nn.Conv3d(ic * expansion_factor, oc * 4, kernel_size=1, stride=stride_tuple, bias=False),
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
        layers.append(block(ic=ic, oc=oc, stride=stride, downsample=downsample,
                          drop_path_rate=drop_path_rates[0]))
    else:
        layers.append(block(ic=ic, oc=oc, stride=stride, downsample=downsample,
                          expansion_factor=1, drop_path_rate=drop_path_rates[0]))
    for i in range(1, n_blocks):
        layers.append(block(oc, oc, drop_path_rate=drop_path_rates[i]))
    return nn.Sequential(*layers)

class ResnetEncoder3d(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.2, in_stride=(2, 2, 2),
                 in_dilation=(1, 1, 1), use_checkpoint=False):
        super().__init__()
        self.cfg = cfg
        self.use_checkpoint = use_checkpoint
        backbone_cfg = {
            "r3d200": ([3, 24, 36, 3], Bottleneck),
        }
        layers, block = backbone_cfg["r3d200"]

        num_blocks = sum(layers)
        flat_drop_path_rates = [drop_path_rate * (i / (num_blocks - 1)) for i in range(num_blocks)]
        drop_path_rates = []
        start = 0
        for b in layers:
            end = start + b
            drop_path_rates.append(flat_drop_path_rates[start:end])
            start = end

        in_padding = tuple(_ * 3 for _ in in_dilation)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=7,
                             stride=in_stride, dilation=in_dilation,
                             padding=in_padding, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(ic=64, oc=64, block=block, n_blocks=layers[0],
                                stride=1, downsample=False, drop_path_rates=drop_path_rates[0])
        self.layer2 = _make_layer(ic=64, oc=128, block=block, n_blocks=layers[1],
                                stride=2, downsample=True, drop_path_rates=drop_path_rates[1])
        self.layer3 = _make_layer(ic=128, oc=256, block=block, n_blocks=layers[2],
                                stride=2, downsample=True, drop_path_rates=drop_path_rates[2])
        self.layer4 = _make_layer(ic=256, oc=512, block=block, n_blocks=layers[3],
                                stride=2, downsample=True, drop_path_rates=drop_path_rates[3])

        with torch.no_grad():
            out = self.forward_features(torch.randn((1, cfg.in_chans, 96, 96, 96)))
            self.channels = [o.shape[1] for o in out]
            del out

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

# Full UNet model with ResNet-200 backbone
class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mixup = Mixup(cfg.mixup_beta)

        self.backbone = ResnetEncoder3d(cfg=cfg, **vars(cfg.encoder_cfg))
        ecs = self.backbone.channels[::-1]

        self.decoder = UnetDecoder3d(encoder_channels=ecs, **vars(cfg.decoder_cfg))
        self.seg_head = SegmentationHead3d(in_channels=self.decoder.decoder_channels[-1],
                                        out_channels=cfg.seg_classes)

        if cfg.deep_supervision:
            self.aux_head = SegmentationHead3d(in_channels=ecs[0], out_channels=cfg.seg_classes)
        else:
            self.aux_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def apply_augmentations(self, batch, labels=None):
        x = batch
        y = labels
        
        if self.training and y is not None:
            if random.random() < self.cfg.mixup_p:
                x, y = self.mixup(x, y)
            x, y = rotate(x, y)
            x, y = flip_3d(x, y)

        return x, y

    def forward(self, batch, labels=None):
        x = batch.float()
        
        x, y = self.apply_augmentations(x, labels)
        
        x_feats = self.backbone.forward_features(x)
        x = x_feats[::-1]
        x = x[:len(self.cfg.decoder_cfg.decoder_channels) + 1]
        x = self.decoder(x)
        x_seg = self.seg_head(x[-1])
        
        if self.aux_head is not None:
            x_aux = self.aux_head(x_feats[-1])
        else:
            x_aux = None

        if self.training:
            return x_seg, x_aux
        else:
            return x_seg

def create_model_config(roi_size=(128, 704, 704), device='cuda'):
    """Create model configuration."""
    cfg = SimpleNamespace()
    cfg.in_chans = 1
    cfg.seg_classes = 1
    cfg.backbone = "r3d200"
    cfg.deep_supervision = True
    cfg.device = device
    cfg.roi_size = roi_size
    cfg.mixup_beta = 1.0
    cfg.mixup_p = 0.25
    cfg.cutmix_p = 0.25
    
    cfg.encoder_cfg = SimpleNamespace()
    cfg.encoder_cfg.use_checkpoint = True
    cfg.encoder_cfg.drop_path_rate = 0.2
    
    cfg.decoder_cfg = SimpleNamespace()
    cfg.decoder_cfg.decoder_channels = (256,)
    cfg.decoder_cfg.attention_type = None
    cfg.decoder_cfg.upsample_mode = "deconv"
    
    return cfg

# -----------------------------
# Training Components
# -----------------------------

class TomographyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.file_list = list(self.data_dir.glob('*.npy'))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(str(file_path))
        
        # Convert to tensor and add channel dimension
        data = torch.from_numpy(data).float()
        if data.ndim == 3:
            data = data.unsqueeze(0)
            
        if self.transform:
            data = self.transform(data)
            
        return data

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target) 
    
    return bce_weight * bce + (1 - bce_weight) * dice

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch)
        if isinstance(outputs, tuple):
            main_output, aux_output = outputs
        else:
            main_output = outputs
            aux_output = None
            
        # For this example, we're using the same data as both input and target
        # In a real scenario, you'd have separate target data
        loss = bce_dice_loss(main_output, batch)
        if aux_output is not None:
            aux_loss = bce_dice_loss(aux_output, batch)
            loss = loss + 0.4 * aux_loss
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
            
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            outputs = model(batch)
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
                
            loss = bce_dice_loss(main_output, batch)
            total_loss += loss.item()
            
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    # Configuration
    data_dir = "DATA"
    batch_size = 2
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders
    dataset = TomographyDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    cfg = create_model_config()
    model = Net(cfg).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            
        # Optional: Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()