import torch
import torch.nn as nn
import torchvision.models as models

def get_finetune_resnet_model(num_classes=3, pretrained=True, grayscale=True, device=None):

    model = models.resnet50(pretrained=pretrained)

    if grayscale:
        # Modify first conv layer to accept 1 channel instead of 3
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)

    # Replace fully connected layer for classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)

# part1_frame_classification/src/models_gray.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# Luminance coefficients (BT.601)
_LUMA = (0.2989, 0.5870, 0.1140)

def _to_gray_conv2d(conv3: nn.Conv2d) -> nn.Conv2d:
    """Convert a 3-in Conv2d to a 1-in Conv2d using luminance mixing of RGB kernels."""
    new = nn.Conv2d(
        in_channels=1, out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size, stride=conv3.stride,
        padding=conv3.padding, dilation=conv3.dilation,
        groups=conv3.groups, bias=(conv3.bias is not None),
        padding_mode=conv3.padding_mode,
    )
    with torch.no_grad():
        W = conv3.weight  # [C_out, 3, k, k]
        Wg = _LUMA[0]*W[:, 0:1] + _LUMA[1]*W[:, 1:2] + _LUMA[2]*W[:, 2:3]
        new.weight.copy_(Wg)
        if conv3.bias is not None:
            new.bias.copy_(conv3.bias)
    return new

def get_finetune_efficientnetv2_s_gray(num_classes: int = 3,
                                       pretrained: bool = True,
                                       device=None) -> nn.Module:
    """
    EfficientNet-V2-S with a 1-channel grayscale stem (luminance remap) and 3-way head.
    Keeps the rest of the pretrained weights intact.
    """
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)

    # First conv is model.features[0][0]
    assert isinstance(model.features[0][0], nn.Conv2d)
    model.features[0][0] = _to_gray_conv2d(model.features[0][0])

    # Classifier: keep dropout, replace last Linear
    in_feats = model.classifier[1].in_features
    p = getattr(model.classifier[0], "p", 0.2)
    model.classifier = nn.Sequential(
        nn.Dropout(p=p, inplace=True),
        nn.Linear(in_feats, num_classes)
    )
    nn.init.zeros_(model.classifier[1].bias)

    return model.to(device) if device is not None else model


# src/models_gray.py
import torch
import torch.nn as nn
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    regnet_y_8gf, RegNet_Y_8GF_Weights,
)

_LUMA = (0.2989, 0.5870, 0.1140)


def _to_gray_conv2d(conv3: nn.Conv2d) -> nn.Conv2d:
    """Convert a 3-in Conv2d to a 1-in Conv2d using luminance mixing of the RGB kernels."""
    new = nn.Conv2d(
        in_channels=1, out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size, stride=conv3.stride,
        padding=conv3.padding, dilation=conv3.dilation,
        groups=conv3.groups, bias=(conv3.bias is not None),
        padding_mode=conv3.padding_mode,
    )
    with torch.no_grad():
        W = conv3.weight  # [C_out, 3, k, k]
        Wg = _LUMA[0]*W[:, 0:1] + _LUMA[1]*W[:, 1:2] + _LUMA[2]*W[:, 2:3]
        new.weight.copy_(Wg)
        if conv3.bias is not None:
            new.bias.copy_(conv3.bias)
    return new

# -------------------------
# 1) DenseNet-121 
# -------------------------
def get_finetune_densenet121_gray(num_classes: int = 2,
                                  pretrained: bool = True,
                                  device=None) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    # First conv: features.conv0
    model.features.conv0 = _to_gray_conv2d(model.features.conv0)
    # Classifier
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    nn.init.zeros_(model.classifier.bias)
    return model.to(device) if device is not None else model


def get_finetune_efficientnet_v2_gray(num_classes: int = 2, 
                                    pretrained: bool = True,
                                    device=None) -> nn.Module:
    """Better baseline for medical grayscale images"""
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    
    # Convert first conv layer to grayscale
    model.features[0][0] = _to_gray_conv2d(model.features[0][0])
    
    # Replace classifier with medical-optimized head
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, 256),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    # Initialize new layers properly
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            
    return model.to(device) if device else model

import torch
import torch.nn as nn
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# Grayscale "ImageNet-like" stats induced by luminance (0.2989, 0.5870, 0.1140)
IMAGENET_GRAY_MEAN = 0.4589225
IMAGENET_GRAY_STD  = 0.15043988512894443

def get_finetune_convnext_small(num_classes: int = 3,
                                pretrained: bool = True,
                                device=None) -> nn.Module:
    """
    ConvNeXt-Small with a 1-channel grayscale stem that preserves pretrained signal
    using luminance mixing, and a custom classifier head.
    """
    weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
    model = convnext_small(weights=weights)

    # --- replace first conv (3->96) with luminance-mixed 1->96 ---
    # torchvision convnext_small: model.features[0][0] is Conv2d(3,96,4,4)
    first = model.features[0][0]
    assert isinstance(first, nn.Conv2d) and first.in_channels == 3, "Unexpected ConvNeXt layout."

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups,
        bias=(first.bias is not None),
        padding_mode=first.padding_mode,
    )

    with torch.no_grad():
        # luminance (BT.601) instead of naive average
        # W_luma = 0.2989*R + 0.5870*G + 0.1140*B
        W = first.weight  # [96, 3, k, k]
        W_gray = 0.2989 * W[:, 0:1, ...] + 0.5870 * W[:, 1:2, ...] + 0.1140 * W[:, 2:3, ...]
        new_conv.weight.copy_(W_gray)
        if first.bias is not None:
            new_conv.bias.copy_(first.bias)

    model.features[0][0] = new_conv

    # --- replace classifier head ---
    in_feats = model.classifier[-1].in_features
    head = nn.Linear(in_feats, num_classes)
    # (Optional) Better starting bias: zero init (or set to log-priors later in the training script)
    nn.init.zeros_(head.bias)
    model.classifier[-1] = head

    return model.to(device) if device is not None else model

