# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained OlmoEarth v1 models."""

import importlib
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import

_olmoearth_transforms = nn.Identity()

_olmoearth_meta = {
    'dataset': 'Major TOM',
    'model': 'OlmoEarthPretrain_v1',
    'architecture': 'Vision Transformer',
    'publication': 'https://arxiv.org/abs/2506.10890',
    'repo': 'https://github.com/allenai/olmoearth_pretrain',
    'license': 'OlmoEarth Artifact License',
    'model_size': None,
    'hf_repo': None,
}


class OlmoEarthV1_Weights(WeightsEnum):
    """OlmoEarth v1 pre-trained weights.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    .. versionadded:: 0.10
    """

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/c48459cd6264704b9d1761a2904c46eb98755fda/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'nano', 'hf_repo': 'allenai/OlmoEarth-v1-Nano'},
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/edd9418badc5a9f769ba1aa622cb6d0af4586f8b/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'tiny', 'hf_repo': 'allenai/OlmoEarth-v1-Tiny'},
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/93589e2dee5b5c95a660d1e9365bc017ea7f35d6/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'base', 'hf_repo': 'allenai/OlmoEarth-v1-Base'},
    )
    LARGE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/8cf072c70d4a1c403531ca9a9653bb1f8f60eb83/weights.pth',
        transforms=_olmoearth_transforms,
        meta=_olmoearth_meta
        | {'model_size': 'large', 'hf_repo': 'allenai/OlmoEarth-v1-Large'},
    )


class _OlmoEarthLinearDecoder(nn.Module):
    """Linear probing head for OlmoEarth temporal segmentation."""

    def __init__(
        self, num_classes: int, patch_size: int, head_kernel_size: int
    ) -> None:
        """Initialize a new linear decoder instance."""
        super().__init__()
        self.patch_size = patch_size
        self.prediction_head = nn.LazyConv2d(
            out_channels=num_classes * patch_size * patch_size,
            kernel_size=head_kernel_size,
            padding=head_kernel_size // 2,
        )

    def forward(self, features: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Decode OlmoEarth features into dense logits."""
        patch_logits = self.prediction_head(features)
        y_hat = OlmoEarthTemporalSegmentation._reshape_patch_logits(
            patch_logits, self.patch_size
        )
        if y_hat.shape[-2:] != target_size:
            y_hat = F.interpolate(
                y_hat, size=target_size, mode='bilinear', align_corners=False
            )
        return y_hat


class _OlmoEarthSimpleDecoder(nn.Module):
    """Simple no-skip decoder for OlmoEarth temporal segmentation."""

    def __init__(self, num_classes: int, decoder_channels: Sequence[int]) -> None:
        """Initialize a new simple decoder instance."""
        super().__init__()

        channels = tuple(decoder_channels)
        blocks: list[nn.Module] = []
        for i, out_channels in enumerate(channels):
            conv: nn.Module
            if i == 0:
                conv = nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
            else:
                conv = nn.Conv2d(
                    channels[i - 1], out_channels, kernel_size=3, padding=1
                )
            blocks.append(nn.Sequential(conv, nn.ReLU(inplace=True)))

        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Conv2d(channels[-1], num_classes, kernel_size=3, padding=1)

    def forward(self, features: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Decode OlmoEarth features into dense logits."""
        x = features
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=False
                )

        y_hat = self.classifier(x)
        if y_hat.shape[-2:] != target_size:
            y_hat = F.interpolate(
                y_hat, size=target_size, mode='bilinear', align_corners=False
            )
        return y_hat


class OlmoEarthTemporalSegmentation(nn.Module):
    """OlmoEarth v1 backbone with a direct patch-wise segmentation head.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        head_kernel_size: int = 1,
        patch_size: int = 8,
        decoder_type: str = 'linear',
        decoder_channels: Sequence[int] = (512, 256, 128),
    ) -> None:
        """Initialize a new OlmoEarthTemporalSegmentation instance.

        Args:
            backbone: Spatiotemporal OlmoEarth backbone.
            num_classes: Number of output classes.
            head_kernel_size: Kernel size of the patch prediction head.
            patch_size: Patch size used by the OlmoEarth encoder.
            decoder_type: Decoder architecture to apply to OlmoEarth encoder features.
            decoder_channels: Output channels for each stage of the simple decoder.

        Raises:
            ValueError: If ``patch_size`` is not positive.
        """
        super().__init__()
        if patch_size <= 0:
            raise ValueError('patch_size must be positive.')
        if decoder_type not in ['linear', 'simple_unet']:
            raise ValueError(f'Unsupported decoder_type: {decoder_type}.')

        self.backbone = backbone
        self.patch_size = patch_size
        self.decoder_type = decoder_type
        self.decoder_channels = tuple(decoder_channels)

        if self.decoder_type == 'linear':
            self.decoder = _OlmoEarthLinearDecoder(
                num_classes=num_classes,
                patch_size=patch_size,
                head_kernel_size=head_kernel_size,
            )
            self.name = 'olmoearth_linear_head'
        else:
            if self.patch_size != 4:
                raise ValueError("decoder_type='simple_unet' requires patch_size=4.")
            if len(self.decoder_channels) != 3:
                raise ValueError(
                    "decoder_type='simple_unet' expects three decoder stages."
                )
            self.decoder = _OlmoEarthSimpleDecoder(
                num_classes=num_classes, decoder_channels=self.decoder_channels
            )
            self.name = 'olmoearth_simple_unet_head'

    def _extract_olmoearth_encoder_features(self, x: Tensor) -> Tensor:
        """Extract spatial features from OlmoEarth encoder tokens."""
        if x.ndim != 5:
            raise ValueError(
                f'Expected 5D input for OlmoEarth, but received shape {tuple(x.shape)}.'
            )
        if not hasattr(self.backbone, 'encoder'):
            raise ValueError('OlmoEarth backbone must define an encoder module.')

        masked_sample_module = importlib.import_module(
            'olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes'
        )
        masked_sample = getattr(masked_sample_module, 'MaskedOlmoEarthSample')

        batch_size, timesteps, channels, height, width = x.shape
        sentinel2_l2a = x.permute(0, 3, 4, 1, 2).contiguous()
        if channels != 12:
            raise ValueError(
                f'OlmoEarth expects 12 Sentinel-2 channels, but received {channels}.'
            )

        sample = masked_sample(
            timestamps=torch.zeros(
                (batch_size, timesteps, 3), device=x.device, dtype=torch.long
            ),
            sentinel2_l2a=sentinel2_l2a,
            sentinel2_l2a_mask=torch.zeros(
                (batch_size, height, width, timesteps, 3),
                device=x.device,
                dtype=torch.long,
            ),
        )
        encoder = getattr(self.backbone, 'encoder')
        encoder_output = encoder(sample, patch_size=self.patch_size)
        if not isinstance(encoder_output, dict):
            raise ValueError('OlmoEarth encoder must return a dictionary output.')

        tokens_and_masks = encoder_output.get('tokens_and_masks')
        if tokens_and_masks is None:
            raise ValueError(
                'OlmoEarth encoder output does not contain tokens_and_masks.'
            )

        return self._tokens_and_masks_to_feature_map(tokens_and_masks)

    @staticmethod
    def _tokens_and_masks_to_feature_map(tokens_and_masks: Any) -> Tensor:
        """Convert ``TokensAndMasks`` to a spatial feature map."""
        spatial_tokens: Tensor | None = None
        fields = getattr(tokens_and_masks, '_fields', ())
        for field in fields:
            if field.endswith('_mask'):
                continue
            value = getattr(tokens_and_masks, field)
            if isinstance(value, Tensor) and value.ndim >= 4:
                spatial_tokens = value
                break

        if spatial_tokens is None:
            raise ValueError(
                'OlmoEarth encoder output does not contain spatial token features.'
            )

        if spatial_tokens.ndim > 4:
            reduce_dims = tuple(range(3, spatial_tokens.ndim - 1))
            spatial_tokens = spatial_tokens.mean(dim=reduce_dims)

        if spatial_tokens.ndim != 4:
            raise ValueError(
                f'Unexpected OlmoEarth token shape: {tuple(spatial_tokens.shape)}.'
            )

        return spatial_tokens.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _reshape_patch_logits(logits: Tensor, patch_size: int) -> Tensor:
        """Reshape patch logits into dense per-pixel logits.

        Args:
            logits: Tensor of shape ``(B, C * P * P, H, W)``.
            patch_size: Patch size ``P``.

        Returns:
            Tensor of shape ``(B, C, H * P, W * P)``.

        Raises:
            ValueError: If ``logits`` does not have 4 dimensions or the channel
                count is incompatible with ``patch_size``.
        """
        if logits.ndim != 4:
            raise ValueError(
                f'Expected 4D logits, but received shape {tuple(logits.shape)}.'
            )

        patch_area = patch_size * patch_size
        if logits.shape[1] % patch_area != 0:
            raise ValueError(
                'Prediction head output channels must be divisible by patch_size**2.'
            )

        return rearrange(
            logits, 'b (c ph pw) h w -> b c (h ph) (w pw)', ph=patch_size, pw=patch_size
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W).
            lengths: Optional sequence lengths. Temporal aggregation is handled
                internally by the OlmoEarth encoder.

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        del lengths
        target_size = x.shape[-2:]
        features = self._extract_olmoearth_encoder_features(x)
        return self.decoder(features, target_size)


def olmoearth_v1(
    weights: OlmoEarthV1_Weights | None = None, **kwargs: Any
) -> nn.Module:
    """OlmoEarth v1 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2511.13655

    This model requires the following additional library to be installed:

    * `olmoearth-pretrain-minimal <https://pypi.org/project/olmoearth-pretrain-minimal/>`_:
      to load the models.

    .. versionadded:: 0.10

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        **kwargs: Passed to
            ``olmoearth_pretrain_minimal.OlmoEarthPretrain_v1``
            (e.g. ``model_size``, ``max_patch_size``).

    Returns:
        An OlmoEarth v1 model.
    """
    olmoearth = lazy_import('olmoearth_pretrain_minimal')

    model_size = kwargs.pop('model_size', 'nano')
    if weights is not None:
        model_size = weights.meta.get('model_size', model_size)
    model: nn.Module = olmoearth.OlmoEarthPretrain_v1(model_size=model_size, **kwargs)
    if weights is not None:
        state_dict = weights.get_state_dict(progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def _verify_olmoearth_weights(
    weights: OlmoEarthV1_Weights | str | None,
) -> OlmoEarthV1_Weights | None:
    """Verify that weights are valid OlmoEarth weights."""
    if isinstance(weights, str):
        name = weights.removeprefix(f'{OlmoEarthV1_Weights.__name__}.')
        try:
            return OlmoEarthV1_Weights[name]
        except KeyError as error:
            raise ValueError(
                f"Weights '{weights}' are not valid for model='olmoearth_v1'."
            ) from error
    if weights is not None and not isinstance(weights, OlmoEarthV1_Weights):
        raise ValueError(f"Weights '{weights}' are not valid for model='olmoearth_v1'.")
    return weights


def olmoearth_v1_temporal_segmentation(
    *,
    weights: OlmoEarthV1_Weights | str | None = None,
    num_classes: int | None = None,
    num_labels: int | None = None,
    model_size: str | None = None,
    max_patch_size: int | None = None,
    max_sequence_length: int | None = None,
    patch_size: int | None = None,
    decoder_type: str = 'linear',
    decoder_channels: Sequence[int] | None = None,
    head_kernel_size: int | None = None,
    freeze_backbone: bool = False,
    **kwargs: Any,
) -> OlmoEarthTemporalSegmentation:
    """OlmoEarth v1 temporal segmentation model.

    Args:
        weights: Pre-trained weights. If ``None``, model is randomly initialized.
        num_classes: Number of prediction classes.
        num_labels: Number of prediction labels.
        model_size: OlmoEarth model size.
        max_patch_size: Maximum patch size supported by the OlmoEarth encoder.
        max_sequence_length: Maximum sequence length supported by the OlmoEarth encoder.
        patch_size: Patch size used by the OlmoEarth encoder.
        decoder_type: Decoder architecture to apply to OlmoEarth encoder features.
        decoder_channels: Output channels for each stage of the simple decoder.
        head_kernel_size: Kernel size of the patch prediction head.
        freeze_backbone: Freeze the OlmoEarth backbone.
        **kwargs: Ignored trainer keyword arguments.

    Returns:
        An OlmoEarth v1 temporal segmentation model.

    Raises:
        ValueError: If incompatible decoder options or weights are provided.
    """
    del kwargs
    weights = _verify_olmoearth_weights(weights)
    if decoder_type == 'simple_unet' and head_kernel_size is not None:
        raise ValueError(
            "head_kernel_size is only supported for decoder_type='linear'."
        )
    if decoder_type == 'linear' and decoder_channels is not None:
        raise ValueError(
            "decoder_channels is only supported for decoder_type='simple_unet'."
        )

    olmoearth_kwargs: dict[str, Any] = {}
    for key, value in {
        'model_size': model_size,
        'max_patch_size': max_patch_size,
        'max_sequence_length': max_sequence_length,
    }.items():
        if value is not None:
            olmoearth_kwargs[key] = value

    model = OlmoEarthTemporalSegmentation(
        backbone=olmoearth_v1(weights=weights, **olmoearth_kwargs),
        num_classes=num_classes or num_labels or 1,
        head_kernel_size=1 if head_kernel_size is None else head_kernel_size,
        patch_size=(max_patch_size or 8) if patch_size is None else patch_size,
        decoder_type=decoder_type,
        decoder_channels=(512, 256, 128)
        if decoder_channels is None
        else decoder_channels,
    )
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model
