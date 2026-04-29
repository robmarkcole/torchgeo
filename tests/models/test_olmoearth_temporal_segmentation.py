# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.models import (
    OlmoEarthTemporalSegmentation,
    olmoearth_v1_temporal_segmentation,
)


def _create_tokens_and_masks(
    batch_size: int = 2,
    height: int = 4,
    width: int = 4,
    timesteps: int = 3,
    embed_dim: int = 8,
) -> object:
    return type(
        'TokensAndMasks',
        (),
        {
            '_fields': ('sentinel2_l2a', 'sentinel2_l2a_mask'),
            'sentinel2_l2a': torch.randn(
                batch_size, height, width, timesteps, 1, embed_dim
            ),
            'sentinel2_l2a_mask': torch.zeros(batch_size, height, width, timesteps, 1),
        },
    )()


class _EncoderBackbone(nn.Module):
    def __init__(self, encoder_output: object) -> None:
        super().__init__()
        self.encoder_output = encoder_output
        self.seen_sample: object | None = None
        self.seen_patch_size: int | None = None

    def encoder(self, sample: object, patch_size: int) -> object:
        self.seen_sample = sample
        self.seen_patch_size = patch_size
        return self.encoder_output


class _NoEncoderBackbone(nn.Module):
    pass


def _mock_masked_sample_module(monkeypatch: MonkeyPatch) -> None:
    module = SimpleNamespace(
        MaskedOlmoEarthSample=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        'torchgeo.models.olmoearth.importlib.import_module', lambda _name: module
    )


class TestOlmoEarthTemporalSegmentation:
    def test_rejects_invalid_weight_object(self) -> None:
        with pytest.raises(
            ValueError, match="Weights '0' are not valid for model='olmoearth_v1'"
        ):
            olmoearth_v1_temporal_segmentation(weights=cast(Any, 0))

    def test_requires_positive_patch_size(self) -> None:
        with pytest.raises(ValueError, match='patch_size must be positive'):
            OlmoEarthTemporalSegmentation(
                backbone=_EncoderBackbone({}), num_classes=2, patch_size=0
            )

    def test_rejects_invalid_decoder_type(self) -> None:
        with pytest.raises(ValueError, match='Unsupported decoder_type'):
            OlmoEarthTemporalSegmentation(
                backbone=_EncoderBackbone({}),
                num_classes=2,
                decoder_type=cast(Any, 'invalid'),
            )

    def test_simple_unet_requires_patch_size_4(self) -> None:
        with pytest.raises(ValueError, match='requires patch_size=4'):
            OlmoEarthTemporalSegmentation(
                backbone=_EncoderBackbone({}),
                num_classes=2,
                decoder_type='simple_unet',
                patch_size=8,
            )

    def test_simple_unet_requires_three_decoder_stages(self) -> None:
        with pytest.raises(ValueError, match='expects three decoder stages'):
            OlmoEarthTemporalSegmentation(
                backbone=_EncoderBackbone({}),
                num_classes=2,
                decoder_type='simple_unet',
                patch_size=4,
                decoder_channels=(16, 8),
            )

    def test_forward_uses_olmoearth_encoder_features(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        backbone = _EncoderBackbone(
            encoder_output={'tokens_and_masks': _create_tokens_and_masks()}
        )
        model = OlmoEarthTemporalSegmentation(
            backbone=backbone, num_classes=3, patch_size=16
        )
        x = torch.randn(2, 5, 12, 32, 32)
        y_hat = model(x, lengths=torch.tensor([5, 4]))

        assert backbone.seen_patch_size == 16
        assert isinstance(backbone.seen_sample, SimpleNamespace)
        assert y_hat.shape == (2, 3, 32, 32)

    def test_forward_supports_simple_unet_decoder(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        backbone = _EncoderBackbone(
            encoder_output={
                'tokens_and_masks': _create_tokens_and_masks(height=8, width=8)
            }
        )
        model = OlmoEarthTemporalSegmentation(
            backbone=backbone,
            num_classes=3,
            patch_size=4,
            decoder_type='simple_unet',
            decoder_channels=(16, 8, 4),
        )
        x = torch.randn(2, 5, 12, 32, 32)

        y_hat = model(x)

        assert backbone.seen_patch_size == 4
        assert model.name == 'olmoearth_simple_unet_head'
        assert y_hat.shape == (2, 3, 32, 32)

    def test_reshape_patch_logits_rearranges_patch_axes(self) -> None:
        patch_logits = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)

        y_hat = OlmoEarthTemporalSegmentation._reshape_patch_logits(
            patch_logits, patch_size=2
        )

        expected = torch.tensor(
            [[[[0, 4, 1, 5], [8, 12, 9, 13], [2, 6, 3, 7], [10, 14, 11, 15]]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(y_hat, expected)

    def test_reshape_patch_logits_requires_4d_input(self) -> None:
        patch_logits = torch.randn(2, 5, 4)

        with pytest.raises(ValueError, match='Expected 4D logits'):
            _ = OlmoEarthTemporalSegmentation._reshape_patch_logits(
                patch_logits, patch_size=2
            )

    def test_reshape_patch_logits_requires_compatible_channels(self) -> None:
        patch_logits = torch.randn(2, 5, 4, 4)

        with pytest.raises(ValueError, match='must be divisible by patch_size\\*\\*2'):
            _ = OlmoEarthTemporalSegmentation._reshape_patch_logits(
                patch_logits, patch_size=2
            )

    def test_forward_interpolates_patch_logits_to_target_size(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        backbone = _EncoderBackbone(
            encoder_output={'tokens_and_masks': _create_tokens_and_masks()}
        )
        model = OlmoEarthTemporalSegmentation(
            backbone=backbone, num_classes=3, patch_size=8
        )
        x = torch.randn(2, 5, 12, 30, 26)

        y_hat = model(x)

        assert y_hat.shape == (2, 3, 30, 26)

    def test_forward_interpolates_simple_unet_logits_to_target_size(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        backbone = _EncoderBackbone(
            encoder_output={
                'tokens_and_masks': _create_tokens_and_masks(height=7, width=6)
            }
        )
        model = OlmoEarthTemporalSegmentation(
            backbone=backbone,
            num_classes=3,
            patch_size=4,
            decoder_type='simple_unet',
            decoder_channels=(16, 8, 4),
        )
        x = torch.randn(2, 5, 12, 30, 26)

        y_hat = model(x)

        assert y_hat.shape == (2, 3, 30, 26)

    def test_extract_olmoearth_encoder_features_requires_5d_input(self) -> None:
        model = OlmoEarthTemporalSegmentation(
            backbone=_EncoderBackbone({}), num_classes=2
        )
        x = torch.randn(2, 3, 32, 32)

        with pytest.raises(ValueError, match='Expected 5D input for OlmoEarth'):
            _ = model._extract_olmoearth_encoder_features(x)

    def test_extract_olmoearth_encoder_features_requires_encoder_module(self) -> None:
        model = OlmoEarthTemporalSegmentation(
            backbone=_NoEncoderBackbone(), num_classes=2
        )
        x = torch.randn(2, 4, 12, 32, 32)

        with pytest.raises(ValueError, match='must define an encoder module'):
            _ = model._extract_olmoearth_encoder_features(x)

    def test_extract_olmoearth_encoder_features_rejects_invalid_channels(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        model = OlmoEarthTemporalSegmentation(
            backbone=_EncoderBackbone({}), num_classes=2
        )
        x = torch.randn(2, 4, 10, 32, 32)

        with pytest.raises(ValueError, match='expects 12 Sentinel-2 channels'):
            _ = model._extract_olmoearth_encoder_features(x)

    def test_extract_olmoearth_encoder_features_rejects_non_dict_encoder_output(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        backbone = _EncoderBackbone(encoder_output=torch.randn(2, 4, 4, 8))
        model = OlmoEarthTemporalSegmentation(
            backbone=backbone, num_classes=2, patch_size=16
        )
        x = torch.randn(2, 4, 12, 32, 32)

        with pytest.raises(ValueError, match='encoder must return a dictionary'):
            _ = model._extract_olmoearth_encoder_features(x)

    def test_extract_olmoearth_encoder_features_rejects_missing_tokens_and_masks(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        backbone = _EncoderBackbone(encoder_output={})
        model = OlmoEarthTemporalSegmentation(backbone=backbone, num_classes=2)
        x = torch.randn(2, 4, 12, 32, 32)

        with pytest.raises(ValueError, match='does not contain tokens_and_masks'):
            _ = model._extract_olmoearth_encoder_features(x)

    def test_tokens_and_masks_to_feature_map_reduces_temporal_dims(self) -> None:
        feature_map = OlmoEarthTemporalSegmentation._tokens_and_masks_to_feature_map(
            _create_tokens_and_masks()
        )
        assert feature_map.shape == (2, 8, 4, 4)

    def test_tokens_and_masks_to_feature_map_skips_mask_fields(self) -> None:
        tokens_and_masks = type(
            'TokensAndMasks',
            (),
            {
                '_fields': ('sentinel2_l2a_mask', 'sentinel2_l2a'),
                'sentinel2_l2a_mask': torch.zeros(2, 4, 4, 3, 1),
                'sentinel2_l2a': torch.randn(2, 4, 4, 3, 1, 8),
            },
        )()

        feature_map = OlmoEarthTemporalSegmentation._tokens_and_masks_to_feature_map(
            tokens_and_masks
        )
        assert feature_map.shape == (2, 8, 4, 4)

    def test_tokens_and_masks_to_feature_map_requires_spatial_tokens(self) -> None:
        tokens_and_masks = type(
            'TokensAndMasks',
            (),
            {
                '_fields': ('sentinel2_l2a', 'sentinel2_l2a_mask'),
                'sentinel2_l2a': torch.randn(2, 8),
                'sentinel2_l2a_mask': torch.zeros(2, 8),
            },
        )()

        with pytest.raises(ValueError, match='does not contain spatial token features'):
            _ = OlmoEarthTemporalSegmentation._tokens_and_masks_to_feature_map(
                tokens_and_masks
            )

    def test_tokens_and_masks_to_feature_map_rejects_unexpected_ndim(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        tokens_and_masks = type(
            'TokensAndMasks',
            (),
            {
                '_fields': ('sentinel2_l2a',),
                'sentinel2_l2a': torch.randn(2, 4, 4, 3, 2, 8),
            },
        )()
        original_mean = torch.Tensor.mean

        def _mean_with_bad_shape(
            self: torch.Tensor,
            dim: int | tuple[int, ...] | None = None,
            keepdim: bool = False,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            del self, dim, keepdim, dtype
            return torch.randn(2, 3, 4)

        monkeypatch.setattr(torch.Tensor, 'mean', _mean_with_bad_shape)

        try:
            with pytest.raises(ValueError, match='Unexpected OlmoEarth token shape'):
                _ = OlmoEarthTemporalSegmentation._tokens_and_masks_to_feature_map(
                    tokens_and_masks
                )
        finally:
            monkeypatch.setattr(torch.Tensor, 'mean', original_mean)
