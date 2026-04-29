# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.models import (
    ConvLSTM,
    OlmoEarthTemporalSegmentation,
    OlmoEarthV1_Weights,
    ResNet18_Weights,
)
from torchgeo.trainers import SpatioTemporalSegmentationTask


class _FakeOlmoEarthBackbone(nn.Module):
    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(1))
        self.seen_sample: object | None = None
        self.seen_patch_size: int | None = None

    def encoder(self, sample: object, patch_size: int) -> dict[str, object]:
        self.seen_sample = sample
        self.seen_patch_size = patch_size
        sentinel2_l2a = sample.sentinel2_l2a
        batch_size, height, width, timesteps, _channels = sentinel2_l2a.shape
        patch_h = max(height // patch_size, 1)
        patch_w = max(width // patch_size, 1)
        tokens_and_masks = type(
            'TokensAndMasks',
            (),
            {
                '_fields': ('sentinel2_l2a', 'sentinel2_l2a_mask'),
                'sentinel2_l2a': torch.randn(
                    batch_size, patch_h, patch_w, timesteps, 1, self.embed_dim
                ),
                'sentinel2_l2a_mask': torch.zeros(
                    batch_size, patch_h, patch_w, timesteps, 1
                ),
            },
        )()
        return {'tokens_and_masks': tokens_and_masks}


def _mock_masked_sample_module(monkeypatch: MonkeyPatch) -> None:
    module = SimpleNamespace(
        MaskedOlmoEarthSample=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        'torchgeo.models.olmoearth.importlib.import_module', lambda _name: module
    )


class TestSpatioTemporalSegmentationTask:
    def test_trainer_with_pastis100_config(self, fast_dev_run: bool) -> None:
        config = os.path.join('tests', 'conf', 'pastis.yaml')

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    @pytest.fixture
    def create_spatiotemporal_model(
        self,
    ) -> Callable[..., SpatioTemporalSegmentationTask]:
        def _create_spatiotemporal_model(
            **kwargs: Any,
        ) -> SpatioTemporalSegmentationTask:
            model = SpatioTemporalSegmentationTask(hidden_dim=8, num_layers=1, **kwargs)
            # Avoid Lightning warnings when calling step hooks without a Trainer.
            setattr(model, 'log', lambda *args, **kwargs: None)
            setattr(model, 'log_dict', lambda *args, **kwargs: None)
            return model

        return _create_spatiotemporal_model

    def test_spatiotemporal_forward_defaults_to_convlstm(self) -> None:
        model = SpatioTemporalSegmentationTask(in_channels=3, num_classes=5)
        assert model.hparams['model'] == 'convlstm'
        assert isinstance(model.model, ConvLSTM)
        assert model.model.head is not None

    def test_spatiotemporal_forward_supports_direct_convlstm_kwargs(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, num_classes=5, hidden_dim=8, num_layers=1
        )
        assert isinstance(model.model, ConvLSTM)
        assert model.model.hidden_dim == [8]

    def test_spatiotemporal_direct_kwargs_are_saved_in_hparams(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, num_classes=5, hidden_dim=8, num_layers=1
        )

        assert model.hparams['hidden_dim'] == 8
        assert model.hparams['num_layers'] == 1
        assert 'kwargs' not in model.hparams

    def test_spatiotemporal_forward_supports_olmoearth_v1(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1',
            lambda **kwargs: _FakeOlmoEarthBackbone(),
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1', in_channels=12, num_classes=4
        )
        batch = {'image': torch.randn(2, 5, 12, 32, 32), 'length': torch.tensor([5, 4])}
        y_hat = model(batch['image'], lengths=batch['length'])
        assert y_hat.shape == (2, 4, 32, 32)

    def test_spatiotemporal_olmoearth_builder_kwargs(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        seen_kwargs: dict[str, Any] = {}

        def _fake_olmoearth_v1(**kwargs: Any) -> nn.Module:
            seen_kwargs.update(kwargs)
            return _FakeOlmoEarthBackbone()

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1', _fake_olmoearth_v1
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1',
            in_channels=12,
            num_classes=2,
            model_size='tiny',
            head_kernel_size=3,
        )
        y_hat = model(torch.randn(2, 4, 12, 32, 32))

        assert seen_kwargs['model_size'] == 'tiny'
        assert 'head_kernel_size' not in seen_kwargs
        assert y_hat.shape == (2, 2, 32, 32)

    def test_spatiotemporal_olmoearth_simple_unet_kwargs(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        seen_kwargs: dict[str, Any] = {}

        def _fake_olmoearth_v1(**kwargs: Any) -> nn.Module:
            seen_kwargs.update(kwargs)
            return _FakeOlmoEarthBackbone()

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1', _fake_olmoearth_v1
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1',
            in_channels=12,
            num_classes=2,
            model_size='tiny',
            max_patch_size=4,
            patch_size=4,
            decoder_type='simple_unet',
            decoder_channels=(16, 8, 4),
        )
        y_hat = model(torch.randn(2, 4, 12, 32, 32))

        assert seen_kwargs['model_size'] == 'tiny'
        assert seen_kwargs['max_patch_size'] == 4
        assert 'patch_size' not in seen_kwargs
        assert 'decoder_type' not in seen_kwargs
        assert 'decoder_channels' not in seen_kwargs
        assert isinstance(model.model, OlmoEarthTemporalSegmentation)
        assert model.model.decoder_type == 'simple_unet'
        assert y_hat.shape == (2, 2, 32, 32)

    def test_spatiotemporal_olmoearth_patch_size_kwargs(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        seen_kwargs: dict[str, Any] = {}

        def _fake_olmoearth_v1(**kwargs: Any) -> nn.Module:
            seen_kwargs.update(kwargs)
            return _FakeOlmoEarthBackbone()

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1', _fake_olmoearth_v1
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1',
            in_channels=12,
            num_classes=2,
            model_size='tiny',
            max_patch_size=16,
            patch_size=4,
        )
        y_hat = model(torch.randn(2, 4, 12, 32, 32))

        assert seen_kwargs['model_size'] == 'tiny'
        assert seen_kwargs['max_patch_size'] == 16
        assert 'patch_size' not in seen_kwargs
        assert isinstance(model.model, OlmoEarthTemporalSegmentation)
        assert model.model.patch_size == 4
        assert y_hat.shape == (2, 2, 32, 32)

    def test_spatiotemporal_olmoearth_saves_direct_kwargs_in_hparams(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        seen_kwargs_list: list[dict[str, Any]] = []

        def _fake_olmoearth_v1(**kwargs: Any) -> nn.Module:
            seen_kwargs_list.append(dict(kwargs))
            return _FakeOlmoEarthBackbone()

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1', _fake_olmoearth_v1
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1',
            in_channels=12,
            num_classes=2,
            model_size='tiny',
            max_patch_size=128,
            max_sequence_length=12,
            patch_size=4,
            head_kernel_size=3,
        )
        y_hat = model(torch.randn(2, 4, 12, 32, 32))

        latest_kwargs = seen_kwargs_list[-1]
        assert model.hparams['model_size'] == 'tiny'
        assert model.hparams['max_patch_size'] == 128
        assert model.hparams['max_sequence_length'] == 12
        assert model.hparams['patch_size'] == 4
        assert model.hparams['head_kernel_size'] == 3
        assert 'kwargs' not in model.hparams
        assert latest_kwargs['model_size'] == 'tiny'
        assert latest_kwargs['max_patch_size'] == 128
        assert latest_kwargs['max_sequence_length'] == 12
        assert 'patch_size' not in latest_kwargs
        assert 'head_kernel_size' not in latest_kwargs
        assert isinstance(model.model, OlmoEarthTemporalSegmentation)
        assert model.model.patch_size == 4
        assert y_hat.shape == (2, 2, 32, 32)

    def test_spatiotemporal_olmoearth_freeze_backbone(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1',
            lambda **kwargs: _FakeOlmoEarthBackbone(),
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1', in_channels=12, num_classes=2, freeze_backbone=True
        )

        assert all(
            not param.requires_grad for param in model.model.backbone.parameters()
        )

    def test_spatiotemporal_olmoearth_rejects_head_kernel_size_for_simple_unet(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1',
            lambda **kwargs: _FakeOlmoEarthBackbone(),
        )

        with pytest.raises(
            ValueError,
            match="head_kernel_size is only supported for decoder_type='linear'",
        ):
            SpatioTemporalSegmentationTask(
                model='olmoearth_v1',
                in_channels=12,
                num_classes=2,
                decoder_type='simple_unet',
                patch_size=4,
                head_kernel_size=3,
            )

    def test_spatiotemporal_olmoearth_rejects_decoder_channels_for_linear(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1',
            lambda **kwargs: _FakeOlmoEarthBackbone(),
        )

        with pytest.raises(
            ValueError,
            match="decoder_channels is only supported for decoder_type='simple_unet'",
        ):
            SpatioTemporalSegmentationTask(
                model='olmoearth_v1',
                in_channels=12,
                num_classes=2,
                decoder_channels=(16, 8, 4),
            )

    def test_spatiotemporal_olmoearth_supports_weight_enum(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        seen_kwargs: dict[str, Any] = {}

        def _fake_olmoearth_v1(**kwargs: Any) -> nn.Module:
            seen_kwargs.update(kwargs)
            return _FakeOlmoEarthBackbone()

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1', _fake_olmoearth_v1
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1',
            weights=OlmoEarthV1_Weights.TINY,
            in_channels=12,
            num_classes=2,
        )
        y_hat = model(torch.randn(2, 4, 12, 32, 32))

        assert seen_kwargs['weights'] is OlmoEarthV1_Weights.TINY
        assert y_hat.shape == (2, 2, 32, 32)

    def test_spatiotemporal_olmoearth_supports_weight_string(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        _mock_masked_sample_module(monkeypatch)
        seen_kwargs: dict[str, Any] = {}

        def _fake_olmoearth_v1(**kwargs: Any) -> nn.Module:
            seen_kwargs.update(kwargs)
            return _FakeOlmoEarthBackbone()

        monkeypatch.setattr(
            'torchgeo.models.olmoearth.olmoearth_v1', _fake_olmoearth_v1
        )
        model = SpatioTemporalSegmentationTask(
            model='olmoearth_v1',
            weights=str(OlmoEarthV1_Weights.TINY),
            in_channels=12,
            num_classes=2,
        )
        y_hat = model(torch.randn(2, 4, 12, 32, 32))

        assert seen_kwargs['weights'] is OlmoEarthV1_Weights.TINY
        assert y_hat.shape == (2, 2, 32, 32)

    def test_spatiotemporal_olmoearth_rejects_non_olmo_weights(self) -> None:
        with pytest.raises(ValueError, match="are not valid for model='olmoearth_v1'"):
            SpatioTemporalSegmentationTask(
                model='olmoearth_v1',
                weights=str(ResNet18_Weights.SENTINEL2_ALL_MOCO),
                in_channels=3,
                num_classes=2,
            )

    def test_convlstm_timeseries_forward_and_step(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            model='convlstm', in_channels=10, num_classes=5, task='multiclass'
        )
        batch = {
            'image': torch.randn(2, 7, 10, 16, 16),
            'mask': torch.randint(0, 5, (2, 16, 16)),
            'length': torch.tensor([7, 5]),
        }
        y_hat = model(batch['image'], lengths=batch['length'])
        assert y_hat.shape == (2, 5, 16, 16)

        # If no lengths are provided, the model uses the last timestep.
        # This should match the explicit `lengths=T` case.
        y_hat_no_lengths = model(batch['image'])
        y_hat_last_step = model(batch['image'], lengths=torch.tensor([7, 7]))
        torch.testing.assert_close(y_hat_no_lengths, y_hat_last_step)

        # Lengths longer than the available sequence should clamp to the
        # final timestep instead of indexing out of bounds.
        y_hat_clamped = model(batch['image'], lengths=torch.tensor([9.0, 12.0]))
        torch.testing.assert_close(y_hat_no_lengths, y_hat_clamped)

        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_ce_class_weights_from_sequence(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3, num_classes=2, task='multiclass', class_weights=[1.0, 2.0]
        )

        assert isinstance(model.criterion, nn.CrossEntropyLoss)
        torch.testing.assert_close(
            model.criterion.weight, torch.tensor([1.0, 2.0], dtype=torch.float32)
        )

    @pytest.mark.parametrize(
        ('loss', 'expected_type'),
        [('jaccard', smp.losses.JaccardLoss), ('focal', smp.losses.FocalLoss)],
    )
    def test_alternate_losses(
        self,
        create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask],
        loss: str,
        expected_type: type[nn.Module],
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3, num_classes=3, task='multiclass', loss=loss, ignore_index=1
        )

        assert isinstance(model.criterion, expected_type)

    def test_binary_steps_and_predict_step(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(in_channels=3, task='binary', loss='bce')
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'mask': torch.randint(0, 2, (2, 16, 16)),
            'length': torch.tensor([4, 2]),
        }

        train_loss = model.training_step(batch, 0)
        assert train_loss.ndim == 0

        assert model.validation_step(batch, 0) is None
        assert model.test_step(batch, 0) is None

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 1, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)

    def test_multiclass_predict_step(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3, num_classes=4, task='multiclass'
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones((2, 16, 16)), atol=1e-5, rtol=1e-5
        )

    def test_multiclass_classwise_metrics(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3,
            num_classes=3,
            task='multiclass',
            labels=['background', 'crop', 'water'],
        )
        y_hat = torch.randn(2, 3, 16, 16)
        y = torch.randint(0, 3, (2, 16, 16))

        model.val_metrics(y_hat, y)
        metrics = model.val_metrics.compute()

        assert 'val_OverallAccuracy' in metrics
        assert 'val_AverageJaccardIndex' in metrics
        assert 'val_ClasswiseAccuracy_background' in metrics
        assert 'val_ClasswiseAccuracy_crop' in metrics
        assert 'val_ClasswiseAccuracy_water' in metrics
        assert 'val_ClasswiseJaccardIndex_background' in metrics
