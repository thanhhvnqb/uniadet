from unittest.mock import patch

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from uniadet.backbones import DINOv3HFEncoder, create_encoder


def tiny_model():
    return DINOv3ViTModel(DINOv3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, patch_size=16, num_register_tokens=4,
    )).eval()


def test_dinov3_auto_download_and_features():
    model = tiny_model()
    with patch('transformers.AutoModel.from_pretrained', return_value=model) as load:
        encoder = create_encoder('dinov3', [1, 2])
    load.assert_called_once_with('facebook/dinov3-vitl16-pretrain-lvd1689m')
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    assert encoder.dims == [32, 32]
    images = torch.randn(2, 3, 32, 48, requires_grad=True)
    outputs = encoder(images)
    reference = model(images).last_hidden_state
    for cls, patches, grid in outputs:
        assert cls.shape == (2, 32)
        assert patches.shape == (2, 6, 32)
        assert grid == (2, 3)
        assert not patches.requires_grad
    torch.testing.assert_close(outputs[-1][0], reference[:, 0])
    torch.testing.assert_close(outputs[-1][1], reference[:, 5:])
    with pytest.raises(ValueError, match='divisible'):
        encoder(torch.randn(1, 3, 33, 32))
    for layers in ([], [0], [2, 1], [1, 1], [3]):
        with pytest.raises(ValueError, match='layers'):
            DINOv3HFEncoder(model, layers)


def test_dinov3_access_error():
    with patch('transformers.AutoModel.from_pretrained', side_effect=OSError('gated')):
        with pytest.raises(RuntimeError, match='hf auth login'):
            create_encoder('dinov3', [1])


@pytest.mark.parametrize('repo', [None, '/local/dinov3'])
def test_dinov3_explicit_weights(repo):
    from torch import nn
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.embed_dim = 32
    with patch('torch.hub.load', return_value=model) as load:
        encoder = create_encoder('dinov3', [1, 2], weights='/weights/model.pth', dino_repo=repo)
    load.assert_called_once_with(repo or 'facebookresearch/dinov3', 'dinov3_vitl16',
                                 source='local' if repo else 'github', weights='/weights/model.pth')
    assert not encoder.training
