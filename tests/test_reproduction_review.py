from unittest.mock import patch

import pytest
import torch

from uniadet import cli
from uniadet.model import UniADet, NormalMemory, smooth_maps
from test_uniadet import TinyEncoder


@pytest.mark.parametrize('pooling',['layer','map'])
@pytest.mark.parametrize('few_shot',[False,True])
def test_image_score_smoothing_keeps_segmentation_and_baseline(pooling,few_shot):
    torch.manual_seed(1)
    model=UniADet(TinyEncoder(),lambda_p=1.).eval()
    image=torch.randn(2,3,8,8)
    memory=NormalMemory() if few_shot else None
    if memory is not None:memory.add('a',model.encoder(torch.randn(1,3,8,8)))
    score,amap=model.predict(image,memory,'a',zero_score_pooling=pooling,gaussian_sigma=1.)
    explicit,same=model.predict(image,memory,'a',zero_score_pooling=pooling,gaussian_sigma=1.,image_score_gaussian_sigma=1.)
    raw,raw_map=model.predict(image,memory,'a',zero_score_pooling=pooling,gaussian_sigma=1.,image_score_gaussian_sigma=0.)
    torch.testing.assert_close(score,explicit,rtol=0,atol=0)
    torch.testing.assert_close(amap,same,rtol=0,atol=0)
    torch.testing.assert_close(amap,raw_map,rtol=0,atol=0)
    assert torch.all(raw>=score-1e-7)
    _,local=model(image)
    layer_maps=torch.stack([x.softmax(1)[:,1] for x in local])
    if few_shot:
        expected=(1-model.lambda_f)*layer_maps.mean(0)+model.lambda_f*memory.predict('a',model.encoder(image),(8,8))
        expected=expected.flatten(1).amax(1)
    elif pooling=='layer':expected=layer_maps.flatten(2).amax(2).mean(0)
    else:expected=layer_maps.mean(0).flatten(1).amax(1)
    torch.testing.assert_close(raw,expected)


def test_image_smoothing_checkpoint_defaults_and_validation():
    old=cli.parser().parse_args(['evaluate'])
    cli.restore_evaluation_settings(old,{'config':{'backbone':'clip','gaussian_sigma':4.}})
    assert old.image_score_gaussian_sigma is None and old.gaussian_sigma==4.
    saved=cli.parser().parse_args(['evaluate'])
    cli.restore_evaluation_settings(saved,{'config':{'image_score_gaussian_sigma':0.}})
    assert saved.image_score_gaussian_sigma==0.
    explicit=cli.parser().parse_args(['evaluate','--image-score-gaussian-sigma','2'])
    cli.restore_evaluation_settings(explicit,{'config':{'image_score_gaussian_sigma':0.}})
    assert explicit.image_score_gaussian_sigma==2.
    for value in ['-1','nan','inf']:
        args=cli.parser().parse_args(['train','--image-score-gaussian-sigma',value])
        with pytest.raises(ValueError,match='image_score_gaussian_sigma'):cli.resolve_training_settings(args)


def test_training_refuses_existing_artifacts_before_loading_model(tmp_path):
    checkpoint=tmp_path/'epoch_15.pt';checkpoint.write_bytes(b'preserve')
    args=cli.parser().parse_args(['train','--output',str(tmp_path),'--device','cpu'])
    with patch.object(cli,'build') as build:
        with pytest.raises(ValueError,match='Training output is not empty'):cli.train(args)
        build.assert_not_called()
    assert checkpoint.read_bytes()==b'preserve'


def test_memory_stores_detached_cpu_banks():
    features=TinyEncoder()(torch.randn(2,3,8,8,requires_grad=True))
    memory=NormalMemory();memory.add(('class','view'),features)
    assert all(bank.device.type=='cpu' and not bank.requires_grad for bank in memory.banks[('class','view')])


def test_frozen_backbone_provenance_rejects_drift_but_supports_old_checkpoints(tmp_path):
    from types import SimpleNamespace
    import hashlib
    model=UniADet(TinyEncoder())
    model.encoder.model=SimpleNamespace(config=SimpleNamespace(_commit_hash='current'))
    args=cli.parser().parse_args(['evaluate'])
    assert not cli.verify_backbone_provenance(model,args,{})['backbone_source_verified']
    assert cli.verify_backbone_provenance(model,args,{'backbone_provenance':{'hf_commit':'current'}})['backbone_source_verified']
    with pytest.raises(ValueError,match='Frozen backbone hf_commit differs'):
        cli.verify_backbone_provenance(model,args,{'backbone_provenance':{'hf_commit':'original'}})
    weights=tmp_path/'weights.pt';weights.write_bytes(b'original weights');args.weights=str(weights)
    expected=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert cli.backbone_provenance(model,args)['weights_sha256']==expected
    weights.write_bytes(b'different weights')
    with pytest.raises(ValueError,match='weights_sha256 differs'):
        cli.verify_backbone_provenance(model,args,{'backbone_provenance':{'weights_sha256':expected}})


@pytest.mark.parametrize('descriptor',[None,'none','knn','residual'])
def test_reproduction_checkpoint_requires_baseline_head(descriptor):
    config={'backbone':'clip'}
    if descriptor is not None:
        config.update(patch_features=descriptor,feature_dim=32)
    args=cli.parser().parse_args(['evaluate'])
    if descriptor in (None,'none'):
        cli.restore_evaluation_settings(args,{'config':config})
        assert not hasattr(args,'patch_features')
    else:
        with pytest.raises(ValueError,match='Descriptor checkpoints'):
            cli.restore_evaluation_settings(args,{'config':config})


def test_baseline_head_retains_original_parameters_and_cosine_logits():
    from uniadet.model import DecoupledHead
    import torch.nn.functional as F
    torch.manual_seed(42)
    expected_cls=torch.randn(2,8)*8**-0.5
    expected_seg=torch.randn(2,8)*8**-0.5
    torch.manual_seed(42)
    head=DecoupledHead([8])
    assert set(head.state_dict())=={'cls.0','seg.0'}
    assert sum(p.numel() for p in head.parameters())==32
    head.load_state_dict({'cls.0':expected_cls,'seg.0':expected_seg},strict=True)
    global_token,patches=torch.randn(2,8),torch.randn(2,6,8)
    global_logits,local_logits=head([(global_token,patches,(2,3))],(2,3))
    expected_global=F.linear(F.normalize(global_token,dim=-1),F.normalize(expected_cls,dim=-1))/.07
    expected_local=F.linear(F.normalize(patches,dim=-1),F.normalize(expected_seg,dim=-1))/.07
    torch.testing.assert_close(global_logits[0],expected_global,rtol=0,atol=0)
    torch.testing.assert_close(local_logits[0],expected_local.transpose(1,2).reshape(2,2,2,3),rtol=0,atol=0)
