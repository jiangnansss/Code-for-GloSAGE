"""Model architectures and strict checkpoint loading for lesion recognition."""
import re
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .glopath import vit_base


class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 512))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class GloSAGEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # The original recognition checkpoints include an unused two-class encoder head.
        self.encoder = vit_base(num_classes=2)
        self.projection = ProjectionHead()
        self.classifier = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.encoder(x, return_before_head=True)
        return self.classifier(self.projection.net[1](self.projection.net[0](x)))


def read_state(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'Expected state dictionary: {path}')
    for key in ('modelT', 'student', 'state_dict', 'model'):
        if key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    return {re.sub(r'^module\.', '', str(k)): v for k, v in payload.items()}


def initialize_encoder(model, encoder_path):
    state = read_state(encoder_path)
    expected = model.encoder.state_dict()
    mapped = {}
    for key, value in state.items():
        key = re.sub(r'^(backbone|encoder)\.', '', key)
        if key in expected and expected[key].shape == value.shape:
            mapped[key] = value
    essential = [k for k in expected if k.startswith(('blocks.', 'patch_embed.', 'norm.'))]
    missing = [k for k in essential if k not in mapped]
    if missing:
        raise ValueError(f'Incomplete GloPath encoder checkpoint; missing {missing[:8]}')
    model.encoder.load_state_dict(mapped, strict=False)


def initialize_glosage(model, encoder_path, projection_path):
    initialize_encoder(model, encoder_path)
    payload = torch.load(projection_path, map_location='cpu', weights_only=False)
    state = payload.get('v_head', payload.get('state_dict', payload))
    state = {re.sub(r'^(module\.|projection\.|v_head\.)+', '', k): v for k, v in state.items()}
    first = {k: state[k] for k in ('net.0.weight', 'net.0.bias') if k in state}
    if len(first) != 2:
        raise ValueError('Visual projection checkpoint must contain net.0.weight and net.0.bias')
    model.projection.load_state_dict(first, strict=False)


def build_model(family, keep_config=None, plip_config='vinid/plip'):
    if family == 'glosage':
        return GloSAGEClassifier()
    if family in ('glopath', 'conch'):
        return vit_base(num_classes=2)
    if family == 'uni':
        import timm
        return timm.create_model('vit_large_patch16_224', img_size=224, patch_size=16,
                                 init_values=1e-5, num_classes=2, dynamic_img_size=True)
    if family == 'plip':
        from transformers import CLIPVisionConfig, CLIPVisionModel
        class PLIP(nn.Module):
            def __init__(self):
                super().__init__()
                config = CLIPVisionConfig.from_pretrained(plip_config)
                self.model = CLIPVisionModel(config).vision_model
                self.fc = nn.Linear(config.hidden_size, 2)
            def forward(self, x):
                return self.fc(self.model(x).pooler_output)
        return PLIP()
    if family == 'keep':
        import json
        from .keep import KEEPConfig, KEEPModel
        if not keep_config:
            raise ValueError('KEEP requires --keep-config pointing to the downloaded KEEP config.json')
        original = __import__('timm').models.vision_transformer.LayerScale
        try:
            backbone = KEEPModel(KEEPConfig(**json.loads(Path(keep_config).read_text())))
        finally:
            __import__('timm').models.vision_transformer.LayerScale = original
        class KEEPClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual, self.visual_head = backbone.visual, backbone.visual_head
                self.classifier = nn.Linear(self.visual_head[-1].out_features, 2)
            def forward(self, x):
                return self.classifier(F.normalize(self.visual_head(self.visual(x)), dim=-1))
        return KEEPClassifier()
    raise ValueError(f'Unknown model: {family}')


def load_classifier(model, path, family):
    state = read_state(path)
    expected = model.state_dict()
    aligned = {}
    for key, value in state.items():
        if family == 'plip' and key.startswith('head.') and 'fc.weight' not in state:
            key = 'fc.' + key[5:]
        if key not in expected and key.endswith('.gamma') and key[:-6] + '.weight' in expected:
            key = key[:-6] + '.weight'
        elif key not in expected and key.endswith('.weight') and key[:-7] + '.gamma' in expected:
            key = key[:-7] + '.gamma'
        aligned[key] = value
    model.load_state_dict(aligned, strict=True)
    return model


def forward_logits(model, family, x):
    if family in ('glopath', 'conch'):
        return model(x, return_before_head=False)
    return model(x)
