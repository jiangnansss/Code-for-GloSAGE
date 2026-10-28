from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MODEL_SETTINGS = {}


class KEEPConfig:
    """Configuration matching KEEP's released quick-start model."""
    def __init__(self, vision_config=None, text_config=None, projection_dim=768, **kwargs):
        self.vision_config = vision_config or {}
        self.text_config = text_config or {}
        self.projection_dim = int(projection_dim)


class KEEPModel(nn.Module):
    """KEEP image/text encoder from MAGIC-AI4Med/KEEP's local inference API."""
    def __init__(self, config: KEEPConfig):
        super().__init__()
        import timm
        from transformers import BertConfig, BertModel

        # KEEP was released with a LayerScale implementation whose parameter is
        # named ``weight``; timm 1.x calls the same parameter ``gamma``.
        class RenameLayerScale(nn.Module):
            def __init__(self, dim, init_values=1e-5, inplace=False):
                super().__init__()
                self.inplace = inplace
                self.weight = nn.Parameter(init_values * torch.ones(dim))

            def forward(self, x):
                return x.mul_(self.weight) if self.inplace else x * self.weight

        timm.models.vision_transformer.LayerScale = RenameLayerScale

        vision_config = config.vision_config
        self.visual = timm.create_model(
            "vit_large_patch16_224", pretrained=False, img_size=224, patch_size=16,
            init_values=vision_config.get("init_values", 1e-5),
            num_classes=vision_config.get("num_classes", 0),
            dynamic_img_size=vision_config.get("dynamic_img_size", False),
        )
        self.visual_head = nn.Sequential(
            nn.Linear(self.visual.num_features, config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim),
        )
        self.text = BertModel(BertConfig(**config.text_config))
        # Present in the released state dict even though evaluation uses cosine similarity.
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.04))

    def encode_image(self, image_inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual_head(self.visual(image_inputs)), dim=-1)

    def encode_text(self, text_inputs) -> torch.Tensor:
        return F.normalize(self.text(**text_inputs).pooler_output, dim=-1)


def _load_config(model_dir: Path) -> KEEPConfig:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        return KEEPConfig(**json.load(handle))


def _strip_prefixes(state):
    output = {}
    for key, value in state.items():
        key = str(key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        output[key] = value
    return output


def load_keep(model_dir: str, device: str) -> Tuple[KEEPModel, object, object]:
    model_dir = Path(model_dir)
    required = (model_dir / "config.json", model_dir / "pytorch_model.bin")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "KEEP local release is incomplete. Missing: " + ", ".join(missing) +
            ". Download the KEEP release and pass --keep-model /path/to/KEEP_release."
        )
    from transformers import AutoTokenizer

    model = KEEPModel(_load_config(model_dir))
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(_strip_prefixes(state), strict=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True, local_files_only=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"KEEP tokenizer files are missing from {model_dir}. "
            "Keep tokenizer_config.json, vocab.txt (or tokenizer.json), and special_tokens_map.json "
            "beside config.json and pytorch_model.bin."
        ) from exc
    image_transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model.to(device).eval(), tokenizer, image_transform


@torch.no_grad()
def encode_keep(records: Sequence[dict], texts: Sequence[str], device: str,
                batch_size: int, model_dir: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    settings = MODEL_SETTINGS["keep-image"]
    model, tokenizer, image_transform = load_keep(model_dir or settings["model_dir"], device)
    image_features = []
    for start in range(0, len(records), batch_size):
        images = []
        for row in records[start:start + batch_size]:
            with Image.open(row["patch_path"]) as image:
                images.append(image_transform(image.convert("RGB")))
        image_features.append(model.encode_image(torch.stack(images).to(device)).cpu())
    text_features = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(list(texts[start:start + batch_size]), max_length=256, padding="max_length",
                           truncation=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        text_features.append(model.encode_text(tokens).cpu())
    return F.normalize(torch.cat(image_features), dim=-1), F.normalize(torch.cat(text_features), dim=-1)
