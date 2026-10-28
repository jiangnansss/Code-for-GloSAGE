"""Visual-prefix VQA adapted from the original vqa/model.py."""
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from .models import GloSAGEClassifier, initialize_encoder, initialize_glosage


class MorphologyVQA(nn.Module):
    def __init__(self, encoder_weight, llm_name, query_tokens=8,
                 visual_representation='raw', projection_weight=None, local_files_only=False):
        super().__init__()
        if query_tokens < 1:
            raise ValueError('query_tokens must be positive')
        visual = GloSAGEClassifier()
        if visual_representation == 'raw':
            initialize_encoder(visual, encoder_weight)
            feature_dim = 768
        elif visual_representation == 'projected':
            if not projection_weight:
                raise ValueError('projected mode requires --projection-weight')
            initialize_glosage(visual, encoder_weight, projection_weight)
            feature_dim = 1024
        else:
            raise ValueError(f'Unsupported visual representation: {visual_representation}')
        self.encoder = visual.encoder
        self.projection = visual.projection if visual_representation == 'projected' else None
        self.encoder.requires_grad_(False)
        if self.projection is not None:
            self.projection.requires_grad_(False)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, local_files_only=local_files_only)
        self.llm.requires_grad_(False)
        hidden = getattr(self.llm.config, 'hidden_size', None)
        if hidden is None:
            hidden = self.llm.config.n_embd
        self.query_tokens = query_tokens
        self.adapter = nn.Sequential(nn.Linear(feature_dim, hidden * query_tokens), nn.GELU(),
                                     nn.LayerNorm(hidden * query_tokens))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.llm.eval()
        if self.projection is not None:
            self.projection.eval()
        return self

    def visual_prefix(self, images):
        with torch.no_grad():
            features = self.encoder(images, return_before_head=True)
            if self.projection is not None:
                features = self.projection.net[1](self.projection.net[0](features))
        return self.adapter(features).reshape(features.shape[0], self.query_tokens, -1)

    def forward(self, images, input_ids, attention_mask, labels):
        prefix = self.visual_prefix(images)
        words = self.llm.get_input_embeddings()(input_ids)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        ignored = torch.full(prefix.shape[:2], -100, dtype=labels.dtype, device=labels.device)
        return self.llm(inputs_embeds=torch.cat((prefix, words), dim=1),
                        attention_mask=torch.cat((prefix_mask, attention_mask), dim=1),
                        labels=torch.cat((ignored, labels), dim=1))

    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask, **kwargs):
        prefix = self.visual_prefix(images)
        words = self.llm.get_input_embeddings()(input_ids)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        return self.llm.generate(inputs_embeds=torch.cat((prefix, words), dim=1),
                                 attention_mask=torch.cat((prefix_mask, attention_mask), dim=1), **kwargs)
