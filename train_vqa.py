"""Train a visual-prefix adapter on fixed train/validation/test JSONL records."""
import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from glosage.vqa import MorphologyVQA

PROMPT = 'Describe the renal glomerular morphology and identified lesions in this image. Answer:'


class VQADataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['patch_path']) as image:
            image = image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        return (tensor - mean) / std, str(row['group_text'])


def collate(batch, tokenizer, max_length):
    images, answers = zip(*batch)
    prompt_ids = tokenizer([PROMPT] * len(batch), add_special_tokens=True)['input_ids']
    encoded = tokenizer([PROMPT + ' ' + answer for answer in answers], padding=True,
                        truncation=True, max_length=max_length, return_tensors='pt')
    labels = encoded['input_ids'].clone()
    labels[encoded['attention_mask'] == 0] = -100
    for index, ids in enumerate(prompt_ids):
        labels[index, :min(len(ids), labels.shape[1])] = -100
    if not (labels != -100).any(dim=1).all():
        raise ValueError('An answer was fully truncated; increase --max-length')
    return torch.stack(images), encoded['input_ids'], encoded['attention_mask'], labels


def read_rows(path, stain):
    rows = []
    manifest = Path(path).resolve()
    with manifest.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(key in row for key in ('patch_path', 'group_text', 'split')):
                raise ValueError(f'{manifest}:{number}: expected patch_path, group_text, split')
            if stain != 'all' and str(row.get('staining', '')).lower() != stain:
                continue
            if str(row['split']).lower() not in ('train', 'val', 'test'):
                raise ValueError(f'{manifest}:{number}: split must be train, val or test')
            image = Path(row['patch_path'])
            row['patch_path'] = str(image if image.is_absolute() else manifest.parent / image)
            rows.append(row)
    return {split: [r for r in rows if str(r['split']).lower() == split]
            for split in ('train', 'val', 'test')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='JSONL: patch_path, group_text, split, optional staining')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--encoder-weight', required=True)
    parser.add_argument('--visual-representation', choices=('raw', 'projected'), default='raw')
    parser.add_argument('--projection-weight')
    parser.add_argument('--llm', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--staining', choices=('all', 'pas_20x', 'masson', 'pasm'), default='all')
    parser.add_argument('--query-tokens', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--max-length', type=int, default=192)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = read_rows(args.data, args.staining)
    if any(not rows for rows in splits.values()):
        raise ValueError(f'All three fixed splits are required: { {k: len(v) for k, v in splits.items()} }')
    sets = {k: {str(Path(r['patch_path']).resolve()) for r in v} for k, v in splits.items()}
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('An image occurs in multiple splits')
    tokenizer = AutoTokenizer.from_pretrained(args.llm, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('The tokenizer must provide an EOS or pad token')
    tokenizer.padding_side = 'right'
    collator = partial(collate, tokenizer=tokenizer, max_length=args.max_length)
    loaders = {key: DataLoader(VQADataset(rows), batch_size=args.batch_size, shuffle=key == 'train',
                               num_workers=args.num_workers, collate_fn=collator)
               for key, rows in splits.items()}
    model = MorphologyVQA(args.encoder_weight, args.llm, args.query_tokens,
                          args.visual_representation, args.projection_weight,
                          args.local_files_only).to(args.device)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = {'epoch': epoch}
        for split in ('train', 'val'):
            model.train(split == 'train')
            total = 0.0
            with torch.set_grad_enabled(split == 'train'):
                for images, ids, mask, labels in tqdm(loaders[split], desc=f'{split} {epoch}'):
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                    loss = model(images.to(args.device), ids.to(args.device),
                                 mask.to(args.device), labels.to(args.device)).loss
                    if split == 'train':
                        loss.backward()
                        optimizer.step()
                    total += loss.item() * len(images)
            metrics[split + '_loss'] = total / len(splits[split])
        history.append(metrics)
        checkpoint = {'adapter': model.adapter.state_dict(), 'args': vars(args), 'epoch': epoch}
        torch.save(checkpoint, output / 'last.pt')
        if metrics['val_loss'] < best:
            best = metrics['val_loss']
            torch.save(checkpoint, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(metrics, flush=True)
    print(f'Best validation loss: {best:.6f}; adapter saved to {output / "best.pt"}')
    print('Test split was reserved and not used for model selection.')


if __name__ == '__main__':
    main()
