"""Fine-tune GloSAGE on a binary lesion task using explicit image splits."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from glosage.data import Images
from glosage.models import GloSAGEClassifier, initialize_glosage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='CSV with path,label,split')
    parser.add_argument('--encoder-weight', required=True)
    parser.add_argument('--projection-weight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    df = pd.read_csv(args.manifest)
    if not {'path', 'label', 'split'}.issubset(df.columns):
        raise ValueError('Manifest needs path,label,split columns')
    if df.path.duplicated().any() or not set(df.label.unique()).issubset({0, 1}):
        raise ValueError('Duplicate images or labels outside {0,1}')
    loaders = {}
    for split in ('train', 'val', 'test'):
        rows = df[df.split == split]
        if rows.empty or set(rows.label) != {0, 1}:
            raise ValueError(f'{split} must contain both binary classes')
        ds = Images(rows.path, rows.label, augment=split == 'train')
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=split == 'train',
                                    num_workers=args.workers)
    model = GloSAGEClassifier()
    initialize_glosage(model, args.encoder_weight, args.projection_weight)
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for x, y in loaders['train']:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
        model.eval()
        y_true, y_prob = [], []
        with torch.inference_mode():
            for x, y in loaders['val']:
                y_true.extend(y.tolist())
                y_prob.extend(model(x.to(args.device)).softmax(1)[:, 1].cpu().tolist())
        auc = roc_auc_score(y_true, y_prob)
        print(f'epoch={epoch + 1} val_auc={auc:.4f}', flush=True)
        if auc > best:
            best = auc
            torch.save({'modelT': model.state_dict(), 'epoch': epoch + 1,
                        'valAUC': auc, 'args': vars(args)}, output)
    print(f'Best validation AUC: {best:.4f}; checkpoint: {output}')


if __name__ == '__main__':
    main()
