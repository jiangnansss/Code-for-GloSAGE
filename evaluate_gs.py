"""Evaluate separately fine-tuned PAS-GS recognition checkpoints on GS_class2."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from glosage.data import Images, gs_files
from glosage.models import build_model, forward_logits, load_classifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', action='append', required=True, metavar='MODEL=PATH')
    p.add_argument('--keep-config', help='KEEP config.json downloaded from Hugging Face')
    p.add_argument('--plip-config', default='vinid/plip', help='PLIP model ID or local configuration directory')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', default='outputs/gs_class2_metrics.csv')
    args = p.parse_args()
    paths, y = gs_files(args.data)
    rows = []
    for entry in args.checkpoint:
        family, path = entry.split('=', 1)
        model = load_classifier(build_model(family, args.keep_config, args.plip_config), path, family).to(args.device).eval()
        loader = DataLoader(Images(paths, y, rgb=family == 'keep'), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False)
        probs = []
        with torch.inference_mode():
            for x, _ in loader:
                probs.extend(forward_logits(model, family, x.to(args.device)).softmax(1)[:, 1].cpu().tolist())
        pred = np.asarray(probs) >= .5
        rows.append({'model': family, 'checkpoint': path, 'n': len(y),
                     'roc_auc': roc_auc_score(y, probs), 'accuracy': accuracy_score(y, pred),
                     'macro_f1': f1_score(y, pred, average='macro')})
        print(rows[-1], flush=True)
        del model
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == '__main__':
    main()
