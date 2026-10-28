from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Images(Dataset):
    def __init__(self, paths, labels, rgb=False, augment=False):
        self.paths, self.labels, self.rgb, self.augment = list(paths), list(labels), rgb, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {self.paths[index]}')
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        if self.augment:
            if torch.rand(()) < .5:
                image = cv2.flip(image, 1)
            if torch.rand(()) < .5:
                image = cv2.flip(image, 0)
        mean = np.asarray([.485, .456, .406], dtype=np.float32) * 255
        std = np.asarray([.229, .224, .225], dtype=np.float32) * 255
        image = (image.astype(np.float32) - mean) / std
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), int(self.labels[index])


def gs_files(root):
    root = Path(root)
    labels = {'normal': 0, 'non_gs': 0, 'non-gs': 0, '0': 0,
              'sclerosed': 1, 'gs': 1, '1': 1}
    paths, targets = [], []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            continue
        for ancestor in path.relative_to(root).parts[:-1][::-1]:
            if ancestor.lower() in labels:
                paths.append(path)
                targets.append(labels[ancestor.lower()])
                break
    if not paths or set(targets) != {0, 1}:
        raise ValueError(f'Both normal and sclerosed images are required under {root}')
    return paths, targets
