"""DatasetGen manifest adapter for single-frame packed RAW."""
import csv
import os
import platform
from pathlib import PureWindowsPath

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve(root, value):
    if platform.system() == 'Linux' and PureWindowsPath(str(value)).drive:
        raise ValueError('Windows path cannot be used on Linux: %s' % value)
    value = os.path.expanduser(str(value)).replace('\\', os.sep).replace('/', os.sep)
    return os.path.normpath(value if os.path.isabs(value) else os.path.join(root, value))


class RawManifestDataset(Dataset):
    def __init__(self, root, manifest, split, patch_size=0, subsets='all'):
        self.root = _resolve(os.getcwd(), root)
        path = _resolve(self.root, manifest)
        with open(path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        wanted = None if subsets == 'all' else set(subsets.split(','))
        self.rows = [r for r in rows if r.get('split') == split and
                     (wanted is None or r.get('subset') in wanted)]
        if not self.rows:
            raise ValueError(f'No manifest rows for split={split}, subsets={subsets}')
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        inp_path, gt_path = _resolve(self.root, row['input_path']), _resolve(self.root, row['gt_path'])
        inp = np.load(inp_path, allow_pickle=False)
        gt = np.load(gt_path, allow_pickle=False)
        if inp.ndim != 3 or gt.ndim != 3:
            raise ValueError(f'Expected HWC packed RAW: {inp_path}')
        if inp.shape[-1] == 4:
            inp, gt = inp.transpose(2, 0, 1), gt.transpose(2, 0, 1)
        if inp.shape[0] != 4 or gt.shape[0] != 4 or inp.shape != gt.shape:
            raise ValueError(f'Expected matching four-plane RAW pair, got {inp.shape} and {gt.shape}')
        inp = torch.from_numpy(np.ascontiguousarray(inp, dtype=np.float32)) / 1023.0
        gt = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32))
        if self.patch_size:
            h, w = inp.shape[-2:]
            if h < self.patch_size or w < self.patch_size:
                ph, pw = max(0, self.patch_size-h), max(0, self.patch_size-w)
                inp, gt = (torch.nn.functional.pad(x, (0,pw,0,ph), mode='replicate') for x in (inp,gt))
                h, w = inp.shape[-2:]
            y = torch.randint(h-self.patch_size+1, ()).item()
            x = torch.randint(w-self.patch_size+1, ()).item()
            inp = inp[:, y:y+self.patch_size, x:x+self.patch_size]
            gt = gt[:, y:y+self.patch_size, x:x+self.patch_size]
        sample_dir = os.path.dirname(gt_path)
        meta = np.load(os.path.join(sample_dir, 'metadata.npy'), allow_pickle=True).item()
        metadata = {k: torch.as_tensor(np.asarray(meta[k], dtype=np.float32)) for k in ('red_gain','blue_gain','cam2rgb')}
        return {'input': inp, 'target': gt, 'metadata': metadata,
                'sample_id': row.get('sample_id') or os.path.basename(sample_dir),
                'subset': row.get('subset', 'unknown')}
