import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('src_path', type=Path)
parser.add_argument('dst_path', type=Path)
args = parser.parse_args()

# select and load models
models = []
for fold_idx in tqdm(range(5)):
    fold_dir = args.src_path / f'fold_{fold_idx}'

    with open(fold_dir / 'summary.p', 'rb') as in_f:
        summary = pickle.load(in_f)

    idx = 30 + summary['f_measure_test'][30:].argmax()

    shutil.copy(
        fold_dir / (str(idx).zfill(5) + '.pth'),
        args.dst_path / (str(fold_idx) + '.pth'),
    )
