import glob
from pathlib import Path
import argparse

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class TestEyeDataset(Dataset):

    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, transform=None):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.transform = transform
        self._image_files = []
        data_folder = Path(data_folder)
        for img_path in sorted(glob.glob(f"{data_folder}/*.png")):
            self._image_files.append(str(img_path))

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self._image_files[idx]

        image = self.read_image(image_path)

        sample = {'image': image}

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self._image_files)


parser = argparse.ArgumentParser()
parser.add_argument('test_dataset_path', type=Path)
parser.add_argument('res_dir_path', type=Path)
args = parser.parse_args()

# load models
checkpoints_path = Path('checkpoints')
models = []
for fold_idx in tqdm(range(5)):
    model = smp.Unet(
        'vgg13',
        encoder_weights=None,
        decoder_use_batchnorm=False,
        activation='logsoftmax',
        classes=2,
    ).cuda().eval()
    model.load_state_dict(torch.load(
        checkpoints_path / (str(fold_idx) + '.pth')
    ))
    models.append(model)

# create test dataset
size = 1624
pad_shape = (1248, 1632)
eval_list = [
    A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
    A.PadIfNeeded(*pad_shape),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255,
    ),
    ToTensorV2(transpose_mask=True),
]
dataset = TestEyeDataset(
    args.test_dataset_path,
    transform=A.Compose(eval_list),
)

# predict
for i, sample in enumerate(tqdm(dataset)):
    img_path = Path(dataset._image_files[i])

    img = sample['image']

    preds = []
    for model in models:
        with torch.no_grad():
            # tta with flips
            prediction = model(img.cuda().unsqueeze(dim=0))
            pred_ask = torch.exp(prediction[0, 1]).cpu().numpy()

            img_f = torch.flip(img, dims=(2,))
            prediction = model(img_f.cuda().unsqueeze(dim=0))
            pred_ask2 = torch.flip(torch.exp(prediction[0]), dims=(2,))
            pred_ask2 = pred_ask2[1].cpu().numpy()

            img_f = torch.flip(img, dims=(1,))
            prediction = model(img_f.cuda().unsqueeze(dim=0))
            pred_ask3 = torch.flip(torch.exp(prediction[0]), dims=(1,))
            pred_ask3 = pred_ask3[1].cpu().numpy()

            img_f = torch.flip(img, dims=(1, 2))
            prediction = model(img_f.cuda().unsqueeze(dim=0))
            pred_ask4 = torch.flip(torch.exp(prediction[0]), dims=(1, 2))
            pred_ask4 = pred_ask4[1].cpu().numpy()

        pred_ask = (pred_ask + pred_ask2 + pred_ask3 + pred_ask4) / 4
        preds.append(pred_ask)

    pred = np.mean(preds, axis=0)

    # undo transform
    diff = [1248 - 1232, 1632 - 1624]
    pred = pred[
        diff[0] // 2: pred.shape[0] - diff[0] // 2,
        diff[1] // 2: pred.shape[1] - diff[1] // 2
    ]

    mask = pred > 0.5

    res = mask.astype(np.uint8) * 255

    Image.fromarray(res).save(args.res_dir_path / img_path.name)
