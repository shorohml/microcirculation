import glob
import json
import os
import pickle
import random
import warnings
from collections import defaultdict
from os import listdir
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import argparse

import albumentations as A
import albumentations.augmentations.geometric.functional as F
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class EyeDataset(Dataset):

    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, transform=None, force_create_mask: bool = False):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.transform = transform
        self.force_create_mask = force_create_mask
        self._image_files = []
        data_folder = Path(data_folder)
        for img_path in sorted(glob.glob(f"{data_folder}/*.png")):
        # for img_path in sorted(glob.glob(f"{data_folder}/*_clahe.png")): 
            label_path = data_folder / (Path(img_path).stem + '.geojson')
            # label_path = data_folder / (Path(img_path).stem.replace('_clahe', '') + '.geojson')
            if not label_path.exists():
                continue
            self._image_files.append(str(img_path))

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def parse_polygon(coordinates, image_size):
        mask = np.zeros(image_size, dtype=np.float32)

        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            points = [np.int32([coordinates[0]])]
            cv2.fillPoly(mask, points, 1)

            for polygon in coordinates[1:]:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 0)
        return mask

    @staticmethod
    def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
        """
        Метод для парсинга фигур из geojson файла
        """
        mask = np.zeros(image_size, dtype=np.bool8)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask |= EyeDataset.parse_polygon(
                    polygon, image_size).astype(np.bool8)
        else:
            mask |= EyeDataset.parse_polygon(
                coordinates, image_size).astype(np.bool8)
        return mask.astype(np.float32)

    def read_layout(self, path: str, image_size: tuple) -> np.ndarray:
        """
        Метод для чтения geojson разметки и перевода в numpy маску
        """
        path = Path(path)
        mask_path = path.parent / path.name.replace('geojson', 'npz')
        if mask_path.exists() and not self.force_create_mask:
            mask = np.load(mask_path)['arr_0']
            mask = np.stack([1 - mask, mask], axis=-1).astype(np.float32)
            return mask

        # some files contain cyrillic letters, thus cp1251
        with open(path, 'r', encoding='cp1251') as f:
            json_contents = json.load(f)

        num_channels = 1 + max(self.class_ids.values())
        mask_channels = [np.zeros(image_size, dtype=np.float32)
                         for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.float32)

        if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            channel_id = self.class_ids["vessel"]
            mask = self.parse_mask(shape['geometry'], image_size)
            mask_channels[channel_id] = np.maximum(
                mask_channels[channel_id], mask)

        mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)

        return np.stack(mask_channels, axis=-1)

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self._image_files[idx]

        # Получаем соответствующий файл разметки
        json_path = image_path.replace("png", "geojson")
        # json_path = image_path.replace("_clahe.png", ".geojson")

        image = self.read_image(image_path)

        mask = self.read_layout(json_path, image.shape[:2])

        sample = {'image': image,
                  'mask': mask}

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self._image_files)

    # Метод для проверки состояния датасета
    def make_report(self):
        reports = []
        if (not self.data_folder):
            reports.append("Путь к датасету не указан")
        if (len(self._image_files) == 0):
            reports.append("Изображения для распознавания не найдены")
        else:
            reports.append(f"Найдено {len(self._image_files)} изображений")
        cnt_images_without_masks = sum(
            [1 - len(glob.glob(filepath.replace("png", "geojson"))) for filepath in self._image_files])
        # cnt_images_without_masks = sum(
        #     [1 - len(glob.glob(filepath.replace("_clahe.png", ".geojson"))) for filepath in self._image_files])
        if cnt_images_without_masks > 0:
            reports.append(
                f"Найдено {cnt_images_without_masks} изображений без разметки")
        else:
            reports.append(f"Для всех изображений есть файл разметки")
        return reports


class DatasetPart(Dataset):
    """
    Обертка над классом датасета для его разбиения на части
    """

    def __init__(self, dataset: Dataset,
                 indices: np.ndarray,
                 transform: A.Compose = None):
        self.dataset = dataset
        self.indices = indices

        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self) -> int:
        return len(self.indices)


class MyShiftScaleRotate(A.ShiftScaleRotate):

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        img = img.astype(np.float32)
        img[..., 1] = F.shift_scale_rotate(
            img[..., 1], angle, scale, dx, dy, cv2.INTER_CUBIC, self.border_mode, self.mask_value)
        img[..., 1] = (img[..., 1] > 0.5).astype(np.float32)
        img[..., 0] = 1 - img[..., 1]
        return img


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(False)
            module.eval()
    return model


class UnetTrainer:
    """
    Класс, реализующий обучение модели
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 device: str, metric_functions: List[Tuple[str, Callable]] = [],
                 epoch_number: int = 0,
                 lr_scheduler: Optional[Any] = None,
                 checkpoints_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.metric_functions = metric_functions

        self.epoch_number = epoch_number

        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def evaluate_batch(self, val_iterator: Iterator, eval_on_n_batches: int) -> Optional[Dict[str, float]]:
        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(eval_on_n_batches):
            try:
                batch = next(val_iterator)

                xs = batch['image'].to(self.device)
                ys_true = batch['mask'].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break
            ys_pred = self.model.eval()(xs)
            loss = self.criterion(ys_pred, ys_true)

            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader, eval_on_n_batches: int = 1) -> Dict[str, float]:
        """
        Вычисление метрик для эпохи
        """
        metrics_sum = defaultdict(float)
        num_batches = 0

        val_iterator = iter(val_loader)

        while True:
            batch_metrics = self.evaluate_batch(
                val_iterator, eval_on_n_batches)

            if batch_metrics is None:
                break

            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]

            num_batches += 1

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit_batch(self, train_iterator: Iterator, update_every_n_batches: int) -> Optional[Dict[str, float]]:
        """
        Тренировка модели на одном батче
        """

        self.optimizer.zero_grad()

        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(update_every_n_batches):
            try:
                batch = next(train_iterator)

                xs = batch['image'].to(self.device)
                ys_true = batch['mask'].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break

            self.model.train()
            # self.model = freeze_bn(self.model)

            ys_pred = self.model(xs)
            loss = self.criterion(ys_pred, ys_true)

            (loss / update_every_n_batches).backward()

            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        self.optimizer.step()

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        return metrics

    def fit_epoch(self, train_loader, update_every_n_batches: int = 1) -> Dict[str, float]:
        """
        Одна эпоха тренировки модели
        """

        metrics_sum = defaultdict(float)
        num_batches = 0

        train_iterator = iter(train_loader)

        pbar = tqdm(range(len(train_loader)))

        while True:
            batch_metrics = self.fit_batch(
                train_iterator, update_every_n_batches)

            if batch_metrics is None:
                break

            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]

            num_batches += 1
            pbar.update()

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit(self, train_loader, num_epochs: int,
            val_loader=None, update_every_n_batches: int = 1,
            ) -> Dict[str, np.ndarray]:
        """
        Метод, тренирующий модель и вычисляющий метрики для каждой эпохи
        """

        summary = defaultdict(list)

        def save_metrics(metrics: Dict[str, float], postfix: str = '') -> None:
          # Сохранение метрик в summary
            nonlocal summary, self

            for metric in metrics:
                metric_name, metric_value = f'{metric}{postfix}', metrics[metric]

                summary[metric_name].append(metric_value)

        for epoch_idx in range(num_epochs - self.epoch_number):
            self.epoch_number += 1

            train_metrics = self.fit_epoch(
                train_loader, update_every_n_batches)

            with torch.no_grad():
                save_metrics(train_metrics, postfix='_train')

                if val_loader is not None:
                    test_metrics = self.evaluate(val_loader)
                    save_metrics(test_metrics, postfix='_test')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            out_path = self.checkpoints_dir / \
                (str(epoch_idx).zfill(5) + '.pth')
            torch.save(self.model.state_dict(), out_path)

        summary = {metric: np.array(summary[metric]) for metric in summary}

        return summary


# F1-мера
class SoftDice:

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, predictions: List[Dict[str, torch.Tensor]],
                 targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        numerator = torch.sum(2 * predictions * targets)
        denominator = torch.sum(predictions + targets)
        return numerator / (denominator + self.epsilon)


# Метрика полноты
class Recall:

    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b

    def __call__(self, predictions: List[Dict[str, torch.Tensor]],
                 targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(targets)
        return numerator / (denominator + self.epsilon)


# Метрика точности
class Accuracy:

    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b

    def __call__(self, predictions: list, targets: list) -> torch.Tensor:
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(predictions)
        return numerator / (denominator + self.epsilon)


class F1:

    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b
        self.acc = Accuracy(epsilon, b)
        self.recall = Recall(epsilon, b)

    def __call__(self, predictions: list, targets: list) -> torch.Tensor:
        acc = self.acc(predictions, targets)
        recall = self.recall(predictions, targets)
        return 2 * acc * recall / (acc + recall)


def make_metrics():
    soft_dice = SoftDice()
    recall = Recall()
    acc = Accuracy()

    def exp_dice(pred, target):
        return soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])

    def accuracy(pred, target):
        return acc(torch.exp(pred[:, 1:]), target[:, 1:])

    def exp_recall(pred, target):
        return recall(torch.exp(pred[:, 1:]), target[:, 1:])

    def f_measure(pred, target):
        exp = torch.exp(pred[:, 1:])
        a = acc(exp, target[:, 1:])
        r = recall(exp, target[:, 1:])
        return 2 * a * r / (a + r)

    return [('exp_dice', exp_dice),
            ('accuracy', accuracy),
            ('recall', exp_recall),
            ('f_measure', f_measure),
            ]


def bn2instance(module):
    module_output = module

    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.InstanceNorm2d(module.num_features)

    for name, child in module.named_children():
        module_output.add_module(name, bn2instance(child))

    return module_output


seed_everything(42)


parser = argparse.ArgumentParser()
parser.add_argument('train_dataset_path', type=Path)
parser.add_argument('checkpoints_path', type=Path)
args = parser.parse_args()


size = 1624
pad_shape = (1248, 1632)
train_list = [
    A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
    A.PadIfNeeded(*pad_shape),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    MyShiftScaleRotate(
        shift_limit=0.2,
        scale_limit=0.00,
        rotate_limit=15,
        interpolation=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=0.5,
    ),
    A.ISONoise(
        p=0.5
    ),
    A.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.1,
        p=0.2,
    ),
    A.RandomGamma(
        gamma_limit=(90, 110),
        p=0.2,
    ),
    A.MotionBlur(
        blur_limit=5,
        p=0.2,
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255,
    ),
    ToTensorV2(transpose_mask=True),
]
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
transforms = {'train': A.Compose(train_list), 'test': A.Compose(eval_list)}

# Поправим geojson и удалим случай с пустой разметкой
data_folder = args.train_dataset_path
to_remove = []
for json_path in tqdm(sorted(glob.glob(f'{data_folder}/*.geojson'))):
    with open(json_path, 'rt') as in_f:
        data = json.load(in_f)
    if not isinstance(data, dict):
        print(json_path)
        data = {
            'type': 'FeatureCollection',
            'features': data,
        }
        with open(json_path, 'wt') as out_f:
            json.dump(data, out_f)
    if 'features' in data and not data['features']:
        json_path = Path(json_path)
        to_remove.append(json_path)
        to_remove.append(json_path.parent /
                         json_path.name.replace('geojson', 'png'))
for path in to_remove:
    if path.exists():
        path.unlink()

# Создадим маски
create_mask = True

dataset = EyeDataset(
    data_folder,
    force_create_mask=create_mask
)

if create_mask:
    for i, sample in enumerate(tqdm(dataset)):
        mask = sample['mask'].astype(np.uint8)[..., 1]
        path = dataset._image_files[i]
        path = Path(path)
        mask_path = path.parent / path.name.replace('png', 'npz')
        np.savez_compressed(mask_path, mask)

# Инициализируем датасет
create_mask = False

dataset = EyeDataset(
    data_folder,
    force_create_mask=create_mask
)

# Проверим состояние загруженного датасета
for msg in dataset.make_report():
    print(msg)

print("Обучающей выборки ", len(
    listdir("/home/shoroh/data/mc/train_dataset_mc")) // 2)
print("Тестовой выборки ", len(
    listdir("/home/shoroh/data/mc/test_dataset_mc/eye_test")))

kf = KFold(
    n_splits=5,
    random_state=42,
    shuffle=True,
)
checkpoints_dir = args.checkpoints_path
for fold_idx, (train_index, test_index) in enumerate(kf.split(np.zeros((len(dataset), 1)))):
    train_dataset = DatasetPart(
        dataset, train_index, transform=transforms['train'])
    valid_dataset = DatasetPart(
        dataset, test_index, transform=transforms['test'])

    print('-' * 60 + f' fold {fold_idx} ' + '-' * 60)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=5,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=5,
        shuffle=False,
        drop_last=True,
    )

    torch.cuda.empty_cache()

    # Подргружаем модель и задаём функцию потерь
    model = smp.Unet(
        'vgg13',
        encoder_weights='imagenet',
        decoder_use_batchnorm=False,
        activation='logsoftmax',
        classes=2,
    ).cuda().train()
    model = freeze_bn(model)

    def make_criterion():
        soft_f1 = F1()
        nll = torch.nn.NLLLoss()

        def criterion(pred, target):
            s1 = nll(
                pred,
                torch.argmax(target, dim=1),
            )
            s2 = soft_f1(torch.exp(pred[:, 1:]), target[:, 1:])
            return s1 - s2

        return criterion

    criterion = make_criterion()

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 if epoch < 20 else 0.1
    )

    # Обучаем модель
    checkpoints_fold_dir = checkpoints_dir / f'fold_{fold_idx}'
    checkpoints_fold_dir.mkdir(exist_ok=True, parents=True)
    trainer = UnetTrainer(
        model,
        optimizer,
        criterion,
        'cuda',
        metric_functions=make_metrics(),
        lr_scheduler=scheduler,
        checkpoints_dir=str(checkpoints_fold_dir),
    )
    summary = trainer.fit(
        train_loader,
        1,
        val_loader=valid_loader
    )

    with open(checkpoints_fold_dir / 'summary.p', 'wb') as out_f:
        pickle.dump(summary, out_f)
