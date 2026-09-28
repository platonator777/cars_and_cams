"""Готовит crop автомобиля, аугментации и батчи для обучения модели."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from vehicle_reid.protocol import Record


class SquarePad:
    """Дополняет прямоугольный crop до квадрата."""
    def __init__(self, fill=(124, 116, 104)):
        """Сохраняет цвет заполнения для квадратного дополнения."""
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Помещает изображение в центр квадратного полотна."""
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return canvas


def build_transform(size: int, train: bool):
    """Создает преобразования для обучения или инференса."""
    steps = [SquarePad(), transforms.Resize((size, size), InterpolationMode.BICUBIC)]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomApply(
            [transforms.ColorJitter(0.2, 0.2, 0.15, 0.04)], p=0.7),
            transforms.RandomAffine(degrees=4, translate=(0.03, 0.03), scale=(0.94, 1.06))]
    steps += [transforms.ToTensor(), transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    if train:
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.4, 2.5), value="random"))
    return transforms.Compose(steps)


def crop_bbox(image: Image.Image, row: Record, context: float = 0.05) -> Image.Image:
    """Вырезает BBox с контекстом и ограничением по краям кадра."""
    dx, dy = row.w * context, row.h * context
    left = max(0, math.floor(row.x - dx))
    top = max(0, math.floor(row.y - dy))
    right = min(image.width, math.ceil(row.x + row.w + dx))
    bottom = min(image.height, math.ceil(row.y + row.h + dy))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid bbox for {row.image_id}: {(row.x,row.y,row.w,row.h)}")
    return image.crop((left, top, right, bottom))


class VehicleDataset(Dataset):
    """Читает кадры и формирует входные тензоры модели."""
    def __init__(self, rows: list[Record], images_dir: str | Path, transform, label_map: dict[int, int] | None = None):
        """Сохраняет записи, каталог кадров и преобразование изображений."""
        self.rows = rows
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.label_map = label_map or {}

    def __len__(self):
        """Возвращает число записей датасета."""
        return len(self.rows)

    def __getitem__(self, index):
        """Читает кадр и возвращает crop, метку класса и индекс."""
        row = self.rows[index]
        with Image.open(self.images_dir / f"{row.image_id}.jpg") as source:
            image = crop_bbox(source.convert("RGB"), row)
        return self.transform(image), self.label_map.get(row.vehicle_id, -1), index


class CameraPKSampler(Sampler[list[int]]):
    """Формирует батчи с несколькими кадрами каждого автомобиля."""
    def __init__(self, rows: list[Record], p: int = 4, k: int = 4, seed: int = 8514):
        """Группирует кадры по автомобилям и камерам для P×K батчей."""
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.by_id: dict[int, list[int]] = defaultdict(list)
        self.by_cam: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            self.by_id[row.vehicle_id].append(index)
            self.by_cam[row.vehicle_id][row.camera_id].append(index)
        if len(self.by_id) < p:
            raise ValueError("training fold has fewer identities than sampler P")
        self.num_batches = max(1, len(rows) // (p * k))

    def set_epoch(self, epoch: int):
        """Меняет seed выборки при переходе к новой эпохе."""
        self.epoch = epoch

    def __len__(self):
        """Возвращает число батчей за эпоху."""
        return self.num_batches

    def __iter__(self):
        """Выбирает P автомобилей по K кадров, используя разные камеры."""
        rng = random.Random(self.seed + self.epoch)
        identities = sorted(self.by_id)
        for _ in range(self.num_batches):
            batch = []
            for vehicle_id in rng.sample(identities, self.p):
                cameras = self.by_cam[vehicle_id]
                pool = self.by_id[vehicle_id]
                if len(cameras) >= 2:
                    cams = rng.sample(sorted(cameras), min(self.k, len(cameras)))
                    selected = [rng.choice(cameras[cam]) for cam in cams]
                    rest = [i for i in pool if i not in selected]
                    need = self.k - len(selected)
                    selected.extend(rng.sample(rest, need) if len(rest) >= need else rng.choices(pool, k=need))
                else:
                    selected = rng.sample(pool, self.k) if len(pool) >= self.k else rng.choices(pool, k=self.k)
                batch.extend(selected)
            yield batch


def seed_worker(worker_id: int):
    """Синхронизирует random seed рабочего процесса DataLoader."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
