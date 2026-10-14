import os
import json
import math
import random
import torch
from torch.utils.data import Dataset
from utils import AddGaussianNoise
from torchvision import transforms
from PIL import Image
from data_preparation import validate_record
from .constants import CLASS_NAMES, DATA_PATH, DOMAINS


def read_metadata(data_path, meta_path):
    """Read and validate a JSONL manifest once, before model training/evaluation."""
    rows = []
    with open(meta_path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL row must be an object")
                if not isinstance(row.get("class_name"), str) or not row["class_name"].strip():
                    raise ValueError("class_name must be a nonempty product name, e.g. wafer")
                row["class_name"] = row["class_name"].strip()
                if not isinstance(row.get("image_path"), str) or not row["image_path"].strip():
                    raise ValueError("image_path must be a nonempty path")
                if "label" not in row:
                    raise ValueError("label is required: 0 (normal) or 1 (defect)")
                validate_record(data_path, row)
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise ValueError(f"Invalid metadata at {meta_path}:{line_number}: {error}") from error
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty dataset: {meta_path}")
    return rows


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
        metadata=None,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.text = text
        self.meta = read_metadata(data_path, meta_path) if metadata is None else metadata
        self.full_shot = "full-shot" in str(meta_path)

        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = []
        # transform_x.append(AddGaussianNoise(std=1, p=0.7))
        if not text:
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
            )
        self.transform_x = transforms.Compose(
            transform_x
            + [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        random_transform = transforms.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask], dim=0)
        assert transform_tensor.shape[0] == 4
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        class_name: str,
        logger=None,
        metadata=None,
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        rows = read_metadata(data_path, meta_path) if metadata is None else metadata
        self.meta = [row for row in rows if row["class_name"] == class_name]

        # Define transforms
        self.transform_x = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

        # logging
        if logger:
            logger.info(f"Class name: {class_name}")
            logger.info(f"Sample number: {len(self.meta)}")
            logger.info("=====================================")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(self.data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


def get_dataset(
    dataset_name: str,
    img_size: int,
    training_mode: str,
    shot: int = -1,
    stage: str = "train",
    logger=None,
    data_path=None,
    meta_path=None,
):
    if meta_path is not None:
        if data_path is None:
            raise ValueError("--data_root is required with --meta_path")
        rows = read_metadata(data_path, meta_path)
        class_names = sorted({row["class_name"] for row in rows})
        CLASS_NAMES[dataset_name] = class_names
        DATA_PATH[dataset_name] = data_path
        DOMAINS.setdefault(dataset_name, "Industrial")
        if stage == "train":
            return (BaseDataset(data_path, meta_path, img_size, text=True, metadata=rows),
                    BaseDataset(data_path, meta_path, img_size, text=False, metadata=rows))
        return {name: BaseSingleClassDataset(data_path, meta_path, img_size, name, logger, metadata=rows)
                for name in class_names}
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )

    if stage == "train":
        if training_mode == "few_shot":
            assert shot > 0, "shot should be positive"
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, f"{shot}-shot.jsonl"
            )
        else:
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, "full-shot.jsonl"
            )

        data_path = DATA_PATH[dataset_name.split("-")[0]]
        rows = read_metadata(data_path, meta_path)
        text_dataset = BaseDataset(data_path, meta_path, img_size, text=True, metadata=rows)
        image_dataset = BaseDataset(data_path, meta_path, img_size, text=False, metadata=rows)
        return text_dataset, image_dataset
    elif stage == "test":
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        class_names = CLASS_NAMES[dataset_name]
        rows = read_metadata(DATA_PATH[dataset_name], meta_path)
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=logger,
                metadata=rows,
            )
            datasets[class_name] = image_dataset
        return datasets
    elif stage == "visualize":
        class_names = CLASS_NAMES[dataset_name]
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        rows = read_metadata(DATA_PATH[dataset_name], meta_path)
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=None,
                metadata=rows,
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")
