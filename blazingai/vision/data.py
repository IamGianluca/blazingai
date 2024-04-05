import os
import re
from pathlib import Path
from typing import Any, Optional

import lightning as pl
import numpy as np
import torch
from PIL import Image, ImageFile
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

# sometimes, you will have images without an ending bit; this
# takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        img_paths: list[Path],
        aug: Compose,
        trgt: Optional[list] = None,
    ) -> None:
        self.img_paths = img_paths
        self.trgt = trgt
        self.aug = aug
        self.length = len(img_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.aug(img)

        if self.trgt is not None:  # train/val dataset
            return img, torch.tensor(self.trgt[idx])
        else:  # test dataset
            return img


class Image3DClassificationDataset(Dataset):
    def __init__(
        self,
        img_paths: list[Path],
        aug: Compose,
        trgt: Optional[list] = None,
    ) -> None:
        self.img_paths = img_paths
        self.trgt = trgt
        self.aug = aug  # TODO: fix, not used yet
        self.length = len(img_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        # for 3D images we load each individual frame as a numpy array,
        # then use Spline Interpolated Zoom (SIZ) to standardize the output
        # volume's shape
        fpaths = sorted(
            list(Path(self.img_paths[idx]).iterdir()),
            key=lambda x: int(re.findall(r"\d+", x.name)[0]),
        )
        # TODO: make a transform for this
        img_list = [np.array(Image.open(fpath)) for fpath in fpaths]
        img = np.stack(img_list, axis=0)
        img = spline_interpolated_zoom(img, desired_depth=15)
        # TODO: use timm's ToTensor
        img = torch.tensor(img).float()

        # if self.augmentations:
        #     image = self.augmentations(image=image)["image"]
        # if image.ndim == 2:  # add channel axis to grayscale images
        #     image = image[None, ...]

        if self.trgt is not None:  # train/val dataset
            return img, torch.tensor(self.trgt[idx])
        else:  # test dataset
            return img


def spline_interpolated_zoom(img, desired_depth: int = 3):
    """Spline Interpolated Zoom
    ref: https://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynb
    """
    current_depth = img.shape[0]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (depth_factor, 1, 1), mode="nearest")
    return img_new


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        img_paths: list[Path],
        aug: Compose,
        trgt: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.img_paths = img_paths
        self.trgt = trgt
        self.aug = aug
        self.length = len(img_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))

        if self.trgt:
            # TODO: move this 'multiplication to utils.get_targets()
            boxes = self.trgt[idx]["boxes"]
            labels = [self.trgt[idx]["labels"]] * boxes.shape[0]
        else:
            boxes, labels = None, None

        # TODO: we should always apply the augmentations, they are not optional.
        # TODO: use timm for data augmentations
        if self.aug:
            transformed = self.aug(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]

        # after applying data augmentation, boxes for only background
        # cases should be brought back to [[0, 0, 1, 1]]
        if labels == [0]:
            boxes = [[0, 0, 1, 1]]

        img = torch.tensor(img).float().permute(2, 0, 1) / 255.0

        if self.trgt:  # train/val dataset
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            return img, target
        else:  # test dataset
            return img


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str,
        bs: int,
        trn_img_paths: Optional[list[Path]] = None,
        val_img_paths: Optional[list[Path]] = None,
        tst_img_paths: Optional[list[Path]] = None,
        trn_trgt: Optional[list] = None,
        val_trgt: Optional[list] = None,
        trn_aug: Optional[Compose] = None,
        val_aug: Optional[Compose] = None,
        tst_aug: Optional[Compose] = None,
    ):
        super().__init__()

        self.task = task
        self.trn_img_paths = trn_img_paths
        self.val_img_paths = val_img_paths
        self.tst_img_paths = tst_img_paths

        self.trn_trgt = trn_trgt
        self.val_trgt = val_trgt

        self.trn_aug = trn_aug
        self.val_aug = val_aug
        self.tst_aug = tst_aug

        self.bs = bs

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.trn_img_paths is None:
                raise ValueError("Missing trn_img_path")
            if self.val_img_paths is None:
                raise ValueError("Missing val_img_path")

            self.trn_ds = get_dataset[self.task](
                img_paths=self.trn_img_paths,
                trgt=self.trn_trgt,
                aug=self.trn_aug,
            )
            self.val_ds = get_dataset[self.task](
                img_paths=self.val_img_paths,
                trgt=self.val_trgt,
                aug=self.val_aug,
            )
        elif stage == "predict":
            if self.tst_img_paths is None:
                raise ValueError("Missing tst_img_paths")
            self.test_ds = get_dataset[self.task](
                img_paths=self.tst_img_paths,
                aug=self.tst_aug,
            )
        else:
            raise ValueError(f"stage `{stage}` currently not supported")

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn if self.task == "detection" else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            drop_last=False,
            collate_fn=self._collate_fn if self.task == "detection" else None,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            drop_last=False,
        )

    def _collate_fn(self, batch):
        return tuple(zip(*batch))


get_dataset = {
    "classification": ImageClassificationDataset,
    "3Dclassification": Image3DClassificationDataset,
    "detection": ObjectDetectionDataset,
}
