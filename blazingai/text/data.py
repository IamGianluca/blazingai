import os

import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, text, trgt, tknz, max_len):
        self.text = text
        self.trgt = trgt
        self.tknz = tknz
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text_tokens = self.tknz.encode_plus(
            self.text[idx],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        otp = {
            "input_ids": text_tokens["input_ids"].flatten(),
            "attention_mask": text_tokens["attention_mask"].flatten(),
        }
        if self.trgt is not None:
            return otp, torch.tensor(self.trgt[idx])
        else:
            return otp


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str,
        bs: int,
        trn_text,
        val_text,
        tst_text,
        trn_trgt,
        val_trgt,
        trn_aug,  # should data aug happen before tokenization? is tokenization something we do in the LightningModule?
        val_aug,
        tst_aug,
    ):
        super().__init__()
        self.task = task
        self.bs = bs
        self.trn_text = trn_text
        self.val_text = val_text
        self.tst_text = tst_text
        self.trn_trgt = trn_trgt
        self.val_trgt = val_trgt
        self.trn_aug = trn_aug  # should data aug happen before tokenization? is tokenization something we do in the LightningModule?
        self.val_aug = val_aug
        self.tst_aug = tst_aug

    def setup(self, stage=None):
        self.dataset = TextClassificationDataset(mode="train")
        self.test_dataset = TextClassificationDataset(mode="test")
        self.train_data, self.val_data = random_split(
            self.dataset, [cfg.train_size, self.val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
