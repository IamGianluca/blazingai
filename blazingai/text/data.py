import os
from pathlib import Path
from typing import Optional

import lightning as pl
import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        fold: int,
        data_path: Path,
        bs: int,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.fold = fold
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )
        self.bs = bs

    def setup(self, stage: Optional[str] = None) -> None:
        """How to split, define dataset, etc..."""
        df = pd.read_csv(self.data_path)
        self.ds = DatasetDict(
            {
                "trn": Dataset.from_pandas(df[df.kfold != 0]),
                "val": Dataset.from_pandas(df[df.kfold == 0]),
                "tst": Dataset.from_pandas(df[df.kfold == 0]),
            }
        )
        self.ds.map(
            lambda x: self.tokenizer(x["full_text"], truncation=True, padding=True),
            batched=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds["trn"],
            batch_size=self.bs,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds["val"],
            batch_size=self.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds["tst"],
            batch_size=self.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            drop_last=False,
        )
