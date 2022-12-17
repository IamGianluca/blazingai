import os
from pathlib import Path
from typing import List, Optional

import lightning as pl
from lightning.pytorch.core.saving import DictConfig
import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        trgt_cols: List[str],
        fold: int,
        data_path: Path,
        bs: int,
        cfg: DictConfig
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.trgt_cols = trgt_cols
        self.fold = fold
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )
        self.bs = bs
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        """How to split, define dataset, etc..."""
        df = pd.read_csv(self.data_path)
        df["labels"] = df[self.trgt_cols].values.tolist()

        if stage == "fit":
            ds = DatasetDict(
                {
                    "trn": Dataset.from_pandas(df[df.kfold != self.fold]),
                    "val": Dataset.from_pandas(df[df.kfold == self.fold]),
                }
            )
        elif stage == "predict":
            ds = DatasetDict({"tst": Dataset.from_pandas(df[df.kfold == self.fold])})
        else:
            raise ValueError(f"stage `{stage}` currently not supported")

        self.ds_encoded = ds.with_format("torch").map(self._encode)

    def _encode(self, examples):
        text = examples["full_text"]
        encoding = self.tokenizer(
            text,
            padding=self.cfg.padding, 
            truncation=self.cfg.truncation, 
            max_length=self.cfg.max_length
        )
        encoding["labels"] = examples["labels"].tolist()
        return encoding

    def train_dataloader(self):
        return DataLoader(
            self.ds_encoded["trn"],
            batch_size=self.bs,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_encoded["val"],
            batch_size=self.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_encoded["tst"],
            batch_size=self.bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            drop_last=False,
        )
