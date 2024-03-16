from typing import Any, List, Optional

import lightning as pl
import timm
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoConfig, AutoModel

from blazingai.loss import loss_factory
from blazingai.metrics import metric_factory

from blazingai.optim import lr_scheduler_factory, optimizer_factory
from blazingai.text.reinitialize import reinit_autoencoder_model


class ImageClassifier(pl.LightningModule):
    """2D or 3D image classification tasks, including binary, multi-class, and
    multi-labels."""

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        self.backbone = timm.create_model(
            model_name=self.hparams.arch,  # type: ignore
            pretrained=self.hparams.pretrained,  # type: ignore
            num_classes=0,
            in_chans=self.hparams.in_channels,  # type: ignore
            drop_rate=self.hparams.dropout,  # type: ignore
        )
        self.head = nn.Sequential(
            nn.LazyLinear(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.Linear(64, self.hparams.num_classes),  # type: ignore
        )

        self.train_metric = metric_factory(cfg=cfg)
        self.val_metric = metric_factory(cfg=cfg)
        self.best_train_metric = None
        self.best_val_metric = None

    def forward(self, x):
        """Contain only tensor operations with your model."""
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        """Encapsulate forward() logic with logging, metrics, and loss
        computation.
        """
        x, target = batch
        loss, target, preds = self._step(x, target)

        if target.shape[1] == 3:  # if using MixUp, compute target
            lam = target[:, 2]
            target = (1 - lam) * target[:, 0] + lam * target[:, 1]
            target = target.reshape(-1, 1)

        self.log("train_loss", loss, on_step=True, on_epoch=False)  # type: ignore
        self.train_metric.update(preds=preds, target=target)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_metric", self.train_metric.compute())

    def validation_step(self, batch, batch_idx):
        x, target = batch
        loss, target, preds = self._step(x, target)

        self.log("val_loss", loss, on_step=True, on_epoch=False)  # type: ignore
        self.val_metric.update(preds=preds, target=target)
        return loss

    def validation_epoch_end(self, outputs: List):
        self.log("val_metric", self.val_metric.compute())
        self._register_best_train_and_val_metrics()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        """Encapsulate forward() with any necessary preprocess or postprocess
        functions.
        """
        x = batch

        preds = self.forward(x)
        # TODO: we don't always need a sigmoid. Handle that case.
        outs = preds.sigmoid()
        return outs.detach().cpu().float().numpy()

    def _step(self, x, target):
        preds = self.forward(x)
        # TODO: handle logit vs. no logit case for both loss and preds
        loss = self._compute_loss(preds=preds, target=target)
        return loss, target, preds.sigmoid()

    def configure_optimizers(self):
        optimizer = optimizer_factory(params=self.parameters(), hparams=self.hparams)

        scheduler = lr_scheduler_factory(
            optimizer=optimizer,
            hparams=self.hparams,
            data_loader=self.trainer.datamodule.train_dataloader(),  # type: ignore
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_metric",
                "strict": True,
                "name": "lr",
            },
        }

    def _compute_loss(self, preds, target):
        if self.hparams.label_smoothing > 0.0:  # type: ignore
            target = (
                target * (1 - self.hparams.label_smoothing)  # type: ignore
                + 0.5 * self.hparams.label_smoothing  # type: ignore
            )

        loss_fn = loss_factory(name=self.hparams.loss)  # type: ignore
        loss = loss_fn(preds, target)
        return loss

    def _register_best_train_and_val_metrics(self):
        try:
            train_metric = self.trainer.callback_metrics["train_metric"]
            val_metric = self.trainer.callback_metrics["val_metric"]
            if self.best_val_metric is None or self._is_metric_better(val_metric):
                self.best_val_metric = val_metric
                self.best_train_metric = train_metric
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass

    def _is_metric_better(self, new_metric):
        if self.hparams.metric_mode == "max":  # type: ignore
            return new_metric > self.best_val_metric
        elif self.hparams.metric_mode == "min":  # type: ignore
            return new_metric < self.best_val_metric
        else:
            raise ValueError("metric_mode can only be min or max")


class TextClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name,  # type: ignore
            output_hidden_states=True,
        )
        self.config.update(
            {
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
            }
        )
        self.backbone = AutoModel.from_pretrained(
            self.hparams.model_name,  # type: ignore
            config=self.config,
        )
        if hasattr(self.hparams, "reinit_layers"):
            self.backbone.encoder = reinit_autoencoder_model(
                self.backbone.encoder, reinit_num_layers=self.hparams.reinit_layers
            )
        self.dropout = nn.Dropout(self.hparams.drop)  # type: ignore
        self.pooling_params = {"pooling_name": "AttentionHead"}
        self.pooling_params.update(
            {
                "in_features": self.config.hidden_size,
                "out_features": self.config.hidden_size,
            }
        )
        self.pooling = NLPPooling(**self.pooling_params)
        self.head = nn.LazyLinear(self.hparams.out_nodes)  # type: ignore

        if hasattr(self.hparams, "yrange"):
            self.ymin, self.ymax = self.hparams.yrange  # type: ignore

        self.train_metric = metric_factory(cfg=cfg)
        self.val_metric = metric_factory(cfg=cfg)
        self.best_train_metric = None
        self.best_val_metric = None

    def forward(self, batch):
        x = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        x = self.dropout(x)  # bs x token_size x emb_size
        x = self.pooling(x, batch["attention_mask"])
        x = self.head(x)
        if hasattr(self.hparams, "yrange"):
            x = torch.sigmoid(x) * (self.ymax - self.ymin) + self.ymin
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch, batch["labels"]
        loss, y, y_hat = self._step(x, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False)  # type: ignore
        self.train_metric.update(preds=y_hat, target=y)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("train_metric", self.train_metric.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch["labels"]
        loss, y, y_hat = self._step(x, y)

        self.log("val_loss", loss, on_step=True, on_epoch=False)  # type: ignore
        self.val_metric.update(preds=y_hat, target=y)
        return loss

    def _step(self, x, target):
        preds = self(x)
        # TODO: handle logit vs. no logit case for both loss and preds
        loss = self._compute_loss(preds=preds, target=target)
        return loss, target, preds

    def _compute_loss(self, preds, target):
        loss_fn = loss_factory(name=self.hparams.loss)  # type: ignore
        loss = loss_fn(preds, target)
        return loss

    def validation_epoch_end(self, outputs: List):
        self.log("val_metric", self.val_metric.compute())
        self._register_best_train_and_val_metrics()

    def predict_step(self, batch, batch_idx) -> torch.tensor:
        x = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = optimizer_factory(params=self.parameters(), hparams=self.hparams)

        scheduler = lr_scheduler_factory(
            optimizer=optimizer,
            hparams=self.hparams,
            data_loader=self.trainer.datamodule.train_dataloader(),  # type: ignore
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_metric",
                "strict": True,
                "name": "lr",
            },
        }

    def _register_best_train_and_val_metrics(self):
        try:
            train_metric = self.trainer.callback_metrics["train_metric"]
            val_metric = self.trainer.callback_metrics["val_metric"]
            if self.best_val_metric is None or self._is_metric_better(val_metric):
                self.best_val_metric = val_metric
                self.best_train_metric = train_metric
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass

    def _is_metric_better(self, new_metric):
        if self.hparams.metric_mode == "max":  # type: ignore
            return new_metric > self.best_val_metric
        elif self.hparams.metric_mode == "min":  # type: ignore
            return new_metric < self.best_val_metric
        else:
            raise ValueError("metric_mode can only be min or max")


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features, attention_mask):
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score[attention_mask == 0] = -1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * weights_mask * features, dim=1)
        return context_vector


class NLPPooling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if self.pooling_name == "AttentionHead":
            self.pooler = AttentionHead(self.in_features, self.out_features)
        elif self.pooling_name not in ("CLS", ""):
            self.pooler = eval(self.pooling_name)(**self.params)

        print(f"Pooling: {self.pooling_name}")

    def forward(self, last_hidden_state, attention_mask):
        if self.pooling_name in ["MeanPooling", "MaxPooling", "MinPooling"]:
            # Pooling between cls and sep / cls and sep embedding are not included
            # last_hidden_state = self.pooler(last_hidden_state[:,1:-1,:],attention_mask[:,1:-1])
            last_hidden_state = self.pooler(last_hidden_state, attention_mask)
        elif self.pooling_name == "CLS":
            # Use only cls embedding
            last_hidden_state = last_hidden_state[:, 0, :]
        elif self.pooling_name == "GeMText":
            # Use Gem Pooling on all tokens
            last_hidden_state = self.pooler(last_hidden_state, attention_mask)

        elif self.pooling_name == "AttentionHead":
            last_hidden_state = self.pooler(last_hidden_state, attention_mask)
        else:
            # No pooling
            last_hidden_state = last_hidden_state
            # print(f"{self.pooling_name} not implemented")
        return last_hidden_state
