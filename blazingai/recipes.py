import os
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import utils
from blazingai import learner
from blazingai.io import save_metrics, save_pred

from blazingai.metrics import metric_factory
from blazingai.vision import data
from lightning_lite.utilities.seed import seed_everything
from omegaconf import OmegaConf

from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import lr_monitor
from pytorch_lightning.loggers.wandb import WandbLogger
from timm.data import transforms_factory


def train(cfg: OmegaConf, constants):
    """Generic scaffolding to train a ML model. Nothing in this function should
    change, irrespectly from the ML task -- NLP, CV, etc."""
    print(OmegaConf.to_yaml(cfg))

    seed_everything(seed=cfg.seed, workers=False)

    logger = WandbLogger(project=os.environ["COMPETITION_SHORT_NAME"])

    is_xval = True if cfg.fold == -1 else False
    if is_xval:
        all_trgt, all_pred, all_val_scores, all_trn_scores = [], [], [], []

        for current_fold in range(cfg.n_folds):
            # NOTE: reassigning value to existing member
            cfg.fold = current_fold

            trn_score, val_score, trgt, pred = train_one_fold(cfg=cfg, logger=logger)
            all_trgt.extend(trgt)
            all_pred.extend(pred)
            all_val_scores.append(val_score)
            all_trn_scores.append(trn_score)

        # needed for final ensemble
        save_pred(fpath=f"preds/model_{cfg.name}_oof.npy", preds=all_pred)

        trn_metric = np.mean(all_trn_scores)
        val_metric = np.mean(all_val_scores)
        oof_metric = compute_oof_metric(cfg=cfg, y_pred=all_pred, y_true=all_trgt)

        save_metrics(
            fpath=constants.metrics_path / f"model_{cfg.name}.json",
            metric=cfg.metric,
            trn_metric=trn_metric,
            val_metric=val_metric,
            oof_metric=oof_metric,
        )
        logger.log_metrics({"cv_train_metric": trn_metric})
        logger.log_metrics({"cv_val_metric": val_metric})
        logger.log_metrics({"oof_val_metric": oof_metric})
    else:
        trn_metric, val_metric = train_one_fold(cfg=cfg, logger=logger)


def compute_oof_metric(cfg: OmegaConf, y_true, y_pred) -> float:
    metric = metric_factory(cfg)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    return metric(y_pred, y_true)


def train_one_fold(cfg: OmegaConf, logger, constants) -> Tuple:
    print()
    print(f"#####################")
    print(f"# FOLD {cfg.fold}")
    print(f"#####################")

    # get image paths and targets
    df = pd.read_csv(constants.train_folds_all_fpath)
    df_train = df[df.kfold != cfg.fold].reset_index()
    df_val = df[df.kfold == cfg.fold].reset_index()

    trn_img_paths, trn_targets = utils.get_image_paths_and_targets(df=df_train, cfg=cfg)
    val_img_paths, val_targets = utils.get_image_paths_and_targets(df=df_val, cfg=cfg)

    # define augmentations
    trn_aug = transforms_factory.create_transform(
        input_size=cfg.sz,
        is_training=True,
        auto_augment=f"rand-n{cfg.n_tfms}-m{cfg.magn}",
    )
    val_aug = transforms_factory.create_transform(
        input_size=cfg.sz,
        is_training=False,
    )

    # create datamodule
    dm = data.ImageDataModule(
        task="classification",
        bs=cfg.bs,
        trn_img_paths=trn_img_paths,
        val_img_paths=val_img_paths,
        tst_img_paths=val_img_paths,
        trn_trgt=trn_targets,
        val_trgt=val_targets,
        trn_aug=trn_aug,
        val_aug=val_aug,
        tst_aug=val_aug,
    )

    model = learner.ImageClassifier(
        in_channels=3,
        num_classes=1,
        pretrained=cfg.pretrained,
        cfg=cfg,
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_metric",
        mode=cfg.metric_mode,
        dirpath=constants.ckpts_path,
        filename=f"model_{cfg.name}_fold{cfg.fold}",
        save_weights_only=True,
    )
    lr_callback = lr_monitor.LearningRateMonitor(
        logging_interval="step", log_momentum=True
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=cfg.precision,
        auto_lr_find=cfg.auto_lr,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        auto_scale_batch_size=cfg.auto_batch_size,
        max_epochs=cfg.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_callback],
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    if cfg.auto_lr or cfg.auto_batch_size:
        trainer.tune(model, dm)

    trainer.fit(model, dm)
    targets_list = df_val.loc[:, "Pawpularity"].values.tolist()  # TODO: generalize
    preds = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    preds_list = [p[0] * 100 for b in preds for p in b]

    print_metrics(cfg.metric, model.best_train_metric, model.best_val_metric)
    return (
        model.best_train_metric.detach().cpu().numpy(),
        model.best_val_metric.detach().cpu().numpy(),
        targets_list,
        preds_list,
    )


def print_metrics(metric: str, trn_metric: float, val_metric: float) -> None:
    print(f"\nBest {metric}: Train {trn_metric:.4f}, Valid: {val_metric:.4f}")
