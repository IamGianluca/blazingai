from pathlib import Path
from types import ModuleType
from typing import Tuple

import lightning as pl
import pandas as pd
from blazingai import learner
from blazingai.io import print_mtrc, save_mtrc, save_pred
from blazingai.metrics import CrossValMetrics
from blazingai.vision import data
from lightning.lite.utilities.seed import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from timm.data import transforms_factory


# TODO: use protocol instead of ModuleType so that we can use a fake module when
# unit testing
def train_loop(cfg: DictConfig, logger: Logger, const: ModuleType, train_routine):
    """Generic scaffolding to train a ML model. Nothing in this function should
    change, irrespectively from the ML task — e.g., NLP, CV, etc."""
    print(OmegaConf.to_yaml(cfg))

    seed_everything(seed=cfg.seed, workers=True)

    if is_crossval(cfg=cfg):
        metrics = CrossValMetrics(cfg=cfg)

        for current_fold in range(cfg.n_folds):
            cfg.fold = current_fold  # NOTE: reassigning value to existing member

            trn_score, val_score, trgt, pred = train_routine(
                cfg=cfg, const=const, logger=logger
            )
            metrics.add(trgt=trgt, pred=pred, val_score=val_score, trn_score=trn_score)

        # TODO: do not access _pred
        save_pred(fpath=Path(f"pred/model_{cfg.name}_oof.npy"), pred=metrics._pred)
        save_mtrc(fpath=const.mtrc_path / f"model_{cfg.name}.json", metrics=metrics)
        log_mtrc(logger=logger, metrics=metrics)
    else:
        train_routine(cfg=cfg, logger=logger)


def is_crossval(cfg: DictConfig) -> bool:
    return True if cfg.fold == -1 else False


def train_one_fold_computer_vision(cfg: DictConfig, logger, const, utils) -> Tuple:
    print(f"\n#####################")
    print(f"# FOLD {cfg.fold}")
    print(f"#####################")

    # get image paths and targets
    df = pd.read_csv(const.train_folds_all_fpath)
    df_trn = df[df.kfold != cfg.fold].reset_index()
    df_val = df[df.kfold == cfg.fold].reset_index()

    trn_img_paths, trn_targets = utils.get_image_paths_and_targets(df=df_trn, cfg=cfg)
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
        task=cfg.task,
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
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        cfg=cfg,
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_metric",
        mode=cfg.metric_mode,
        dirpath=const.ckpts_path,
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
    targets_list = df_val.loc[:, "target"].values.tolist()  # TODO: generalize
    preds = trainer.predict(model, dm.test_dataloader(), ckpt_path="best")
    preds_list = [p[0] * 100 for b in preds for p in b]

    print_mtrc(cfg.metric, model.best_train_metric, model.best_val_metric)
    return (
        model.best_train_metric.detach().cpu().numpy(),
        model.best_val_metric.detach().cpu().numpy(),
        targets_list,
        preds_list,
    )


def log_mtrc(logger: Logger, metrics: CrossValMetrics) -> None:
    logger.log_metrics({"cv_trn_metric": metrics.trn_metric})
    logger.log_metrics({"cv_val_metric": metrics.val_metric})
    logger.log_metrics({"oof_val_metric": metrics.oof_metric})
