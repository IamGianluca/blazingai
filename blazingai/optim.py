from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

# torch.optim.lr_scheduler.OneCycleLR


def optimizer_factory(params, hparams):
    if hparams.opt == "adam":
        return Adam(
            params,
            lr=hparams.lr,
            weight_decay=hparams.wd,
        )
    if hparams.opt == "adamw":
        return AdamW(
            params,
            lr=hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-06,
            weight_decay=hparams.wd,
            correct_bias=True,
        )
    # if hparams.opt == "sam":
    #     return SAM(
    #         params,
    #         lr=hparams.lr,
    #         momentum=hparams.mom,
    #         weight_decay=hparams.wd,
    #     )
    else:
        raise ValueError("Optimizer not supported yet.")


def lr_scheduler_factory(optimizer, hparams, data_loader):
    steps_per_epoch = len(data_loader)
    if hparams.sched == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=2,
            threshold=0.01,
            factor=0.1,
            verbose=True,
        )
    elif hparams.sched == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=hparams.lr,
            cycle_momentum=True,
            pct_start=0.1,  # hparams.warmup_epochs / hparams.epochs,
            div_factor=25.0,
            final_div_factor=100000.0,
            steps_per_epoch=steps_per_epoch,
            epochs=hparams.epochs,
        )
    elif hparams.sched == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * hparams.warmup_epochs,
            num_training_steps=steps_per_epoch * hparams.epochs,
        )
    elif hparams.sched == "cosine_with_restart":
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            eta_min=1e-4,
        )
    else:
        raise ValueError("Learning rate scheduler not supported yet.")
