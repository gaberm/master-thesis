from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def get_callbacks(config, ckpt_dir):
    lr_callback = LearningRateMonitor(logging_interval="step")
    if "fft" in config.trainer.exp_name:
        epoch_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="epoch",
            filename=f"{{epoch}}-{{{config.params.pred_metric}:.3f}}",
            save_top_k=config.trainer.max_epochs,
            save_on_train_epoch_end=True,
        )
        return [lr_callback, epoch_checkpoint]
    else:
        val_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=config.params.pred_metric,
            mode="max",
            filename=f"{{epoch}}-{{step}}-{{{config.params.pred_metric}:.3f}}",
            save_top_k=-1
        )
        return [lr_callback, val_checkpoint]
