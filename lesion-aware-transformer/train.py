import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from fundus_datamodules import AptosClassificationDataModule
from lesion_aware_transformer import LesionAwareTransformer
from lesion_gnn.callbacks import ConfusionMatrixCallback


def main():
    dm = AptosClassificationDataModule("data/APTOS2019", batch_size=16, img_size=(512, 512), num_workers=6)
    dm.setup("fit")

    model = LesionAwareTransformer(num_classes=5, embed_dim=256, num_filters=4)

    logger = WandbLogger(
        project="lesion-aware-transformer",
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
    )
    run_name = logger.experiment.name

    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{run_name}/",
            monitor="val_kappa",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
        ConfusionMatrixCallback(),
    ]

    trainer = Trainer(devices=[0], logger=logger, max_epochs=10, log_every_n_steps=1, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)

    dm.setup("test")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
