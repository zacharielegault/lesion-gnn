import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from cabnet import CABNet
from fundus_datamodules import DDRClassificationDataModule


def main():
    NUM_WORKERS = 6

    NUM_CLASSES = 5
    IMG_SIZE = (512, 512)
    BACKBONE = "densenet121"
    K = 5
    DROPOUT = 0.5
    OPTIMIZER = "adam"
    MAX_EPOCHS = 70  # In the paper they also do a first epoch with the backbone frozen and LR = 5e-3
    LR = 1e-4
    WEIGHT_DECAY = 0.00001
    BATCH_SIZE = 16

    # TODO: implement learning rate scheduler monitoring "val_loss"
    # >>> scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=3, mode="min")

    # NOTE: The paper uses less data augmentation than the one used here (only horizontal/vertical flips and up to 90
    # degrees rotation)
    dm = DDRClassificationDataModule(
        "data/DDR-dataset", batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS
    )
    dm.setup("fit")

    model = CABNet(
        num_classes=NUM_CLASSES,
        k=K,
        dropout=DROPOUT,
        backbone=BACKBONE,
        pretrained=True,
        optimizer=OPTIMIZER,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    logger = WandbLogger(
        project="cabnet",
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
    ]

    trainer = Trainer(
        devices=[0], logger=logger, max_epochs=MAX_EPOCHS, log_every_n_steps=1, callbacks=callbacks, precision=16
    )
    trainer.fit(model, datamodule=dm)

    dm.setup("test")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
