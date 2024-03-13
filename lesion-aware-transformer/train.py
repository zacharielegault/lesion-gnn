import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from fundus_datamodules import DDRClassificationDataModule
from lesion_aware_transformer import LesionAwareTransformer
from lesion_gnn.callbacks import ConfusionMatrixCallback


def main():
    NUM_WORKERS = 6

    # From the paper
    IMG_SIZE = (512, 512)
    BACKBONE = "resnet50"
    W_TRIPLET = 0.04
    W_CONSISTENCY = 0.01
    PRE_LN = False
    NUM_HEADS = 8
    FFN_MULTIPLIER = 1
    NUM_FILTERS = 4

    # Unspecified in the paper
    MAX_EPOCHS = 20
    BATCH_SIZE = 16
    EMBED_DIM = 256
    TRIPLET_MARGIN = 1.0
    OPTIMIZER = "adam"
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.1

    dm = DDRClassificationDataModule(
        "data/DDR-dataset", batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS
    )
    dm.setup("fit")

    model = LesionAwareTransformer(
        num_classes=5,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        backbone=BACKBONE,
        pretrained=True,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        ffn_multiplier=FFN_MULTIPLIER,
        pre_ln=PRE_LN,
        triplet_margin=TRIPLET_MARGIN,
        w_triplet=W_TRIPLET,
        w_consistency=W_CONSISTENCY,
        optimizer=OPTIMIZER,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

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

    trainer = Trainer(devices=[0], logger=logger, max_epochs=MAX_EPOCHS, log_every_n_steps=1, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)

    dm.setup("test")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
