import wandb
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from fundus_datamodules import DDRClassificationDataModule
from lesion_aware_transformer import LesionAwareTransformer


def main():
    NUM_WORKERS = 6
    BATCH_SIZE = 16
    IMG_SIZE = (512, 512)

    dm = DDRClassificationDataModule(
        "data/DDR-dataset", batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS
    )
    dm.setup("test")

    logger = WandbLogger(
        project="lesion-aware-transformer",
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
    )

    ckpt_path = "checkpoints/serene-vortex-38/last.ckpt"
    model = LesionAwareTransformer.load_from_checkpoint(ckpt_path)

    trainer = Trainer(devices=[0], logger=logger, log_every_n_steps=1, precision=16)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
