import dataclasses

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import masked_vit
from fundus_datamodules import DDRSegmentationDataModule
from lesion_gnn.callbacks import ConfusionMatrixCallback


@dataclasses.dataclass(kw_only=True)
class Config:
    project_name: str
    tags: list[str] | None = None


def main():
    config = Config(project_name="Masked ViT")
    L.seed_everything(1234)

    dm = DDRSegmentationDataModule(
        "data/DDR-dataset",
        return_label=True,
        img_size=(1024, 1024),
        batch_size=32,
        persistent_workers=True,
        training_data_aug=True,
    )
    dm.setup("fit")

    logged_args = dataclasses.asdict(config)

    model = MaskedViTLightningModule()

    logger = WandbLogger(
        project=config.project_name,
        settings=wandb.Settings(code_dir="."),
        entity="liv4d-polytechnique",
        tags=config.tags,
        config=logged_args,
    )
    run_name = logger.experiment.name

    trainer = L.Trainer(
        devices=[0],
        max_epochs=config.max_epochs,
        logger=logger,
        check_val_every_n_epoch=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"checkpoints/{run_name}/",
                monitor=config.monitored_metric,
                mode=config.monitor_mode,
                save_last=True,
                save_top_k=1,
            ),
            ConfusionMatrixCallback(),
        ],
    )
    trainer.fit(model, datamodule=dm)

    dm.setup("test")
    trainer.test(model, datamodule=dm, ckpt_path="best")


class MaskedViTLightningModule(L.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = masked_vit.make_model()


if __name__ == "__main__":
    main()
