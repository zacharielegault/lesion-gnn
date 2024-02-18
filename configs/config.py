from drgnet.datasets.aptos import AptosConfig
from drgnet.datasets.datamodule import DataConfig
from drgnet.datasets.ddr import DDRConfig, DDRVariant
from drgnet.datasets.nodes.lesions import LesionsNodesConfig, WhichFeatures
from drgnet.models.base import LossType, LRSchedulerConfig, OptimizerAlgo, OptimizerConfig
from drgnet.models.gat import GATConfig
from drgnet.transforms import TransformConfig
from drgnet.utils import ClassWeights
from drgnet.utils.config import Config

__all__ = ["cfg"]

NODES_CONFIG = LesionsNodesConfig(
    which_features=WhichFeatures.ENCODER,
    feature_layer=4,
    reinterpolation=(512, 512),
)
MAX_EPOCHS = 500

cfg = Config(
    dataset=DataConfig(
        train_datasets=[
            DDRConfig(
                root="data/DDR",
                nodes=NODES_CONFIG,
                variant=DDRVariant.TRAIN,
            ),
        ],
        val_datasets=[
            DDRConfig(
                root="data/DDR",
                nodes=NODES_CONFIG,
                variant=DDRVariant.VALID,
            ),
        ],
        test_datasets=[
            DDRConfig(
                root="data/DDR",
                nodes=NODES_CONFIG,
                variant=DDRVariant.TEST,
            ),
            AptosConfig(
                root="data/aptos",
                nodes=NODES_CONFIG,
            ),
        ],
        transforms=[
            TransformConfig(name="KNNGraph", kwargs={"k": 32, "loop": True}),
        ],
        batch_size=10000,
        num_workers=0,
    ),
    model=GATConfig(
        optimizer=OptimizerConfig(
            lr=1e-3,
            lr_scheduler=LRSchedulerConfig(
                name="LinearWarmupCosineAnnealingLR",
                kwargs={"warmup_epochs": 10, "max_epochs": MAX_EPOCHS},
            ),
            weight_decay=1e-4,
            algo=OptimizerAlgo.ADAMW,
            loss_type=LossType.SMOOTH_L1,
            class_weights_mode=ClassWeights.QUADRATIC_INVERSE,
        ),
        hiddden_channels=[256, 256, 256],
        heads=8,
        dropout=0.5,
        compile=True,
    ),
    monitored_metric="val_DDR_micro_acc",
    monitor_mode="max",
    max_epochs=MAX_EPOCHS,
    seed=1234,
    project_name="Aptos-GNN",
    tag="LESIONS",
)
