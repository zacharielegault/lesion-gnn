from lesion_gnn.datasets.aptos import AptosConfig
from lesion_gnn.datasets.datamodule import DataConfig
from lesion_gnn.datasets.ddr import DDRConfig, DDRVariant
from lesion_gnn.datasets.nodes.lesions import LesionsNodesConfig, WhichFeatures
from lesion_gnn.models.base import LossType, LRSchedulerConfig, OptimizerAlgo, OptimizerConfig
from lesion_gnn.models.gin import GINConfig
from lesion_gnn.transforms import TransformConfig
from lesion_gnn.utils import ClassWeights
from lesion_gnn.utils.config import Config

__all__ = ["cfg"]

NODES_CONFIG = LesionsNodesConfig(
    which_features=WhichFeatures.DECODER,
    feature_layer=0,
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
    model=GINConfig(
        optimizer=OptimizerConfig(
            lr=1e-3,
            lr_scheduler=LRSchedulerConfig(
                name="LinearWarmupCosineAnnealingLR",
                kwargs={"warmup_epochs": 10, "max_epochs": MAX_EPOCHS},
            ),
            weight_decay=1e-4,
            algo=OptimizerAlgo.ADAMW,
            loss_type=LossType.CE,
            class_weights_mode=ClassWeights.UNIFORM,
        ),
        hidden_channels=[128, 128, 128],
        dropout=0.5,
        compile=True,
    ),
    monitored_metric="val_DDR_kappa",
    monitor_mode="max",
    early_stopping_patience=100,
    max_epochs=MAX_EPOCHS,
    seed=1234,
    project_name="SweepLesionsGNN",
    tags=["LESIONS"],
)
