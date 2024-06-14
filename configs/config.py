from lesion_gnn.datasets.aptos import AptosConfig
from lesion_gnn.datasets.datamodule import DataConfig
from lesion_gnn.datasets.ddr import DDRConfig, DDRVariant
from lesion_gnn.datasets.nodes.lesions import LesionsNodesConfig, TimmEncoderFeatures
from lesion_gnn.models.base import LossType, OptimizerAlgo, OptimizerConfig
from lesion_gnn.models.gat import GATConfig
from lesion_gnn.transforms import TransformConfig
from lesion_gnn.utils import ClassWeights
from lesion_gnn.utils.config import Config

__all__ = ["cfg"]

NODES_CONFIG = LesionsNodesConfig(
    feature_source=TimmEncoderFeatures(timm_model="hf_hub:ClementP/FundusDRGrading-convnext_base", layer=-1),
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
            TransformConfig(name="KNNGraph", kwargs={"k": 6, "loop": True}),
        ],
        batch_size=10000,
        num_workers=0,
    ),
    model=GATConfig(
        optimizer=OptimizerConfig(
            lr=1e-3,
            lr_scheduler=None,
            weight_decay=2e-6,
            algo=OptimizerAlgo.ADAM,
            loss_type=LossType.MSE,
            class_weights_mode=ClassWeights.UNIFORM,
        ),
        hiddden_channels=[128] * 4,
        heads=2,
        dropout=0.35,
        compile=True,
    ),
    monitored_metric="val_DDR_kappa",
    monitor_mode="max",
    early_stopping_patience=None,
    max_epochs=MAX_EPOCHS,
    seed=1234,
    project_name="SweepLesionsGNN",
    tags=["LESIONS"],
)
