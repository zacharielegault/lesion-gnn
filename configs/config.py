from drgnet.datasets.aptos import AptosConfig
from drgnet.datasets.datamodule import DataConfig
from drgnet.datasets.ddr import DDRConfig, DDRVariant
from drgnet.datasets.nodes.lesions import LesionsNodesConfig, WhichFeatures
from drgnet.models.base import LossType, OptimizerAlgo, OptimizerConfig
from drgnet.models.pointnet import PointNetModelConfig
from drgnet.transforms import TransformConfig
from drgnet.utils import ClassWeights
from drgnet.utils.config import Config

__all__ = ["cfg"]

lesions = LesionsNodesConfig(
    which_features=WhichFeatures.ENCODER,
    feature_layer=4,
    reinterpolation=(512, 512),
)

cfg = Config(
    dataset=DataConfig(
        train_datasets=[
            DDRConfig(
                root="data/DDR",
                nodes=lesions,
                variant=DDRVariant.TRAIN,
            ),
        ],
        val_datasets=[
            DDRConfig(
                root="data/DDR",
                nodes=lesions,
                variant=DDRVariant.VALID,
            ),
        ],
        test_datasets=[
            DDRConfig(
                root="data/DDR",
                nodes=lesions,
                variant=DDRVariant.TEST,
            ),
            AptosConfig(
                root="data/aptos",
                nodes=lesions,
            ),
        ],
        transforms=[
            TransformConfig(name="RadiusGraph", kwargs={"r": 3 * 50.0, "loop": True}),
            TransformConfig(name="GaussianDistance", kwargs={"sigma": 50.0}),
        ],
        batch_size=5000,
        num_workers=0,
    ),
    model=PointNetModelConfig(
        optimizer=OptimizerConfig(
            lr=1e-3,
            schedule_warmup_epochs=10,
            weight_decay=1e-4,
            algo=OptimizerAlgo.ADAMW,
            loss_type=LossType.CE,
            class_weights_mode=ClassWeights.INVERSE,
        ),
        pos_dim=2,
        compile=False,
    ),
    monitored_metric="val_DDR_kappa",
    monitor_mode="max",
    max_epochs=500,
    seed=42,
    project_name="Aptos-GNN",
    tag="LESIONS",
)
