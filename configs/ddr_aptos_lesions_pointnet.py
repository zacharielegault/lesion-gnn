from drgnet.datasets import DatasetConfig
from drgnet.datasets.nodes.lesions import LesionsNodesConfig
from drgnet.models.base import LossType, OptimizerAlgo, OptimizerConfig
from drgnet.models.pointnet import PointNetModelConfig
from drgnet.utils.config import Config
from drgnet.utils.placeholder import Placeholder

__all__ = ["cfg"]

cfg = Config(
    dataset=DatasetConfig(
        name="aptos_lesions",
        root_aptos="data/aptos",
        root_ddr="data/DDR",
        split=(0.8, 0.2),
        distance_sigma_px=10.0,
        nodes=LesionsNodesConfig(
            which_features="encoder",
            feature_layer=4,
            reinterpolation=(512, 512),
        ),
    ),
    model=PointNetModelConfig(
        num_classes=Placeholder(),
        input_features=Placeholder(),
        pos_dim=2,
        optimizer=OptimizerConfig(
            lr=0.023809590039981624,
            algo=OptimizerAlgo.ADAMW,
            loss_type=LossType.CE,
            weight_decay=0.03128685798976109,
        ),
    ),
    batch_size=5000,
    max_epochs=500,
    seed=42,
    project_name="Aptos-GNN",
    tag="LESIONS",
)
