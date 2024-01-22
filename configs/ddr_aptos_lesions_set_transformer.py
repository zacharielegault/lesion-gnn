from drgnet.datasets import DatasetConfig
from drgnet.datasets.nodes.lesions import LesionsNodesConfig
from drgnet.models.base import LossType, OptimizerAlgo, OptimizerConfig
from drgnet.models.set_transformer import SetTransformerModelConfig
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
    model=SetTransformerModelConfig(
        num_classes=Placeholder(),
        input_features=Placeholder(),
        inner_dim=128,
        num_inducing_points=8,
        num_seed_points=1,
        num_encoder_blocks=2,
        num_decoder_blocks=0,
        heads=4,
        layer_norm=True,
        dropout=0.5,
        compile=False,
        optimizer=OptimizerConfig(
            lr=0.023809590039981624,
            algo=OptimizerAlgo.ADAMW,
            loss_type=LossType.CE,
            weight_decay=0.03128685798976109,
        ),
    ),
    batch_size=512,
    max_epochs=500,
    seed=42,
    project_name="Aptos-GNN",
    tag="LESIONS",
)
