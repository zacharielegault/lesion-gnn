from drgnet.models.drgnet import DRGNetModelConfig
from drgnet.utils import Config, DatasetConfig

__all__ = ["cfg"]

cfg = Config(
    dataset=DatasetConfig(
        name="aptos-ddr",
        root_aptos="data/aptos",
        root_ddr="data/DDR",
        split=[0.8, 0.2],
        num_keypoints=50,
        sift_sigma=1.6,
        distance_sigma_px=70,
    ),
    model=DRGNetModelConfig(
        gnn_hidden_dim=64,
        num_layers=6,
        sortpool_k=50,
        conv_hidden_dims=(16, 32),
        compile=False,
    ),
    batch_size=5000,
    max_epochs=500,
    seed=42,
    project_name="Aptos-GNN",
    tag="SIFT",
)
