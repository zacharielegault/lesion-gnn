import logging
import sys

import optuna
import torch
from optuna.integration.wandb import WeightsAndBiasesCallback

from lesion_gnn.datasets.aptos import AptosConfig
from lesion_gnn.datasets.datamodule import DataConfig
from lesion_gnn.datasets.ddr import DDRConfig, DDRVariant
from lesion_gnn.datasets.nodes.lesions import LesionsNodesConfig, WhichFeatures
from lesion_gnn.models.base import LossType, LRSchedulerConfig, OptimizerAlgo, OptimizerConfig
from lesion_gnn.models.drgnet import DRGNetModelConfig
from lesion_gnn.models.gat import GATConfig
from lesion_gnn.models.pointnet import PointNetModelConfig  # noqa: F401
from lesion_gnn.models.set_transformer import SetTransformerModelConfig
from lesion_gnn.training import train
from lesion_gnn.transforms import TransformConfig
from lesion_gnn.utils import ClassWeights
from lesion_gnn.utils.config import Config

PROJECT_NAME = "SweepLesionsGNN"
wandb_callback = WeightsAndBiasesCallback(wandb_kwargs={"project": PROJECT_NAME})


def main():
    # Add stream handler to stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study_name = "sweep"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=500, callbacks=[wandb_callback])


@wandb_callback.track_in_wandb()
def objective(trial):
    config = make_config(trial)

    try:
        metrics = train(config)
    except torch.cuda.OutOfMemoryError:
        raise optuna.TrialPruned("Out of memory")

    return metrics["test_Aptos_kappa"], metrics["test_DDR_kappa"]


def make_config(trial: optuna.Trial) -> Config:
    MAX_EPOCHS = 500

    # Dataset config
    which_features = trial.suggest_categorical("nodes/which_features", list(WhichFeatures))
    feature_layer = trial.suggest_int("nodes/feature_layer", 1, 4) if which_features == WhichFeatures.ENCODER else 0

    nodes_config = LesionsNodesConfig(
        which_features=which_features,
        feature_layer=feature_layer,
        reinterpolation=(512, 512),
    )

    # Optimizer config
    scheduler_name = trial.suggest_categorical(
        "optimizer/scheduler/algo",
        ["CosineAnnealingWarmRestarts", "LinearWarmupCosineAnnealingLR", ""],
    )
    match scheduler_name:
        case "CosineAnnealingWarmRestarts":
            lr_scheduler_config = LRSchedulerConfig(
                name=scheduler_name,
                kwargs={
                    "T_0": trial.suggest_int("optimizer/scheduler/CosineAnnealingWarmRestarts/T_0", 1, 200, log=True),
                    "T_mult": trial.suggest_int("optimizer/scheduler/CosineAnnealingWarmRestarts/T_mult", 1, 3),
                },
            )
        case "LinearWarmupCosineAnnealingLR":
            lr_scheduler_config = LRSchedulerConfig(
                name=scheduler_name,
                kwargs={
                    "max_epochs": MAX_EPOCHS,
                    "warmup_epochs": trial.suggest_int(
                        "optimizer/scheduler/LinearWarmupCosineAnnealingLR/warmup_epochs", 1, 100, log=True
                    ),
                },
            )
        case "":
            lr_scheduler_config = None
        case _:
            raise ValueError("Unknown scheduler")

    optimizer_config = OptimizerConfig(
        lr=trial.suggest_float("optimizer/lr", 1e-6, 1e-1, log=True),
        lr_scheduler=lr_scheduler_config,
        weight_decay=trial.suggest_float("optimizer/weight_decay", 1e-6, 1e-1, log=True),
        algo=trial.suggest_categorical("optimizer/algo", list(OptimizerAlgo)),
        loss_type=trial.suggest_categorical("optimizer/loss_type", list(LossType)),
        class_weights_mode=trial.suggest_categorical("optimizer/class_weights_mode", list(ClassWeights)),
    )

    # Transforms
    transforms = [
        TransformConfig(name="KNNGraph", kwargs={"k": 32, "loop": True}),
    ]

    # Model config
    model_arch = trial.suggest_categorical("model/arch", ["DRGNet", "SetTransformer", "GAT"])
    match model_arch:
        case "DRGNet":
            model_config = DRGNetModelConfig(
                optimizer=optimizer_config,
                gnn_hidden_dim=trial.suggest_int("model/drgnet/gnn_hidden_dim", 32, 512, log=True),
                num_layers=trial.suggest_int("model/drgnet/num_layers", 1, 8),
                sortpool_k=trial.suggest_int("model/drgnet/sortpool_k", 1, 128, log=True),
                conv_hidden_dims=(16, 32),
                compile=True,
            )
        # case "PointNet":
        #     model_config = PointNetModelConfig(
        #         optimizer=optimizer_config,
        #         pos_dim=3,
        #         compile=True,
        #     )
        case "SetTransformer":
            model_config = SetTransformerModelConfig(
                optimizer=optimizer_config,
                inner_dim=trial.suggest_categorical("model/set_transformer/inner_dim", [32, 64, 128, 256, 512]),
                num_inducing_points=trial.suggest_int("model/set_transformer/num_inducing_points", 1, 32, log=True),
                num_seed_points=trial.suggest_int("model/set_transformer/num_seed_points", 1, 21, log=True),
                num_encoder_blocks=trial.suggest_int("model/set_transformer/num_encoder_blocks", 1, 8),
                num_decoder_blocks=trial.suggest_int("model/set_transformer/num_decoder_blocks", 1, 2),
                heads=trial.suggest_categorical("model/set_transformer/heads", [1, 2, 4, 8]),
                layer_norm=True,
                dropout=trial.suggest_float("model/set_transformer/dropout", 0.1, 0.9),
                compile=True,
            )
        case "GAT":
            layer_size = trial.suggest_categorical("model/gat/layer_size", [8, 16, 32, 64, 128, 256, 512])
            num_layers = trial.suggest_int("model/gat/num_layers", 1, 8, log=True)
            model_config = GATConfig(
                optimizer=optimizer_config,
                hiddden_channels=[layer_size] * num_layers,
                heads=trial.suggest_categorical("model/gat/heads", [1, 2, 4, 8]),
                dropout=trial.suggest_float("model/gat/dropout", 0.1, 0.9),
                compile=True,
            )
        case _:
            raise ValueError("Unknown model")

    # Finally, the full config
    config = Config(
        dataset=DataConfig(
            train_datasets=[
                DDRConfig(
                    root="data/DDR",
                    nodes=nodes_config,
                    variant=DDRVariant.TRAIN,
                ),
            ],
            val_datasets=[
                DDRConfig(
                    root="data/DDR",
                    nodes=nodes_config,
                    variant=DDRVariant.VALID,
                ),
            ],
            test_datasets=[
                DDRConfig(
                    root="data/DDR",
                    nodes=nodes_config,
                    variant=DDRVariant.TEST,
                ),
                AptosConfig(
                    root="data/aptos",
                    nodes=nodes_config,
                ),
            ],
            transforms=transforms,
            batch_size=10000,
            num_workers=0,
        ),
        model=model_config,
        monitored_metric="val_DDR_kappa",
        monitor_mode="max",
        max_epochs=MAX_EPOCHS,
        seed=1234,
        project_name=PROJECT_NAME,
        tags=["LESIONS"],
    )

    return config


if __name__ == "__main__":
    main()
