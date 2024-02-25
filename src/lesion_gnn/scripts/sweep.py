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
from lesion_gnn.models.gat import GATConfig
from lesion_gnn.models.gin import GINConfig
from lesion_gnn.training import train
from lesion_gnn.transforms import TransformConfig
from lesion_gnn.utils import ClassWeights
from lesion_gnn.utils.config import Config

PROJECT_NAME = "SweepLesionsGNN"
wandb_callback = WeightsAndBiasesCallback(wandb_kwargs={"project": PROJECT_NAME}, as_multirun=True)


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
    study.optimize(objective, n_trials=500)


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
    COMPILE = True
    EARLY_STOPPING_PATIENCE = 50

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
        ["CosineAnnealingWarmRestarts", "LinearWarmupCosineAnnealingLR", None],
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
        case None:
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
    connectivity = trial.suggest_categorical("connectivity", ["knn", "radius"])
    match connectivity:
        case "knn":
            transforms = [
                TransformConfig(
                    name="KNNGraph", kwargs={"k": trial.suggest_int("conectivity/k", 2, 32, log=True), "loop": True}
                ),
            ]
        case "radius":
            transforms = [
                TransformConfig(
                    name="RadiusGraph", kwargs={"r": trial.suggest_float("connectivity/r", 1, 1536, log=True)}
                ),
            ]
        case _:
            raise ValueError("Unknown connectivity")

    # Model config
    model_arch = trial.suggest_categorical("model/arch", ["GAT", "GIN"])
    match model_arch:
        case "GAT":
            layer_size = trial.suggest_categorical("model/gat/layer_size", [32, 64, 128, 256, 512])
            num_layers = trial.suggest_int("model/gat/num_layers", 1, 8, log=True)
            model_config = GATConfig(
                optimizer=optimizer_config,
                hiddden_channels=[layer_size] * num_layers,
                heads=trial.suggest_categorical("model/gat/heads", [1, 2, 4, 8]),
                dropout=trial.suggest_float("model/gat/dropout", 0.1, 0.9),
                compile=COMPILE,
            )
        case "GIN":
            layer_size = trial.suggest_categorical("model/gin/layer_size", [32, 64, 128, 256, 512])
            num_layers = trial.suggest_int("model/gin/num_layers", 1, 8, log=True)
            model_config = GINConfig(
                optimizer=optimizer_config,
                hidden_channels=[layer_size] * num_layers,
                dropout=trial.suggest_float("model/gin/dropout", 0.1, 0.9),
                compile=COMPILE,
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
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        max_epochs=MAX_EPOCHS,
        seed=1234,
        project_name=PROJECT_NAME,
        tags=["LESIONS"],
    )

    return config


if __name__ == "__main__":
    main()
