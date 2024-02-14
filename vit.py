import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CohenKappa, ConfusionMatrix, F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from drgnet.fundus_datamodules import (
    AptosClassificationDataModule,
    DDRSegmentationDataModule,
    MaplesSegmentationDataModule,
)
from masked_vit import make_model


def main():
    # batch_size = 4
    # num_workers = 6

    ddr = DDRSegmentationDataModule(
        "data/DDR-dataset",
        img_size=(1024, 1024),
        return_label=True,
        batch_size=32,
        num_workers=32,
        persistent_workers=True,
        training_data_aug=False,
    )
    maples = MaplesSegmentationDataModule(
        "data/MAPLES-DR",
        img_size=(1024, 1024),
        return_label=True,
        batch_size=32,
        num_workers=32,
        persistent_workers=True,
        training_data_aug=False,
    )
    aptos = AptosClassificationDataModule(
        "data/APTOS2019",
        img_size=(1024, 1024),
        batch_size=8,
        num_workers=6,
        persistent_workers=True,
        training_data_aug=False,
    )

    run(aptos, "APTOS", compute_masks=True)

    #                       acc      prec       rec        f1       qwk
    # no_mask          0.711914  0.542139  0.546030  0.509347  0.891624
    # with_mask        0.108398  0.295695  0.230812  0.132031  0.597662
    # with_index       0.112793  0.305985  0.233570  0.137170  0.600433
    # with_mask_flip   0.658203  0.532826  0.441550  0.465054  0.696549
    # with_index_flip  0.658203  0.532826  0.441550  0.465054  0.696549

    run(ddr, "DDR", compute_masks=False)
    #                       acc      prec       rec        f1       qwk
    # no_mask          0.672727  0.460128  0.515280  0.447120  0.581169
    # with_mask        0.327273  0.348676  0.435925  0.267967  0.452276
    # with_index       0.225455  0.283019  0.384174  0.210096  0.279944
    # with_mask_flip   0.138182  0.235336  0.102309  0.123336  0.259691
    # with_index_flip  0.600000  0.399528  0.447420  0.397416  0.514214

    run(maples, "MAPLES", compute_masks=False)
    #                       acc      prec       rec        f1       qwk
    # no_mask          0.340580  0.455342  0.486414  0.367571  0.532950
    # with_mask        0.376812  0.269510  0.372882  0.237791  0.348931
    # with_index       0.318841  0.452826  0.480473  0.356967  0.534733
    # with_mask_flip   0.478261  0.189535  0.218812  0.194706  0.066280
    # with_index_flip  0.318841  0.450562  0.502695  0.356173  0.547391


def run(dm: LightningDataModule, dataset_name: str, compute_masks: bool) -> None:
    dm.setup("fit")
    dataloader = dm.train_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = make_model()

    metrics = compute_preds(model, dataloader, device, compute_masks=compute_masks)

    df = compute_metrics(metrics)
    print(f"\n{dataset_name}\n")
    print(df)

    plot_confusion_matrices(metrics, dataset_name)


def plot_confusion_matrices(metrics: dict[str, MetricCollection], save_name) -> None:
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    for ax, (key, metric) in zip(axs, metrics.items()):
        metric["cm"].plot(ax=ax)
        ax.set_title(key)

    fig.tight_layout()
    fig.savefig(f"confusion_matrices_{save_name}.png")


def compute_preds(
    model: nn.Module, dataloader: DataLoader, device: torch.device, compute_masks: bool
) -> dict[str, MetricCollection]:
    _metrics = MetricCollection(
        {
            "acc": Accuracy(task="multiclass", average="micro", num_classes=5),
            "prec": Precision(task="multiclass", average="macro", num_classes=5),
            "rec": Recall(task="multiclass", average="macro", num_classes=5),
            "f1": F1Score(task="multiclass", average="macro", num_classes=5),
            "qwk": CohenKappa(task="multiclass", weights="quadratic", num_classes=5),
            "cm": ConfusionMatrix(task="multiclass", num_classes=5),
        }
    ).to(device)

    metrics = {
        "no_mask": _metrics.clone(),
        "with_mask": _metrics.clone(),
        "with_index": _metrics.clone(),
        "with_mask_flip": _metrics.clone(),
        "with_index_flip": _metrics.clone(),
    }
    with torch.inference_mode():
        model = model.to(device).eval()
        for batch in tqdm(dataloader):
            if len(batch) == 2:
                img, label = batch
                mask = None
            else:  # len(batch) == 3
                img, mask, label = batch

            img = img.to(device)
            mask = mask.to(device) if mask is not None else None
            label = label.to(device)

            no_mask = model(img)
            with_mask = model(img, mask=mask, mode="mask", compute_masks=compute_masks)
            with_index = model(img, mask=mask, mode="index", compute_masks=compute_masks)
            with_mask_flip = model(img, mask=mask, mode="mask", flip_mask=True, compute_masks=compute_masks)
            with_index_flip = model(img, mask=mask, mode="index", flip_mask=True, compute_masks=compute_masks)

            metrics["no_mask"].update(no_mask.round().long().squeeze(), label)
            metrics["with_mask"].update(with_mask.round().long().squeeze(), label)
            metrics["with_index"].update(with_index.round().long().squeeze(), label)
            metrics["with_mask_flip"].update(with_mask_flip.round().long().squeeze(), label)
            metrics["with_index_flip"].update(with_index_flip.round().long().squeeze(), label)

    return metrics


def compute_metrics(data: dict[str, MetricCollection]) -> pd.DataFrame:
    df = pd.DataFrame(columns=["acc", "prec", "rec", "f1", "qwk"])

    for key, metrics in data.items():
        values = metrics.compute()
        values.pop("cm")
        values = {k: v.cpu().item() for k, v in values.items()}
        df.loc[key] = [values["acc"], values["prec"], values["rec"], values["f1"], values["qwk"]]

    return df


if __name__ == "__main__":
    main()
