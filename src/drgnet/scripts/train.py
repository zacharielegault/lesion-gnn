#! /usr/bin/env python

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, ToSparseTensor
from tqdm import tqdm

from drgnet.datasets import Aptos
from drgnet.model import DRGNet
from drgnet.transforms import GaussianDistance


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    radius_sigma_px = 10
    transform = Compose(
        [
            RadiusGraph(3 * radius_sigma_px),
            GaussianDistance(sigma=radius_sigma_px),
            ToSparseTensor(),
        ]
    )
    dataset = Aptos(root="data/aptos/", transform=transform)
    train, val = dataset.split(0.8, 0.2)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64)

    model = DRGNet(dataset.num_features, 32, 4, 50, dataset.num_classes).to(device)
    optim = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    acc, kappa = [], []
    outer_pbar = tqdm(range(1, 210 + 1))
    for _ in outer_pbar:
        # Train
        model.train()
        pbar = tqdm(train_loader, position=1, leave=False)
        for data in pbar:
            data = data.to(device)
            logits = model(data.x, data.adj_t, data.batch, data.edge_attr.squeeze())
            loss = criterion(logits, data.y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, position=1, leave=False)
            for i, data in enumerate(pbar):
                data = data.to(device)
                logits = model(data.x, data.adj_t, data.batch, data.edge_attr.squeeze())
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(data.y.cpu().tolist())

        val_acc = accuracy_score(targets, preds)
        val_kappa = cohen_kappa_score(targets, preds, weights="quadratic")
        outer_pbar.set_description(f"Accuracy: {val_acc:.4f}, Kappa: {val_kappa:.4f}")
        acc.append(val_acc)
        kappa.append(val_kappa)

    print(f"Best validation accuracy: {max(acc):.4f}")
    print(f"Best validation kappa: {max(kappa):.4f}")


if __name__ == "__main__":
    main()
