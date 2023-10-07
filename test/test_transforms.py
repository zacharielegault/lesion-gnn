import pytest
import torch
from torch_geometric.data import Data

from drgnet.transforms import GaussianDistance


def test_gaussian_distance():
    # Create a simple graph with two nodes and one edge
    edge_index = torch.tensor([[0, 1], [1, 0]])
    pos = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
    pseudo = torch.tensor([1, 1], dtype=torch.float)
    _data = Data(edge_index=edge_index, pos=pos, edge_attr=pseudo)

    # Test with sigma=1
    data = _data.clone()
    transform = GaussianDistance(sigma=1)
    data = transform(data)
    expected_edge_attr = torch.tensor([[1, 0.2420], [1, 0.2420]], dtype=torch.float)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with sigma=0.5
    data = _data.clone()
    transform = GaussianDistance(sigma=0.5)
    data = transform(data)
    expected_edge_attr = torch.tensor([[1, 0.1080], [1, 0.1080]], dtype=torch.float)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with sigma=2
    data = _data.clone()
    transform = GaussianDistance(sigma=2)
    data = transform(data)
    expected_edge_attr = torch.tensor([[1, 0.1760], [1, 0.1760]], dtype=torch.float)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with a graph with no edges
    with pytest.warns(UserWarning):
        edge_index = torch.tensor([], dtype=torch.long)
        pos = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
        pseudo = torch.tensor([], dtype=torch.float)
        data = Data(edge_index=edge_index, pos=pos, edge_attr=pseudo)
        transform = GaussianDistance(sigma=1)
        data = transform(data)
        assert data.edge_attr.numel() == 0

    # Test with a graph with one node
    with pytest.warns(UserWarning):
        edge_index = torch.tensor([], dtype=torch.long)
        pos = torch.tensor([[0, 0]], dtype=torch.float)
        pseudo = torch.tensor([], dtype=torch.float)
        data = Data(edge_index=edge_index, pos=pos, edge_attr=pseudo)
        transform = GaussianDistance(sigma=1)
        data = transform(data)
        assert data.edge_attr.numel() == 0

    # Test with cat=False
    data = _data.clone()
    transform = GaussianDistance(sigma=1, cat=False)
    data = transform(data)
    expected_edge_attr = torch.tensor([[0.2420], [0.2420]], dtype=torch.float)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)
