import pytest
import torch
from torch_geometric.data import Data

from lesion_gnn.transforms import GaussianDistance, SaveAs


def test_gaussian_distance():
    # Create a simple graph with two nodes and one edge
    edge_index = torch.tensor([[0, 1], [1, 0]])
    pos = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
    edge_weight = torch.tensor([1, 1], dtype=torch.float)
    edge_attr = torch.tensor([1, 1], dtype=torch.float)
    _data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight)

    # Test with sigma=1
    data = _data.clone()
    transform = GaussianDistance(sigma=1, save_as=SaveAs.EDGE_WEIGHT_REPLACE)
    data = transform(data)
    expected_edge_weight = torch.tensor([0.2420, 0.2420], dtype=torch.float)
    expected_edge_attr = torch.tensor([1, 1], dtype=torch.float)
    torch.testing.assert_close(data.edge_weight, expected_edge_weight, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with sigma=0.5
    data = _data.clone()
    transform = GaussianDistance(sigma=0.5, save_as=SaveAs.EDGE_WEIGHT_REPLACE)
    data = transform(data)
    expected_edge_weight = torch.tensor([0.1080, 0.1080], dtype=torch.float)
    expected_edge_attr = torch.tensor([1, 1], dtype=torch.float)
    torch.testing.assert_close(data.edge_weight, expected_edge_weight, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with sigma=2
    data = _data.clone()
    transform = GaussianDistance(sigma=2, save_as=SaveAs.EDGE_WEIGHT_REPLACE)
    data = transform(data)
    expected_edge_weight = torch.tensor([0.1760, 0.1760], dtype=torch.float)
    expected_edge_attr = torch.tensor([1, 1], dtype=torch.float)
    torch.testing.assert_close(data.edge_weight, expected_edge_weight, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with a graph with no edges
    with pytest.warns(UserWarning):
        edge_index = torch.tensor([], dtype=torch.long)
        pos = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
        edge_weight = torch.tensor([], dtype=torch.float)
        edge_attr = torch.tensor([], dtype=torch.float)
        data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight)
        transform = GaussianDistance(sigma=1, save_as=SaveAs.EDGE_WEIGHT_REPLACE)
        data = transform(data)
        assert data.edge_weight.numel() == 0

    # Test with a graph with one node
    with pytest.warns(UserWarning):
        edge_index = torch.tensor([], dtype=torch.long)
        pos = torch.tensor([[0, 0]], dtype=torch.float)
        edge_attr = torch.tensor([], dtype=torch.float)
        edge_weight = torch.tensor([], dtype=torch.float)
        data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight)
        transform = GaussianDistance(sigma=1, save_as=SaveAs.EDGE_WEIGHT_REPLACE)
        data = transform(data)
        assert data.edge_weight.numel() == 0

    # Test with edge_attr_cat
    data = _data.clone()
    transform = GaussianDistance(sigma=1, save_as=SaveAs.EDGE_ATTR_CAT)
    data = transform(data)
    expected_edge_attr = torch.tensor([[1, 0.2420], [1, 0.2420]], dtype=torch.float)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)

    # Test with edge_attr_replace
    data = _data.clone()
    transform = GaussianDistance(sigma=1, save_as=SaveAs.EDGE_ATTR_REPLACE)
    data = transform(data)
    expected_edge_attr = torch.tensor([[0.2420], [0.2420]], dtype=torch.float)
    torch.testing.assert_close(data.edge_attr, expected_edge_attr, rtol=1e-3, atol=1e-3)
