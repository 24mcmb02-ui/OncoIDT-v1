"""
GraphSAGE GNN for OncoIDT patient-ward knowledge graph.

2-layer GraphSAGE with mean aggregation over heterogeneous edge types.
Produces node embeddings of shape (N, 256) for use in the graph-aware
transformer fusion layer.

Requirements: 6.2, 7.5
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False


# Edge type registry — must match the graph service edge types
EDGE_TYPES = [
    "OCCUPIES",
    "TREATED_BY",
    "CO_LOCATED",
    "EXPOSED_TO",
    "TRIGGERED",
    "RESULTED_IN",
]
NUM_EDGE_TYPES = len(EDGE_TYPES)


class HeteroSAGELayer(nn.Module):
    """Single GraphSAGE layer with per-edge-type weight matrices.

    For each edge type e, a separate SAGEConv is applied and the results
    are summed, then a learned edge-type embedding is added to the logits
    before the final activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = NUM_EDGE_TYPES,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types

        # One SAGEConv per edge type
        self.convs = nn.ModuleList(
            [SAGEConv(in_channels, out_channels) for _ in range(num_edge_types)]
        )

        # Learned edge-type embedding added to aggregated logits
        self.edge_type_emb = nn.Embedding(num_edge_types, out_channels)

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features, shape (N, in_channels).
            edge_index: COO edge indices, shape (2, E).
            edge_type:  Integer edge type per edge, shape (E,).

        Returns:
            Updated node embeddings, shape (N, out_channels).
        """
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)

        for etype_idx in range(self.num_edge_types):
            mask = edge_type == etype_idx
            if mask.sum() == 0:
                continue
            ei_sub = edge_index[:, mask]   # edges of this type only
            # SAGEConv aggregation for this edge type
            h = self.convs[etype_idx](x, ei_sub)   # (N, out_channels)
            # Add learned edge-type bias (broadcast over all nodes)
            h = h + self.edge_type_emb(
                torch.tensor(etype_idx, device=x.device)
            )
            out = out + h

        return self.act(self.norm(out))


class GraphSAGE(nn.Module):
    """2-layer heterogeneous GraphSAGE producing 256-dim node embeddings.

    Args:
        in_channels:    Input node feature dimension.
        hidden_channels: Hidden dimension (default 256).
        num_edge_types:  Number of distinct edge types (default 6).

    Forward:
        x          (N, in_channels)  — node feature matrix
        edge_index (2, E)            — COO edge indices
        edge_type  (E,)              — integer edge type per edge

    Returns:
        Node embeddings of shape (N, 256).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_edge_types: int = NUM_EDGE_TYPES,
    ) -> None:
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GraphSAGE. "
                "Install with: pip install torch-geometric"
            )

        self.layer1 = HeteroSAGELayer(in_channels, hidden_channels, num_edge_types)
        self.layer2 = HeteroSAGELayer(hidden_channels, hidden_channels, num_edge_types)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = self.layer1(x, edge_index, edge_type)   # (N, 256)
        h = self.layer2(h, edge_index, edge_type)   # (N, 256)
        return h
