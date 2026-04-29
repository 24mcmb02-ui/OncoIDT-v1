"""
Graph-Aware Transformer fusion layer for OncoIDT.

Cross-attention between the patient's Neural CDE hidden state (query) and
GNN neighbor embeddings (keys/values), with a learned structural bias term
per edge type added to the attention logits before softmax.

The output is concatenated with the CDE hidden state to form the final
shared representation fed into the task heads.

Requirements: 6.2
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .gnn import NUM_EDGE_TYPES


class GraphAwareTransformer(nn.Module):
    """Cross-attention fusion of CDE hidden state with GNN neighbor embeddings.

    Architecture:
        Q = Linear(cde_dim, d_model)  from patient CDE hidden state
        K = Linear(gnn_dim, d_model)  from GNN neighbor embeddings
        V = Linear(gnn_dim, d_model)  from GNN neighbor embeddings
        Structural bias = EdgeTypeEmbedding(num_edge_types, num_heads)
                          added to attention logits before softmax
        Output = Concat(cross_attn_output, cde_hidden) → Linear → repr

    Args:
        cde_dim:        Dimension of the CDE hidden state (default 256).
        gnn_dim:        Dimension of GNN node embeddings (default 256).
        d_model:        Internal attention dimension (default 256).
        num_heads:      Number of attention heads (default 8).
        num_edge_types: Number of distinct edge types for structural bias.
        out_dim:        Output shared representation dimension (default 256).
    """

    def __init__(
        self,
        cde_dim: int = 256,
        gnn_dim: int = 256,
        d_model: int = 256,
        num_heads: int = 8,
        num_edge_types: int = NUM_EDGE_TYPES,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections
        self.q_proj = nn.Linear(cde_dim, d_model)
        self.k_proj = nn.Linear(gnn_dim, d_model)
        self.v_proj = nn.Linear(gnn_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Structural bias: one scalar bias per (head, edge_type)
        # Shape: (num_edge_types, num_heads)
        self.structural_bias = nn.Embedding(num_edge_types, num_heads)

        # Final projection: concat(cross_attn, cde_hidden) → out_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + cde_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        cde_hidden: torch.Tensor,
        neighbor_embeddings: torch.Tensor,
        neighbor_edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            cde_hidden:          Patient CDE hidden state, shape (batch, cde_dim).
            neighbor_embeddings: GNN embeddings of k-hop neighbors,
                                 shape (batch, num_neighbors, gnn_dim).
                                 If a patient has no neighbors, pass a zero
                                 tensor of shape (batch, 1, gnn_dim).
            neighbor_edge_types: Integer edge type for each neighbor,
                                 shape (batch, num_neighbors).
                                 If None, no structural bias is applied.

        Returns:
            Fused shared representation, shape (batch, out_dim).
        """
        batch, num_neighbors, _ = neighbor_embeddings.shape

        # Project Q from CDE hidden state: (batch, 1, d_model)
        Q = self.q_proj(cde_hidden).unsqueeze(1)

        # Project K, V from neighbor embeddings: (batch, num_neighbors, d_model)
        K = self.k_proj(neighbor_embeddings)
        V = self.v_proj(neighbor_embeddings)

        # Reshape for multi-head attention
        # (batch, num_heads, seq, head_dim)
        Q = Q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, num_neighbors, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, num_neighbors, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention logits: (batch, num_heads, 1, num_neighbors)
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Add structural bias if edge types are provided
        if neighbor_edge_types is not None:
            # bias: (batch, num_neighbors, num_heads) → (batch, num_heads, 1, num_neighbors)
            bias = self.structural_bias(neighbor_edge_types)
            bias = bias.permute(0, 2, 1).unsqueeze(2)
            attn_logits = attn_logits + bias

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Weighted sum of values: (batch, num_heads, 1, head_dim)
        attn_out = torch.matmul(attn_weights, V)

        # Reshape back: (batch, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Concatenate with original CDE hidden state and project
        fused = torch.cat([attn_out, cde_hidden], dim=-1)   # (batch, d_model + cde_dim)
        return self.fusion_proj(fused)                        # (batch, out_dim)
