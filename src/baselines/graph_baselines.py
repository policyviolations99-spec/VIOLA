"""
Baseline models for comparison against TraceGNN.

Implements four baselines that share the same two-head interface as TraceGNN:
  forward(x, edge_index, node_type, batch) -> (agent_logits, failure_logits)

Models:
  LinearBaseline   — logistic regression on mean-pooled features (no hidden layers)
  MLPBaseline      — mean-pools node features, ignores graph structure entirely
  GCNBaseline      — standard Graph Convolutional Network (Kipf & Welling 2017)
  GraphSAGEBaseline — inductive representation learning via neighbor sampling

Note on LinearBaseline vs. the old span-level LR:
  The previous logistic regression operated on the single violation span
  and achieved ~100% accuracy because it had direct access to the modified
  system prompt.  This version mean-pools ALL node features across the full
  trace graph — the same view every other baseline gets — making it a fair
  linear lower bound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool,
)
from typing import Optional, Literal, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared classification head (agent + failure)
# ─────────────────────────────────────────────────────────────────────────────

class _DualHead(nn.Module):
    def __init__(self, in_dim: int, num_agents: int, num_failure_classes: int,
                 dropout: float = 0.15):
        super().__init__()
        trunk_dim = in_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.agent_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim // 2, num_agents),
        )
        self.failure_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim // 2, num_failure_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, graph_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.trunk(graph_emb)
        return self.agent_head(t), self.failure_head(t)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Linear baseline  (logistic regression — graph-level, no hidden layers)
# ─────────────────────────────────────────────────────────────────────────────

class LinearBaseline(nn.Module):
    """
    Logistic regression on mean-pooled node features.

    No hidden layers, no non-linearities — a purely linear map from the
    mean-pooled trace representation to class logits.  Trained via
    cross-entropy + L2 regularisation (weight_decay in the optimizer).

    Answers: "Is a non-linear model necessary, or does the signal lie in a
    linearly-separable subspace of the pooled feature vector?"
    """

    def __init__(
        self,
        input_dim: int,
        num_agents: int = 2,
        num_failure_classes: int = 5,
        **_ignored,
    ):
        super().__init__()
        self.agent_head   = nn.Linear(input_dim, num_agents)
        self.failure_head = nn.Linear(input_dim, num_failure_classes)
        nn.init.xavier_uniform_(self.agent_head.weight)
        nn.init.xavier_uniform_(self.failure_head.weight)
        nn.init.zeros_(self.agent_head.bias)
        nn.init.zeros_(self.failure_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = global_mean_pool(x, batch)
        return self.agent_head(graph_emb), self.failure_head(graph_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MLP on pooled features  (ablation: does structure matter at all?)
# ─────────────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Multi-layer perceptron on mean-pooled node features.

    Discards all graph edges and topology.  If this performs comparably to
    the GNN the graph structure adds no signal; if it underperforms, graph
    structure is necessary.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.15,
        num_agents: int = 2,
        num_failure_classes: int = 5,
        **_ignored,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.heads = _DualHead(hidden_dim, num_agents, num_failure_classes, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = global_mean_pool(x, batch)
        h = self.mlp(graph_emb)
        return self.heads(h)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GCN baseline
# ─────────────────────────────────────────────────────────────────────────────

class GCNBaseline(nn.Module):
    """
    Standard Graph Convolutional Network (Kipf & Welling, ICLR 2017).

    Uses GCNConv layers with residual connections and layer norm,
    matching the depth / width of the primary GNN for a fair comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.15,
        pool_type: Literal['mean', 'max', 'add'] = 'mean',
        num_agents: int = 2,
        num_failure_classes: int = 5,
        **_ignored,
    ):
        super().__init__()
        self.dropout = dropout
        self.pool_type = pool_type

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.heads = _DualHead(hidden_dim, num_agents, num_failure_classes, dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'mean':
            return global_mean_pool(h, batch)
        elif self.pool_type == 'max':
            return global_max_pool(h, batch)
        return global_add_pool(h, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h = norm(F.relu(h_new) + h)          # residual + LN
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = self._pool(h, batch)
        return self.heads(graph_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GraphSAGE baseline
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGEBaseline(nn.Module):
    """
    Inductive representation learning (Hamilton et al., NeurIPS 2017).

    Uses SAGEConv (mean aggregation) with residual connections and layer norm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.15,
        pool_type: Literal['mean', 'max', 'add'] = 'mean',
        num_agents: int = 2,
        num_failure_classes: int = 5,
        **_ignored,
    ):
        super().__init__()
        self.dropout = dropout
        self.pool_type = pool_type

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, aggr='mean', normalize=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.heads = _DualHead(hidden_dim, num_agents, num_failure_classes, dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'mean':
            return global_mean_pool(h, batch)
        elif self.pool_type == 'max':
            return global_max_pool(h, batch)
        return global_add_pool(h, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h = norm(F.relu(h_new) + h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = self._pool(h, batch)
        return self.heads(graph_emb)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'linear':    LinearBaseline,
    'mlp':       MLPBaseline,
    'gcn':       GCNBaseline,
    'graphsage': GraphSAGEBaseline,
}


def build_baseline(
    name: str,
    input_dim: int,
    num_agents: int,
    num_failure_classes: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.15,
    **kwargs,
) -> nn.Module:
    """Instantiate a baseline model by name."""
    cls = MODEL_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown baseline: {name!r}. Choose from {list(MODEL_REGISTRY)}")
    return cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_agents=num_agents,
        num_failure_classes=num_failure_classes,
        **kwargs,
    )
