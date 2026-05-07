"""
Graph Neural Network model for execution trace analysis.

Two-node-type architecture (LLM calls and non-LLM calls) with
graph-level multi-task classification:
  - Head 1: which agent was injected (binary or N-class)
  - Head 2: which failure type was injected (N-class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Literal, Tuple
import logging

logger = logging.getLogger(__name__)


class NodeTypeEncoder(nn.Module):
    """
    Encode nodes based on their type with separate transformation layers.
    Maps both node types to common hidden dimension.
    """
    
    def __init__(
        self,
        llm_input_dim: int,
        non_llm_input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Separate encoders for each node type
        self.llm_encoder = nn.Sequential(
            nn.Linear(llm_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.non_llm_encoder = nn.Sequential(
            nn.Linear(non_llm_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes based on type.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            node_type: Node type indicators [num_nodes] (0=LLM, 1+=non-LLM)
            
        Returns:
            Encoded features [num_nodes, hidden_dim]
        """
        # Create mask for each node type
        llm_mask = (node_type == 0)
        non_llm_mask = (node_type != 0)
        
        # Initialize output
        h = torch.zeros(x.size(0), self.llm_encoder[-1].out_features, 
                       device=x.device, dtype=x.dtype)
        
        # Encode LLM nodes
        if llm_mask.any():
            h[llm_mask] = self.llm_encoder(x[llm_mask])
        
        # Encode non-LLM nodes
        if non_llm_mask.any():
            h[non_llm_mask] = self.non_llm_encoder(x[non_llm_mask])
        
        return h


class GATLayer(nn.Module):
    """
    Graph Attention Layer with optional residual connection and normalization.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_dim == out_dim)
        self.use_layer_norm = use_layer_norm
        
        # GAT convolution
        self.conv = GATConv(
            in_dim,
            out_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True
        )
        
        # Normalization
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        
        # Residual projection if dimensions don't match
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        identity = x
        
        # Apply GAT
        x = self.conv(x, edge_index)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity
        
        # Layer normalization
        if self.use_layer_norm:
            x = self.norm(x)
        
        return x


class AttentionPooling(nn.Module):
    """
    Attention-based graph-level pooling.
    Learns importance weights for each node.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted pooling.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [num_nodes, 1]
        
        # Apply softmax per graph
        attn_weights = torch.exp(attn_weights)
        
        # Normalize by graph
        batch_size = batch.max().item() + 1
        denominator = torch.zeros(batch_size, device=x.device)
        denominator.scatter_add_(0, batch, attn_weights.squeeze(-1))
        denominator = denominator[batch].unsqueeze(-1)
        
        attn_weights = attn_weights / (denominator + 1e-8)
        
        # Weighted sum
        weighted_x = x * attn_weights
        
        # Pool per graph
        graph_features = torch.zeros(batch_size, x.size(1), device=x.device)
        graph_features.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted_x), weighted_x)
        
        return graph_features


class TraceGNN(nn.Module):
    """
    GNN for execution trace analysis with multi-task classification.

    Architecture:
    1. Node type-specific encoding
    2. Multi-layer GAT message passing
    3. Graph-level pooling
    4. Two classification heads:
         - agent_head:   predicts which agent was injected (num_agents classes)
         - failure_head: predicts which failure type was injected (num_failure_classes)
    """

    def __init__(
        self,
        llm_feature_dim: int,
        non_llm_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.15,
        pool_type: Literal['mean', 'max', 'add', 'attention'] = 'mean',
        use_residual: bool = True,
        use_layer_norm: bool = True,
        num_agents: int = 2,
        num_failure_classes: int = 5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool_type = pool_type

        # Node type encoding
        self.node_encoder = NodeTypeEncoder(
            llm_feature_dim,
            non_llm_feature_dim,
            hidden_dim,
            dropout=dropout
        )

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm
            )
            for _ in range(num_layers)
        ])

        # Graph-level pooling
        if pool_type == 'attention':
            self.pooling = AttentionPooling(hidden_dim)
        # else: use functional pooling (mean/max/add)

        # Shared trunk before both heads
        self._trunk_dim = hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, self._trunk_dim),
            nn.LayerNorm(self._trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1: agent identification
        self.agent_head = nn.Sequential(
            nn.Linear(self._trunk_dim, self._trunk_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self._trunk_dim // 2, num_agents),
        )

        # Head 2: failure type
        self.failure_head = nn.Sequential(
            nn.Linear(self._trunk_dim, self._trunk_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self._trunk_dim // 2, num_failure_classes),
        )

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:          Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            node_type:  Node type indicators [num_nodes] (0=LLM, 1+=non-LLM)
            batch:      Batch assignment [num_nodes]

        Returns:
            Tuple of:
              - agent_logits:   [batch_size, num_agents]
              - failure_logits: [batch_size, num_failure_classes]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. Node type encoding
        h = self.node_encoder(x, node_type)

        # 2. Message passing
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index)

        # 3. Graph-level pooling
        if self.pool_type == 'mean':
            graph_emb = global_mean_pool(h, batch)
        elif self.pool_type == 'max':
            graph_emb = global_max_pool(h, batch)
        elif self.pool_type == 'add':
            graph_emb = global_add_pool(h, batch)
        elif self.pool_type == 'attention':
            graph_emb = self.pooling(h, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")

        # 4. Shared trunk + two heads
        trunk = self.trunk(graph_emb)
        agent_logits   = self.agent_head(trunk)    # [B, num_agents]
        failure_logits = self.failure_head(trunk)  # [B, num_failure_classes]

        return agent_logits, failure_logits
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract graph-level embeddings (after pooling, before heads)."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x, node_type)
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index)

        if self.pool_type == 'mean':
            return global_mean_pool(h, batch)
        elif self.pool_type == 'max':
            return global_max_pool(h, batch)
        elif self.pool_type == 'add':
            return global_add_pool(h, batch)
        else:
            return self.pooling(h, batch)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: TraceGNN) -> None:
    """Print model architecture summary."""
    print("\n" + "=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Number of GAT layers: {model.num_layers}")
    print(f"Pooling type: {model.pool_type}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print("=" * 80 + "\n")
