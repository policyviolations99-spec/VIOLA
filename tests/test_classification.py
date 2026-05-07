"""
Smoke test: instantiate TraceGNN and run a forward pass on a dummy graph.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "classification"))


class TestTraceGNN(unittest.TestCase):

    def test_forward_pass(self):
        try:
            import torch
            from torch_geometric.data import Data, Batch
        except ImportError:
            self.skipTest("torch or torch-geometric not installed")

        from src.classification.training.model import TraceGNN

        # Hyperparameters matching defaults
        num_nodes = 12
        llm_dim = 768
        non_llm_dim = 64
        hidden_dim = 64
        num_agents = 6
        num_classes = 13

        # Build a dummy heterogeneous graph
        x = torch.randn(num_nodes, max(llm_dim, non_llm_dim))
        node_type = torch.zeros(num_nodes, dtype=torch.long)
        node_type[num_nodes // 2:] = 1   # half LLM, half non-LLM
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Create a Data object the model expects
        data = Data(x=x, edge_index=edge_index, node_type=node_type, batch=batch)
        # Attach split features for heterogeneous encoder
        data.x_llm = torch.randn(num_nodes, llm_dim)
        data.x_non_llm = torch.randn(num_nodes, non_llm_dim)

        model = TraceGNN(
            llm_input_dim=llm_dim,
            non_llm_input_dim=non_llm_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            num_failure_classes=num_classes,
        )
        model.eval()

        with torch.no_grad():
            agent_logits, failure_logits = model(x=data.x_llm, edge_index=data.edge_index,
                                                  node_type=data.node_type, batch=data.batch)

        self.assertEqual(agent_logits.shape, (1, num_agents))
        self.assertEqual(failure_logits.shape, (1, num_classes))

    def test_model_parameter_count(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from src.classification.training.model import TraceGNN, count_parameters

        model = TraceGNN(
            llm_input_dim=768,
            non_llm_input_dim=64,
            hidden_dim=64,
            num_agents=6,
            num_failure_classes=13,
        )
        num_params = count_parameters(model)
        self.assertGreater(num_params, 0)
        self.assertLess(num_params, 50_000_000)  # sanity: < 50M params


class TestCheckpointRoundtrip(unittest.TestCase):
    """Save a checkpoint with torch.save and reload it via eval.py logic."""

    def test_save_and_load_checkpoint(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        import tempfile
        from src.classification.training.model import TraceGNN

        llm_dim, non_llm_dim, hidden_dim = 64, 32, 32
        num_agents, num_classes = 6, 13

        model = TraceGNN(
            llm_input_dim=llm_dim,
            non_llm_input_dim=non_llm_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            num_failure_classes=num_classes,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)

        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "llm_input_dim": llm_dim,
                    "non_llm_input_dim": non_llm_dim,
                    "hidden_dim": hidden_dim,
                    "num_agents": num_agents,
                    "num_failure_classes": num_classes,
                },
            }, ckpt_path)

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            cfg = checkpoint["model_config"]
            model2 = TraceGNN(
                llm_input_dim=cfg["llm_input_dim"],
                non_llm_input_dim=cfg["non_llm_input_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_agents=cfg["num_agents"],
                num_failure_classes=cfg["num_failure_classes"],
            )
            model2.load_state_dict(checkpoint["model_state_dict"])
            model2.eval()

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                self.assertTrue(torch.allclose(p1, p2), f"Mismatch in {n1}")
        finally:
            ckpt_path.unlink(missing_ok=True)


class TestDistorters(unittest.TestCase):

    def test_all_distorters_importable(self):
        from src.generation.distorters import VIOLATION_MAP
        self.assertGreaterEqual(len(VIOLATION_MAP), 11)

    def test_v1_apply(self):
        from src.generation.distorters import get_distorter

        distorter = get_distorter("V1")
        system_prompt = (
            "You are an API planner.\n"
            "the task must not mention any specific API names, or API response structures.\n"
            "Plan carefully."
        )
        modified_system, modified_user, change_info = distorter.apply(
            system_prompt=system_prompt,
            user_prompt="retrieve contacts",
            target_agent="APIPlannerAgent",
            params={},
        )
        self.assertNotEqual(modified_system, system_prompt)
        self.assertIn("changes", change_info)

    def test_incompatible_agent_raises(self):
        from src.generation.distorters import check_compatibility
        with self.assertRaises(ValueError):
            check_compatibility("V8", "APIPlannerAgent")


if __name__ == "__main__":
    unittest.main()
