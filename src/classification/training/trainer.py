"""
Multi-task training loop for root-cause analysis GNN.

Predicts:
  - which agent was injected with a failure  (CrossEntropy, Head 1)
  - which failure type was injected           (CrossEntropy, Head 2)

Combined loss:  L = L_agent + lambda_failure * L_failure

History tracks all three losses (agent / failure / combined) for train and val
separately so they can be plotted / analysed afterwards.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch as PyGBatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
class MultiTaskMetricsTracker:
    """Accumulate predictions and compute per-head metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_preds:   List[int] = []
        self.agent_labels:  List[int] = []
        self.failure_preds: List[int] = []
        self.failure_labels: List[int] = []

    def update(
        self,
        agent_logits:   torch.Tensor,   # [B, num_agents]
        failure_logits: torch.Tensor,   # [B, num_failure_classes]
        y_agent:        torch.Tensor,   # [B]
        y_failure:      torch.Tensor,   # [B]
    ):
        self.agent_preds.extend(agent_logits.argmax(dim=-1).cpu().tolist())
        self.agent_labels.extend(y_agent.cpu().tolist())
        self.failure_preds.extend(failure_logits.argmax(dim=-1).cpu().tolist())
        self.failure_labels.extend(y_failure.cpu().tolist())

    def compute_metrics(self) -> Dict[str, float]:
        a_pred  = np.array(self.agent_preds)
        a_true  = np.array(self.agent_labels)
        f_pred  = np.array(self.failure_preds)
        f_true  = np.array(self.failure_labels)

        avg = 'macro'
        return {
            'agent_accuracy':   accuracy_score(a_true, a_pred),
            'agent_f1':         f1_score(a_true, a_pred, average=avg, zero_division=0),
            'agent_precision':  precision_score(a_true, a_pred, average=avg, zero_division=0),
            'agent_recall':     recall_score(a_true, a_pred, average=avg, zero_division=0),
            'failure_accuracy': accuracy_score(f_true, f_pred),
            'failure_f1':       f1_score(f_true, f_pred, average=avg, zero_division=0),
            'failure_precision':precision_score(f_true, f_pred, average=avg, zero_division=0),
            'failure_recall':   recall_score(f_true, f_pred, average=avg, zero_division=0),
        }

    def get_confusion_matrices(self):
        return {
            'agent':   confusion_matrix(self.agent_labels,   self.agent_preds),
            'failure': confusion_matrix(self.failure_labels, self.failure_preds),
        }


# ──────────────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Multi-task trainer for GNN root-cause analysis.
    Tracks loss_agent, loss_failure, and loss_combined separately.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.config = config
        self.device = device

        self.lambda_failure: float = config.training.lambda_failure

        self.optimizer  = self._create_optimizer()
        self.scheduler  = self._create_scheduler()
        self.criterion  = nn.CrossEntropyLoss()          # same for both heads

        self.train_metrics = MultiTaskMetricsTracker()
        self.val_metrics   = MultiTaskMetricsTracker()

        self.writer = self._create_writer()

        # Training state
        self.epoch               = 0
        self.global_step         = 0
        self.best_val_metric     = 0.0
        self.best_epoch          = 0
        self.early_stopping_counter = 0

        # History — three losses per split
        self.history = {
            'train_loss':         [],
            'train_loss_agent':   [],
            'train_loss_failure': [],
            'val_loss':           [],
            'val_loss_agent':     [],
            'val_loss_failure':   [],
            'train_metrics':      [],
            'val_metrics':        [],
            'learning_rates':     [],
        }

    # ── optimiser / scheduler / writer ──────────────────────────────────────

    def _create_optimizer(self):
        opt = self.config.training
        if opt.optimizer == 'adam':
            return optim.Adam(self.model.parameters(),
                              lr=opt.learning_rate, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamw':
            return optim.AdamW(self.model.parameters(),
                               lr=opt.learning_rate, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(),
                             lr=opt.learning_rate, momentum=opt.momentum,
                             weight_decay=opt.weight_decay)
        raise ValueError(f"Unknown optimizer: {opt.optimizer}")

    def _create_scheduler(self):
        if not self.config.training.use_scheduler:
            return None
        sched = self.config.training.scheduler_type
        if sched == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                min_lr=self.config.training.scheduler_min_lr,
            )
        elif sched == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs,
                eta_min=self.config.training.scheduler_min_lr,
            )
        elif sched == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30,
                gamma=self.config.training.scheduler_factor,
            )
        raise ValueError(f"Unknown scheduler: {sched}")

    def _create_writer(self):
        if not self.config.experiment.use_tensorboard:
            return None
        run_name = self.config.experiment.run_name or (
            f"{self.config.experiment.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return SummaryWriter(self.config.experiment.log_dir / run_name)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _compute_losses(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass → (loss_combined, loss_agent, loss_failure)."""
        agent_logits, failure_logits = self.model(
            batch.x, batch.edge_index, batch.node_type, batch.batch
        )
        loss_agent   = self.criterion(agent_logits,   batch.y_agent)
        loss_failure = self.criterion(failure_logits, batch.y_failure)
        loss_combined = loss_agent + self.lambda_failure * loss_failure
        return loss_combined, loss_agent, loss_failure, agent_logits, failure_logits

    # ── train epoch ─────────────────────────────────────────────────────────

    def train_epoch(self) -> Tuple[float, float, float, Dict[str, float]]:
        """Train one epoch. Returns (loss, loss_agent, loss_failure, metrics)."""
        self.model.train()
        self.train_metrics.reset()

        total, total_a, total_f = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]', leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            loss, loss_a, loss_f, agent_logits, failure_logits = \
                self._compute_losses(batch)

            loss.backward()

            if self.config.training.grad_clip_norm and self.config.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )

            self.optimizer.step()

            self.train_metrics.update(agent_logits, failure_logits,
                                      batch.y_agent, batch.y_failure)
            total   += loss.item()
            total_a += loss_a.item()
            total_f += loss_f.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'a': f'{loss_a.item():.4f}',
                'f': f'{loss_f.item():.4f}',
            })

            if self.writer and self.global_step % self.config.training.log_every_n_steps == 0:
                self.writer.add_scalar('train/batch_loss',         loss.item(),   self.global_step)
                self.writer.add_scalar('train/batch_loss_agent',   loss_a.item(), self.global_step)
                self.writer.add_scalar('train/batch_loss_failure', loss_f.item(), self.global_step)

        n = max(num_batches, 1)
        metrics = self.train_metrics.compute_metrics()
        return total / n, total_a / n, total_f / n, metrics

    # ── validate ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, Dict[str, float]]:
        """Validate. Returns (loss, loss_agent, loss_failure, metrics)."""
        self.model.eval()
        self.val_metrics.reset()

        total, total_a, total_f = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]', leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            loss, loss_a, loss_f, agent_logits, failure_logits = \
                self._compute_losses(batch)

            self.val_metrics.update(agent_logits, failure_logits,
                                    batch.y_agent, batch.y_failure)
            total   += loss.item()
            total_a += loss_a.item()
            total_f += loss_f.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        n = max(num_batches, 1)
        metrics = self.val_metrics.compute_metrics()
        return total / n, total_a / n, total_f / n, metrics

    # ── main training loop ───────────────────────────────────────────────────

    def train(self):
        """Full training loop."""
        logger.info("Starting multi-task training...")
        logger.info(f"  Device:          {self.device}")
        logger.info(f"  Epochs:          {self.config.training.num_epochs}")
        logger.info(f"  lambda_failure:  {self.lambda_failure}")

        for epoch in range(1, self.config.training.num_epochs + 1):
            self.epoch = epoch

            train_loss, train_loss_a, train_loss_f, train_metrics = self.train_epoch()
            val_loss,   val_loss_a,   val_loss_f,   val_metrics   = self.validate()

            # LR scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # ── history ──────────────────────────────────────────────────────
            self.history['train_loss'].append(train_loss)
            self.history['train_loss_agent'].append(train_loss_a)
            self.history['train_loss_failure'].append(train_loss_f)
            self.history['val_loss'].append(val_loss)
            self.history['val_loss_agent'].append(val_loss_a)
            self.history['val_loss_failure'].append(val_loss_f)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)

            # ── TensorBoard ───────────────────────────────────────────────────
            if self.writer:
                self.writer.add_scalar('train/loss',         train_loss,   epoch)
                self.writer.add_scalar('train/loss_agent',   train_loss_a, epoch)
                self.writer.add_scalar('train/loss_failure', train_loss_f, epoch)
                self.writer.add_scalar('val/loss',           val_loss,     epoch)
                self.writer.add_scalar('val/loss_agent',     val_loss_a,   epoch)
                self.writer.add_scalar('val/loss_failure',   val_loss_f,   epoch)
                self.writer.add_scalar('learning_rate',      current_lr,   epoch)
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

            # ── logging ───────────────────────────────────────────────────────
            logger.info(f"\nEpoch {epoch}/{self.config.training.num_epochs}")
            logger.info(
                f"  Loss  combined: train={train_loss:.4f}  val={val_loss:.4f}"
            )
            logger.info(
                f"  Loss  agent:    train={train_loss_a:.4f}  val={val_loss_a:.4f}"
            )
            logger.info(
                f"  Loss  failure:  train={train_loss_f:.4f}  val={val_loss_f:.4f}"
            )
            logger.info(
                f"  Agent   F1: train={train_metrics['agent_f1']:.4f}  "
                f"val={val_metrics['agent_f1']:.4f}"
            )
            logger.info(
                f"  Failure F1: train={train_metrics['failure_f1']:.4f}  "
                f"val={val_metrics['failure_f1']:.4f}"
            )
            logger.info(f"  LR: {current_lr:.6f}")

            # ── early stopping on average F1 across both heads ────────────────
            current_metric = (
                val_metrics['agent_f1'] + val_metrics['failure_f1']
            ) / 2.0

            if current_metric > self.best_val_metric + self.config.training.early_stopping_min_delta:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                self.save_checkpoint(is_best=True)
                logger.info(f"  ✓ New best (avg F1={current_metric:.4f})")
            else:
                self.early_stopping_counter += 1

            if epoch % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(is_best=False)

            if (self.config.training.early_stopping and
                    self.early_stopping_counter >= self.config.training.patience):
                logger.info(
                    f"\nEarly stopping after epoch {epoch} "
                    f"(best epoch {self.best_epoch}, avg F1={self.best_val_metric:.4f})"
                )
                break

        if self.writer:
            self.writer.close()

        logger.info("\nTraining complete!")
        return self.history

    # ── test ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self) -> Dict:
        """Evaluate on test set using best model weights."""
        logger.info("Running final evaluation on test set...")
        self.model.eval()
        tracker = MultiTaskMetricsTracker()

        total, total_a, total_f = 0.0, 0.0, 0.0
        num_batches = 0

        for batch in tqdm(self.test_loader, desc='Testing'):
            batch = batch.to(self.device)
            loss, loss_a, loss_f, agent_logits, failure_logits = \
                self._compute_losses(batch)

            tracker.update(agent_logits, failure_logits,
                           batch.y_agent, batch.y_failure)
            total   += loss.item()
            total_a += loss_a.item()
            total_f += loss_f.item()
            num_batches += 1

        n = max(num_batches, 1)
        metrics = tracker.compute_metrics()
        metrics['loss']         = total   / n
        metrics['loss_agent']   = total_a / n
        metrics['loss_failure'] = total_f / n
        cms = tracker.get_confusion_matrices()
        metrics['confusion_matrix_agent']   = cms['agent']
        metrics['confusion_matrix_failure'] = cms['failure']

        logger.info("\n" + "=" * 80)
        logger.info("TEST SET RESULTS")
        logger.info("=" * 80)
        logger.info(f"Loss (combined): {metrics['loss']:.4f}")
        logger.info(f"Loss (agent):    {metrics['loss_agent']:.4f}")
        logger.info(f"Loss (failure):  {metrics['loss_failure']:.4f}")
        logger.info(f"Agent   — Acc={metrics['agent_accuracy']:.4f}  "
                    f"F1={metrics['agent_f1']:.4f}  "
                    f"P={metrics['agent_precision']:.4f}  "
                    f"R={metrics['agent_recall']:.4f}")
        logger.info(f"Failure — Acc={metrics['failure_accuracy']:.4f}  "
                    f"F1={metrics['failure_f1']:.4f}  "
                    f"P={metrics['failure_precision']:.4f}  "
                    f"R={metrics['failure_recall']:.4f}")
        logger.info(f"Agent CM:\n{cms['agent']}")
        logger.info(f"Failure CM:\n{cms['failure']}")
        logger.info("=" * 80 + "\n")

        return metrics

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch':               self.epoch,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict':self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric':     self.best_val_metric,
            'history':             self.history,
            'config':              self.config,
        }

        path = (self.config.experiment.checkpoint_dir /
                f"checkpoint_epoch_{self.epoch}.pt")
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.config.experiment.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best model to {best_path}")

        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        files = sorted(
            self.config.experiment.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        keep = self.config.training.keep_last_n_checkpoints
        for old in files[:-keep]:
            old.unlink()

    def load_checkpoint(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt['scheduler_state_dict']:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch           = ckpt['epoch']
        self.best_val_metric = ckpt['best_val_metric']
        self.history         = ckpt['history']
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
