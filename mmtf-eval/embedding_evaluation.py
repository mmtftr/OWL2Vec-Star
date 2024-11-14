"""
Embedding Evaluation Framework

This script evaluates and compares different GO term embeddings (Anc2Vec, OWL2Vec, BioBERT, GT2Vec (planned))
using various metrics and visualization techniques. Results are logged to Weights & Biases
for experiment tracking.
"""

from re import sub
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Union, Tuple
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.rich import tqdm
import logging
import torch
import gc
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import yaml
import sys
sys.path.append('..')
from owl2vec_star.lib.Evaluator import Evaluator
from rich.traceback import install
install(show_locals=True)
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [200, 100], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline."""
    project_name: str = "go-embedding-evaluation"
    base_ontology: Literal["go-basic", "go-full"] = "go-full"
    batch_size: int = 128
    learning_rate: float = 0.001
    num_epochs: int = 100
    test_size: float = 0.2
    random_seed: int = 42
    enhanced: bool = False
    input_type: Literal["concatenate", "minus"] = "concatenate"
    device: str = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'EvalConfig':
        """Load configuration from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class EmbeddingDataset:
    """Handles loading and preprocessing of different embedding types."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    def load_anc2vec(self, file_path: Path) -> None:
        """Load Anc2Vec embeddings from .npy file."""
        logger.info(f"Loading Anc2Vec embeddings from {file_path}")
        self.embeddings['anc2vec'] = np.load(file_path, allow_pickle=True).item()

        # key format is different from the other embeddings
        self.embeddings['anc2vec'] = {k.replace(":", "_"): v for k, v in self.embeddings['anc2vec'].items()}


        logger.info(f"Loaded {len(self.embeddings['anc2vec'])} Anc2Vec embeddings")

    def load_owl2vec(self, file_path: Path) -> None:
        """Load OWL2Vec embeddings."""
        logger.info(f"Loading OWL2Vec embeddings from {file_path}")
        self.embeddings['owl2vec'] = np.load(file_path, allow_pickle=True).item()
        logger.info(f"Loaded {len(self.embeddings['owl2vec'])} OWL2Vec embeddings")

    def load_biobert(self, file_path: Path) -> None:
        """Load BioBERT embeddings."""
        logger.info(f"Loading BioBERT embeddings from {file_path}")
        self.embeddings['biobert'] = np.load(file_path, allow_pickle=True).item()
        logger.info(f"Loaded {len(self.embeddings['biobert'])} BioBERT embeddings")

    def load_gt2vec(self, file_path: Path) -> None:
        """Load GT2Vec embeddings."""
        logger.info(f"Loading GT2Vec embeddings from {file_path}")
        self.embeddings['gt2vec'] = np.load(file_path, allow_pickle=True).item()
        logger.info(f"Loaded {len(self.embeddings['gt2vec'])} GT2Vec embeddings")



class SimilarityModel(nn.Module):
    """Neural network for computing GO term similarities."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x1, x2], dim=1)
        return self.projection(combined)

class EmbeddingEvaluator(Evaluator):
    """Evaluator for embedding-based GO term classification."""
    embedding_type: str

    def __init__(self, config: EvalConfig, valid_samples: pd.DataFrame, test_samples: pd.DataFrame):
        self.config = config
        self.dataset = EmbeddingDataset(config)
        self.valid_samples = valid_samples
        self.test_samples = test_samples
        self.inferred_ancestors = {}
        wandb.init(project=config.project_name, config=config.__dict__)

    def prepare_data(self, similarity_pairs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from similarity pairs."""
        train_x_list, train_y_list = [], []

        embedding_dim = next(iter(self.dataset.embeddings[self.embedding_type].values())).shape[0]

        for _, row in tqdm(similarity_pairs.iterrows(), desc="Preparing data", total=len(similarity_pairs)):
            sub, sup, label = row
            go_id = sub.split('/')[-1]
            sub_v = self.dataset.embeddings[self.embedding_type][go_id] if go_id in self.dataset.embeddings[self.embedding_type] else np.zeros(embedding_dim)

            go_id = sup.split('/')[-1]
            sup_v = self.dataset.embeddings[self.embedding_type][go_id] if go_id in self.dataset.embeddings[self.embedding_type] else np.zeros(embedding_dim)

            if not (np.all(sub_v == 0) or np.all(sup_v == 0)):
                if self.config.input_type == 'concatenate':
                    train_x_list.append(np.concatenate((sub_v, sup_v)))
                else:
                    train_x_list.append(sub_v - sup_v)
                train_y_list.append(int(label))

        train_X = np.array(train_x_list)
        train_y = np.array(train_y_list)
        logger.info(f'train_X: {train_X.shape}, train_y: {train_y.shape}')

        self.train_X = train_X
        self.train_y = train_y

    def evaluate(self, model, eva_samples: pd.DataFrame) -> Tuple[float, float, float, float]:
        """Evaluate model performance using MRR and Hits@k metrics."""
        model.verbose = False
        batch_size = 32  # Adjust based on your memory constraints

        # Pre-compute all_classes and embeddings once
        all_classes = np.array(list(self.dataset.embeddings[self.embedding_type].keys()))
        all_class_v = np.array([self.dataset.embeddings[self.embedding_type][c] for c in all_classes])
        embedding_dim = all_class_v.shape[1]

        metrics = np.zeros(4)  # [MRR, hits@1, hits@5, hits@10]
        n_samples = len(eva_samples)

        # Process in batches
        for batch_start in tqdm(range(0, n_samples, batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_samples = eva_samples.iloc[batch_start:batch_end]

            # Prepare batch inputs
            batch_subs = [s.split('/')[-1] for s in batch_samples[0]]
            batch_gts = [g.split('/')[-1] for g in batch_samples[1]]

            # Get embeddings for batch subjects
            batch_sub_v = np.array([
                self.dataset.embeddings[self.embedding_type].get(sub_id, np.zeros(embedding_dim))
                for sub_id in batch_subs
            ])

            # Prepare predictions for all pairs in batch
            if self.config.input_type == 'concatenate':
                # Shape: (batch_size, n_classes, embedding_dim * 2)
                X = np.concatenate([
                    np.repeat(batch_sub_v[:, None, :], len(all_classes), axis=1),
                    np.repeat(all_class_v[None, :, :], len(batch_sub_v), axis=0)
                ], axis=2)
            else:
                # Shape: (batch_size, n_classes, embedding_dim)
                X = np.repeat(batch_sub_v[:, None, :], len(all_classes), axis=1) - \
                    np.repeat(all_class_v[None, :, :], len(batch_sub_v), axis=0)

            # Reshape for prediction
            X_flat = X.reshape(-1, X.shape[-1])
            P = model.predict_proba(X_flat)[:, 1].reshape(len(batch_sub_v), -1)

            # Calculate rankings and metrics for batch
            for idx, (sub_id, gt_id, probs) in enumerate(zip(batch_subs, batch_gts, P)):
                # Filter out inferred ancestors
                mask = np.ones(len(all_classes), dtype=bool)
                if sub_id in self.inferred_ancestors:
                    mask[np.isin(all_classes, self.inferred_ancestors[sub_id])] = False

                # Get sorted indices of valid predictions
                valid_probs = probs[mask]
                valid_classes = all_classes[mask]
                sorted_indices = np.argsort(valid_probs)[::-1]

                # Find rank of ground truth
                gt_rank = np.where(valid_classes[sorted_indices] == gt_id)[0][0] + 1

                # Update metrics
                metrics[0] += 1.0 / gt_rank  # MRR
                metrics[1] += gt_id == valid_classes[sorted_indices[0]]  # hits@1
                metrics[2] += gt_id in valid_classes[sorted_indices[:5]]  # hits@5
                metrics[3] += gt_id in valid_classes[sorted_indices[:10]]  # hits@10

            # Log progress periodically
            if (batch_start + batch_size) % 1000 == 0:
                n = batch_start + batch_size
                logger.info(
                    f'Evaluated {n} samples - '
                    f'MRR: {metrics[0]/n:.3f}, '
                    f'Hits@1: {metrics[1]/n:.3f}, '
                    f'Hits@5: {metrics[2]/n:.3f}, '
                    f'Hits@10: {metrics[3]/n:.3f}'
                )

        # Compute final metrics
        metrics /= n_samples

        model_name = model.__class__.__name__
        # Log to wandb
        wandb.log({
            f"{self.embedding_type}/{model_name}/MRR": metrics[0],
            f"{self.embedding_type}/{model_name}/Hits@1": metrics[1],
            f"{self.embedding_type}/{model_name}/Hits@5": metrics[2],
            f"{self.embedding_type}/{model_name}/Hits@10": metrics[3]
        })

        return tuple(metrics)

    def run_torch_mlp(self):
        """Run MLP training using PyTorch"""
        # Convert data to torch tensors
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_X, self.train_y,
            test_size=0.1,
            random_state=42
        )

        X_train = torch.FloatTensor(X_train).to(self.config.device)
        y_train = torch.FloatTensor(y_train).to(self.config.device)
        X_val = torch.FloatTensor(X_val).to(self.config.device)
        y_val = torch.FloatTensor(y_val).to(self.config.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )

        # Initialize model
        input_dim = X_train.shape[1]
        model = TorchMLP(input_dim).to(self.config.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    val_loss += criterion(outputs, batch_y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            # Log metrics
            wandb.log({
                f"{self.embedding_type}/torch_mlp/train_loss": train_loss,
                f"{self.embedding_type}/torch_mlp/val_loss": val_loss,
                f"{self.embedding_type}/torch_mlp/learning_rate": optimizer.param_groups[0]['lr'],
                f"{self.embedding_type}/torch_mlp/epoch": epoch
            })

            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Load best model
        model.load_state_dict(best_state)

        # Create a sklearn-compatible wrapper for evaluation
        class ModelWrapper:
            def __init__(self, torch_model, device):
                self.model = torch_model
                self.device = device

            def predict_proba(self, X):
                self.model.eval()
                X_tensor = torch.FloatTensor(X).to(self.device)
                with torch.no_grad():
                    probs = self.model(X_tensor).cpu().numpy()
                return np.column_stack([1 - probs, probs])

        wrapper = ModelWrapper(model, self.config.device)
        MRR, hits1, hits5, hits10 = self.evaluate(model=wrapper, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' %
            (MRR, hits1, hits5, hits10))

    def run_mlp(self):
        if self.config.enhanced:
            self.run_torch_mlp()
        else:
            super().run_mlp()

def main():
    """Main execution function."""
    # Load configuration
    config = EvalConfig.from_yaml("config.yaml")

    base_path = Path(f"{config.base_ontology}")


    # Load data
    train_data = pd.read_csv(base_path / "split" / "train.csv", header=None)  # Add header=None
    valid_data = pd.read_csv(base_path / "split" / "valid.csv", header=None)
    test_data = pd.read_csv(base_path / "split" / "test.csv", header=None)

    # Initialize evaluator
    evaluator = EmbeddingEvaluator(config, valid_data, test_data)

    # Load embeddings
    # evaluates whichever ones are loaded
    # evaluator.dataset.load_anc2vec(base_path / "anc2vec" / "ontology.embeddings.npy")
    # evaluator.dataset.load_owl2vec(base_path / "owl2vec" / "ontology.embeddings.npy")
    # if (base_path / "gt2vec" / "ontology.embeddings.npy").exists():
    #     evaluator.dataset.load_gt2vec(base_path / "gt2vec" / "ontology.embeddings.npy")
    evaluator.dataset.load_biobert(base_path / "biobert" / "ontology.embeddings.npy")

    # Validate data structure
    logger.info(f"Train data shape: {train_data.shape}")

    # Load inferred ancestors
    with open(base_path / "split" / "inferred_ancestors.txt") as f:
        evaluator.inferred_ancestors = {
            line.strip().split(',')[0]: line.strip().split(',')
            for line in f.readlines()
        }

    # Run evaluation for each embedding type
    for embedding_type in evaluator.dataset.embeddings.keys():
        evaluator.embedding_type = embedding_type
        logger.info(f"\nEvaluating {embedding_type} embeddings")

        # Prepare data
        evaluator.prepare_data(train_data)

        # # Run different models
        # logger.info("\nRandom Forest:")
        # takes too long to eval
        # evaluator.run_random_forest()

        logger.info("\nMLP:")
        evaluator.run_mlp()
        # logger.info("\nLogistic Regression:")
        # evaluator.run_logistic_regression()

        # logger.info("\nSVM:")
        # takes too long to train
        # evaluator.run_svm()

        # logger.info("\nLinear SVC:")
        # evaluator.run_linear_svc()

        # logger.info("\nDecision Tree:")
        # evaluator.run_decision_tree()

        # logger.info("\nSGD Logistic:")
        # evaluator.run_sgd_log()
        gc.collect()

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
