"""Train a DenseNet121 classifier for one task and variant.

Examples (from the repository root, with .venv active):
    python scripts/train_classifier.py                                  # M0, celeba_smiling
    python scripts/train_classifier.py task=celeba_young
    python scripts/train_classifier.py task=chestmnist_effusion epochs=10   # setup: ~10 epochs
    python scripts/train_classifier.py -m task=celeba_smiling,celeba_young,chestmnist_effusion
    python scripts/train_classifier.py task=celeba_young variant=seed seed=1          # E2
    python scripts/train_classifier.py task=celeba_young variant=subsample seed=1     # E2

    # quick smoke run
    python scripts/train_classifier.py train.max_epochs=1 train.limit_train_batches=20 \
        train.limit_val_batches=10 train.limit_test_batches=10 train.overwrite=true

Output: outputs/classifiers/{dataset}/{target}/{variant}/ (see robust_gcf/training/train.py).
"""
import hydra
from omegaconf import DictConfig

from robust_gcf.training.train import train


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
