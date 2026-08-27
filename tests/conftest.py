"""
Shared fixtures for testing src/trainer.py.

"""
import sys
import types
from unittest.mock import MagicMock

import pytest
from sklearn.datasets import make_classification

def _stub_module(name):
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    mod.save = MagicMock()
    mod.load = MagicMock()
    sys.modules[name] = mod


_stub_module("torch")


@pytest.fixture
def sample_data():
    """A tiny, fast, synthetic binary-classification dataset."""
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X_train, y_train = X[:40], y[:40]
    X_val, y_val = X[40:], y[40:]
    return X_train, y_train, X_val, y_val


@pytest.fixture
def base_trainer_config():
    """
    A trainer config shaped like configs/trainer_config.yml, but with
    tiny/fast hyperparameters so real .fit() calls stay quick in tests.
    """
    return {
        "trainer": {
            "Random_forest": {
                "train": True,
                "Randomized_Search": False,
                "parameters": {
                    "n_estimators": 5,
                    "min_samples_leaf": 2,
                    "min_samples_split": 2,
                },
            },
            "KNN": {
                "train": False,
                "grid_search": False,
                "parameters": {"n_neighbors": 3},
            },
            "Logistic_Regression": {
                "train": True,
                "grid_search": False,
                "parameters": {"C": 1.0, "max_iter": 200},
            },
            "Neural_Network": {
                "train": True,
                "Randomized_Search": False,
                "parameters": {
                    "hidden_layer_sizes": "(4, 2)",
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.01,
                    "batch_size": 8,
                    "learning_rate_init": 0.01,
                    "max_iter": 50,
                },
            },
            "Voting_Classifier": {
                "parameters": {
                    "voting": "soft",
                    "weights": [0.3, 0.3, 0.4],
                    "fit_base_estimators": False,
                    "use_clones": False,
                },
            },
        },
        "evaluation": {
            "plot_path": "evaluation/plots/",
            "confusion_matrix": False,
            "precision_recall_threshold": False,
            "roc_curve": False,
            "train": False,
            "validation": True,
            "optimal_threshold": False,
            "metric": {
                "pos": {"f1-score": True, "precision": True, "recall": True},
                "neg": {"f1-score": False, "precision": False, "recall": False},
                "PR_AUC": False,
                "macro_avg": True,
            },
        },
    }


@pytest.fixture(autouse=True)
def patch_trainer_globals(monkeypatch):
    """
    trainer.py's train_* functions reference a bare `path` name that is
    only ever assigned inside trainer.py's own `if __name__ == "__main__":`
    block. That block only runs when trainer.py is executed directly as a
    script (`python -m src.trainer`); it does NOT run on import. Since
    these tests import train_random_forest/train_knn/etc. and call them
    directly -- without going through that __main__ block -- `path` is
    still undefined in that context and referencing it raises
    NameError: name 'path' is not defined.

    We patch it onto the trainer module so the functions are actually
    callable in isolation, the same way any other caller (a notebook, a
    different script, another test) would need to. Passing `path` as an
    explicit function parameter instead of relying on a module global
    would remove the need for this fixture entirely.
    """
    from src import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "path", "test_output/", raising=False)
