"""
Unit tests for src/trainer.py

Strategy:
- `evaluate_model` (plotting + metrics) is mocked in every test. It's
  eval_utils' job to test, not trainer's, and the real version needs a
  writable `path` and produces matplotlib figures we don't care about here.
- Model fitting itself is real (tiny synthetic data + tiny hyperparameters)
  so we actually verify trainer.py wires sklearn correctly, rather than
  mocking sklearn away entirely.
- RandomizedSearchCV / GridSearchCV are mocked in the "search" tests so
  the tests stay fast and deterministic.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src import trainer


# ── Random Forest ──────────────────────────────────────────────────
def test_train_random_forest_uses_config_params(sample_data, base_trainer_config):
    X_train, y_train, X_val, y_val = sample_data
    fake_comparison = {"Random Forest": {"F1 Score Positive class": 0.8}}

    with patch.object(
        trainer, "evaluate_model", return_value=(fake_comparison, 0.42)
    ) as mock_eval:
        result = trainer.train_random_forest(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=base_trainer_config,
        )

    assert isinstance(result["model"], RandomForestClassifier)
    assert result["parameters"] == base_trainer_config["trainer"]["Random_forest"]["parameters"]
    assert result["threshold"] == 0.42
    # model was actually fit on the training data
    assert result["model"].n_estimators == 5
    mock_eval.assert_called_once()


def test_train_random_forest_randomized_search_overrides_config_params(
    sample_data, base_trainer_config
):
    X_train, y_train, X_val, y_val = sample_data
    config = copy.deepcopy(base_trainer_config)
    config["trainer"]["Random_forest"]["Randomized_Search"] = True

    best_params = {"n_estimators": 200, "min_samples_leaf": 2, "min_samples_split": 5}
    mock_search = MagicMock()
    mock_search.best_params_ = best_params

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)), \
         patch.object(trainer, "RandomizedSearchCV", return_value=mock_search) as mock_rscv:
        result = trainer.train_random_forest(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=config,
        )

    mock_search.fit.assert_called_once_with(X_train, y_train)
    mock_rscv.assert_called_once()
    assert result["parameters"] == best_params
    assert result["model"].n_estimators == 200


# ── KNN ─────────────────────────────────────────────────────────────
def test_train_knn_uses_config_params(sample_data, base_trainer_config):
    X_train, y_train, X_val, y_val = sample_data

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)) as mock_eval:
        result = trainer.train_knn(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=base_trainer_config,
        )

    assert isinstance(result["model"], KNeighborsClassifier)
    assert result["parameters"] == base_trainer_config["trainer"]["KNN"]["parameters"]
    assert result["model"].n_neighbors == 3
    # train_knn's return dict has no "threshold" key (unlike the others)
    assert "threshold" not in result
    mock_eval.assert_called_once()


def test_train_knn_grid_search_overrides_config_params(sample_data, base_trainer_config):
    X_train, y_train, X_val, y_val = sample_data
    config = copy.deepcopy(base_trainer_config)
    config["trainer"]["KNN"]["grid_search"] = True

    best_params = {"n_neighbors": 7, "weights": "distance", "algorithm": "auto"}
    mock_search = MagicMock()
    mock_search.best_params_ = best_params

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)), \
         patch.object(trainer, "RandomizedSearchCV", return_value=mock_search):
        result = trainer.train_knn(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=config,
        )

    mock_search.fit.assert_called_once_with(X_train, y_train)
    assert result["parameters"] == best_params
    assert result["model"].n_neighbors == 7


# ── Logistic Regression ────────────────────────────────────────────
def test_train_logistic_regression_uses_config_params(sample_data, base_trainer_config):
    X_train, y_train, X_val, y_val = sample_data

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.37)) as mock_eval:
        result = trainer.train_logistic_regression(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=base_trainer_config,
        )

    assert isinstance(result["model"], LogisticRegression)
    assert result["parameters"] == base_trainer_config["trainer"]["Logistic_Regression"]["parameters"]
    assert result["threshold"] == 0.37
    mock_eval.assert_called_once()


def test_train_logistic_regression_grid_search_overrides_config_params(
    sample_data, base_trainer_config
):
    X_train, y_train, X_val, y_val = sample_data
    config = copy.deepcopy(base_trainer_config)
    config["trainer"]["Logistic_Regression"]["grid_search"] = True

    best_params = {"C": 10.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 400}
    mock_search = MagicMock()
    mock_search.best_params_ = best_params

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)), \
         patch.object(trainer, "GridSearchCV", return_value=mock_search):
        result = trainer.train_logistic_regression(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=config,
        )

    mock_search.fit.assert_called_once_with(X_train, y_train)
    assert result["parameters"] == best_params


# ── Neural Network ──────────────────────────────────────────────────
def test_train_neural_network_parses_hidden_layer_sizes_string(
    sample_data, base_trainer_config
):
    """hidden_layer_sizes is stored as the *string* '(4, 2)' in config
    (see trainer_config.yml) and trainer.py does eval() on it -- check
    it actually becomes the tuple (4, 2) on the fitted model."""
    X_train, y_train, X_val, y_val = sample_data

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)) as mock_eval:
        result = trainer.train_neural_network(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=base_trainer_config,
        )

    assert isinstance(result["model"], MLPClassifier)
    assert result["model"].hidden_layer_sizes == (4, 2)
    assert result["model"].activation == "relu"
    mock_eval.assert_called_once()


def test_train_neural_network_randomized_search_uses_best_params(
    sample_data, base_trainer_config
):
    X_train, y_train, X_val, y_val = sample_data
    config = copy.deepcopy(base_trainer_config)
    config["trainer"]["Neural_Network"]["Randomized_Search"] = True

    best_params = {"hidden_layer_sizes": (10, 5), "activation": "relu", "max_iter": 50}
    mock_search = MagicMock()
    mock_search.best_params_ = best_params

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)), \
         patch.object(trainer, "MLPClassifier", wraps=MLPClassifier) as mock_mlp, \
         patch.object(trainer, "RandomizedSearchCV", return_value=mock_search):
        trainer.train_neural_network(
            X_train, y_train, X_val, y_val,
            random_seed=42,
            model_comparison={},
            trainer=config,
        )

    mock_mlp.assert_called_with(**best_params)


# ── Voting Classifier ───────────────────────────────────────────────
@pytest.fixture
def fitted_submodels(sample_data):
    X_train, y_train, _, _ = sample_data
    lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=50, random_state=42).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_train, y_train)
    return {
        "Logistic_Regression": {"model": lr},
        "Neural_Network": {"model": mlp},
        "Random_forest": {"model": rf},
    }


def test_train_voting_classifier_missing_models_raises_value_error(
    sample_data, base_trainer_config
):
    X_train, y_train, X_val, y_val = sample_data
    incomplete_models = {"Logistic_Regression": {"model": LogisticRegression()}}

    with pytest.raises(ValueError, match="missing"):
        trainer.train_voting_classifier(
            X_train, y_train, X_val, y_val,
            models=incomplete_models,
            random_seed=42,
            model_comparison={},
            trainer=base_trainer_config,
        )


def test_train_voting_classifier_success(
    sample_data, base_trainer_config, fitted_submodels
):
    X_train, y_train, X_val, y_val = sample_data

    with patch.object(trainer, "evaluate_model", return_value=({}, 0.5)) as mock_eval:
        result = trainer.train_voting_classifier(
            X_train, y_train, X_val, y_val,
            models=fitted_submodels,
            random_seed=42,
            model_comparison={},
            trainer=base_trainer_config,
        )

    assert result["model"] is not None
    assert result["parameters"] == base_trainer_config["trainer"]["Voting_Classifier"]["parameters"]
    mock_eval.assert_called_once()
    # sanity check the ensemble can actually produce predictions
    preds = result["model"].predict(X_val)
    assert len(preds) == len(y_val)


def test_train_voting_classifier_init_failure_wrapped_as_runtime_error(
    sample_data, base_trainer_config, fitted_submodels
):
    X_train, y_train, X_val, y_val = sample_data

    with patch.object(
        trainer, "EnsembleVoteClassifier", side_effect=Exception("boom")
    ):
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            trainer.train_voting_classifier(
                X_train, y_train, X_val, y_val,
                models=fitted_submodels,
                random_seed=42,
                model_comparison={},
                trainer=base_trainer_config,
            )
