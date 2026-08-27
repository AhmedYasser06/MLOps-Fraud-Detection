"""Convert the mlxtend EnsembleVoteClassifier to ONNX.

The problem
-----------
`models/Voting_Classifier.pkl` wraps:

    EnsembleVoteClassifier(
        clfs=[
            Pipeline(RobustScaler, LogisticRegression),
            Pipeline(RobustScaler, MLPClassifier),
            RandomForestClassifier,          # unscaled
        ],
        weights=[0.04, 0.8, 0.16],
        voting="soft",
    )

`skl2onnx` has no registered converter for `mlxtend.EnsembleVoteClassifier`
(it only knows sklearn's own `VotingClassifier`), so calling
`convert_sklearn()` on it directly raises a MissingShapeCalculator /
MissingConverter error. There's no clean way around that short of writing
and registering a custom skl2onnx converter class.

The fix used here
------------------
Soft voting is just `sum(weight_i * predict_proba_i(X)) / sum(weights)`.
That's cheap to reproduce as an ONNX graph ourselves:

  1. Convert each of the 3 sub-estimators to ONNX individually (each *is*
     skl2onnx-convertible -- LR and MLP already went through their own
     RobustScaler in sklearn_to_onnx.py, RF takes raw features).
  2. Use `onnx.compose` to merge the three graphs into one, feeding the
     same 30-feature input to all three.
  3. Add ONNX arithmetic nodes (Mul + Add + Div by weight sum) on top of
     the three probability outputs to reproduce the weighted average.

Output: onnx_models/voting_classifier.onnx, single input "features"
[batch, 30] -> single output "probabilities" [batch, 2].
"""

from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from onnx import compose, helper, numpy_helper, TensorProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "onnx_models"
OUTPUT_DIR.mkdir(exist_ok=True)

N_FEATURES = 30
INITIAL_TYPE = [("features", FloatTensorType([None, N_FEATURES]))]
NO_ZIPMAP = {"zipmap": False}


def sub_estimator_to_onnx(model, prefix: str) -> onnx.ModelProto:
    """Convert one sklearn/Pipeline estimator, renaming every node/tensor
    with `prefix` so the three graphs don't collide when merged."""
    onnx_model = convert_sklearn(
        model,
        initial_types=INITIAL_TYPE,
        options={id(model): NO_ZIPMAP},
        # skl2onnx picks a different default ai.onnx opset per estimator
        # type unless we pin BOTH domains explicitly -- otherwise
        # onnx.compose.merge_models refuses to merge graphs that declare
        # different opset versions for the same domain.
        target_opset={"": 17, "ai.onnx.ml": 1},
    )
    # Belt-and-suspenders: some estimator converters (e.g. the MLP subgraph)
    # still emit a lower default opset for the "" domain regardless of the
    # target_opset hint above. Force every graph to declare the exact same
    # opset_import before merging, or onnx.compose will refuse to merge them.
    for opset in onnx_model.opset_import:
        if opset.domain == "":
            opset.version = 17
        elif opset.domain == "ai.onnx.ml":
            opset.version = 1
    onnx.checker.check_model(onnx_model)
    # onnx.compose needs globally-unique names across graphs being merged
    onnx_model = compose.add_prefix(onnx_model, prefix=prefix)
    return onnx_model


def main():
    voting_data = joblib.load(MODELS_DIR / "Voting_Classifier.pkl")
    voting_model = voting_data["model"]
    threshold = voting_data["threshold"]
    weights = list(voting_model.weights)  # [0.04, 0.8, 0.16]
    print(f"weights={weights}  threshold={threshold:.6f}")

    lr_pipeline, nn_pipeline, rf_model = voting_model.clfs

    # --- 1. Convert each sub-estimator, prefixing names to avoid clashes ---
    g_lr = sub_estimator_to_onnx(lr_pipeline, "lr_")
    g_nn = sub_estimator_to_onnx(nn_pipeline, "nn_")
    g_rf = sub_estimator_to_onnx(rf_model, "rf_")

    # After add_prefix, each graph has its own input, e.g. "lr_features".
    # Rename all three back to a single shared name "features" so we can
    # feed one tensor to all three branches.
    for g, prefix in ((g_lr, "lr_"), (g_nn, "nn_"), (g_rf, "rf_")):
        g.graph.input[0].name = "features"
        for node in g.graph.node:
            for i, inp in enumerate(node.input):
                if inp == f"{prefix}features":
                    node.input[i] = "features"

    # --- 2. Merge the three graphs side by side --------------------------
    # onnx.compose.merge_models is built for *sequential* composition
    # (output of model A feeds input of model B), not 3 parallel branches
    # sharing one input -- it errors on the shared "features" name. So we
    # build the merged GraphProto by hand instead: take all nodes,
    # initializers, and value_info from all three graphs (already uniquely
    # prefixed) and declare a single "features" input.
    merged = onnx.ModelProto()
    merged.CopyFrom(g_lr)
    merged.graph.ClearField("node")
    merged.graph.ClearField("initializer")
    merged.graph.ClearField("value_info")
    merged.graph.ClearField("input")
    merged.graph.ClearField("output")

    merged.graph.input.extend([g_lr.graph.input[0]])  # shared "features" input

    proba_outputs = ["lr_probabilities", "nn_probabilities", "rf_probabilities"]
    for g in (g_lr, g_nn, g_rf):
        merged.graph.node.extend(g.graph.node)
        merged.graph.initializer.extend(g.graph.initializer)
        merged.graph.value_info.extend(g.graph.value_info)
        merged.graph.value_info.extend(
            g.graph.output
        )  # keep intermediate proba tensors typed

    # --- 3. Add weighted-average nodes on top of the 3 proba outputs ----
    weight_sum = float(sum(weights))
    new_nodes = []
    new_initializers = []
    weighted_names = []

    for i, (out_name, w) in enumerate(zip(proba_outputs, weights)):
        w_tensor_name = f"weight_{i}"
        new_initializers.append(
            numpy_helper.from_array(np.array([w], dtype=np.float32), name=w_tensor_name)
        )
        weighted_name = f"weighted_{i}"
        new_nodes.append(
            helper.make_node(
                "Mul", [out_name, w_tensor_name], [weighted_name], name=f"mul_{i}"
            )
        )
        weighted_names.append(weighted_name)

    sum_01 = "sum_01"
    new_nodes.append(
        helper.make_node(
            "Add", [weighted_names[0], weighted_names[1]], [sum_01], name="add_01"
        )
    )
    sum_all = "sum_all"
    new_nodes.append(
        helper.make_node("Add", [sum_01, weighted_names[2]], [sum_all], name="add_all")
    )

    wsum_tensor_name = "weight_sum"
    new_initializers.append(
        numpy_helper.from_array(
            np.array([weight_sum], dtype=np.float32), name=wsum_tensor_name
        )
    )
    new_nodes.append(
        helper.make_node(
            "Div", [sum_all, wsum_tensor_name], ["probabilities"], name="div_final"
        )
    )

    merged.graph.node.extend(new_nodes)
    merged.graph.initializer.extend(new_initializers)

    # merged.graph.output was cleared earlier, so this is the ONLY graph
    # output -- a clean single-output interface, the 3 sub-model
    # label/probability outputs stay as internal (typed) intermediates.
    final_output = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, [None, 2]
    )
    merged.graph.output.extend([final_output])

    onnx.checker.check_model(merged)
    out_path = OUTPUT_DIR / "voting_classifier.onnx"
    onnx.save(merged, out_path)
    print(f"[OK] voting_classifier -> {out_path}")
    print(f"     inputs : {[i.name for i in merged.graph.input]}")
    print(f"     outputs: {[o.name for o in merged.graph.output]}")

    # --- 4. Smoke test against the original mlxtend model ----------------
    rng = np.random.default_rng(16)
    X_sample = rng.standard_normal((5, N_FEATURES)).astype(np.float32)

    sklearn_proba = voting_model.predict_proba(X_sample)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_proba = sess.run(["probabilities"], {"features": X_sample})[0]

    max_diff = np.abs(sklearn_proba - onnx_proba).max()
    status = "OK" if max_diff < 1e-4 else "MISMATCH"
    print(f"     smoke test: max abs diff = {max_diff:.2e} [{status}]")


if __name__ == "__main__":
    main()
