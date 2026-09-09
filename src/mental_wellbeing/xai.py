"""SHAP and LIME explanations, with imports delayed until an explanation is requested."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

# Keep Matplotlib's cache inside the project so restricted Windows profiles do
# not prevent LIME from creating its font cache.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))


def shap_explanation(model, transformed_row, feature_names: list[str], class_index: int) -> pd.DataFrame:
    import shap
    estimator = model.named_steps["model"]
    values = shap.TreeExplainer(estimator).shap_values(transformed_row)
    if isinstance(values, list):
        selected = values[class_index][0]
    elif getattr(values, "ndim", 0) == 3:
        selected = values[0, :, class_index]
    else:
        selected = values[0]
    result = pd.DataFrame({"feature": feature_names, "shap_value": selected})
    return result.assign(abs_shap=lambda x: x.shap_value.abs()).sort_values("abs_shap", ascending=False)


def lime_explanation(model, training_transformed, transformed_row, feature_names: list[str], class_index: int) -> pd.DataFrame:
    from lime.lime_tabular import LimeTabularExplainer
    estimator = model.named_steps["model"]
    explainer = LimeTabularExplainer(np.asarray(training_transformed), feature_names=feature_names, class_names=[str(v) for v in estimator.classes_], mode="classification", random_state=42)
    explanation = explainer.explain_instance(np.asarray(transformed_row)[0], estimator.predict_proba, labels=(class_index,), num_features=min(10, len(feature_names)))
    return pd.DataFrame(explanation.as_list(label=class_index), columns=["feature", "weight"])
