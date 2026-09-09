"""Train the complete clustering, prediction, and explanation-ready system."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .clustering import benchmark_clusters
from .config import ARTIFACT_PATH, CLUSTERED_DATA_PATH, CLUSTER_FEATURES, FEATURES, METRICS_PATH, MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET
from .data import clean_lifestyle, load_lifestyle
from .features import classifier_candidates


def train_all() -> dict:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = clean_lifestyle(load_lifestyle())

    cluster_metrics, clusterer, _ = benchmark_clusters(data, CLUSTER_FEATURES)
    data = data.copy()
    data["Cluster"] = clusterer.predict(data[CLUSTER_FEATURES])
    data.to_csv(CLUSTERED_DATA_PATH, index=False)

    x_train, x_test, y_train, y_test = train_test_split(data[FEATURES + ["Cluster"]], data[TARGET], test_size=0.2, stratify=data[TARGET], random_state=RANDOM_STATE)
    model_results, fitted = [], {}
    for name, model in classifier_candidates().items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        model_results.append({"model": name, "accuracy": round(float(accuracy_score(y_test, prediction)), 4), "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 4), "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 4), "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 4)})
        fitted[name] = model
    results_df = pd.DataFrame(model_results).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_name = str(results_df.iloc[0]["model"])
    best_model = fitted[best_name]
    transformed_train = best_model.named_steps["preprocessor"].transform(x_train.iloc[:1000])
    transformed_train = transformed_train.toarray() if hasattr(transformed_train, "toarray") else transformed_train
    feature_names = list(best_model.named_steps["preprocessor"].get_feature_names_out())

    artifact = {"model": best_model, "clusterer": clusterer, "features": FEATURES, "cluster_features": CLUSTER_FEATURES, "training_reference": transformed_train, "transformed_feature_names": feature_names, "best_model_name": best_name}
    joblib.dump(artifact, ARTIFACT_PATH)
    best_cluster = cluster_metrics.iloc[0]
    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset_rows": int(len(data)), "target": TARGET, "cluster_selection": cluster_metrics.round(4).to_dict(orient="records"), "selected_clusterer": {"algorithm": str(best_cluster["algorithm"]), "n_clusters": int(best_cluster["n_clusters"]), "silhouette": round(float(best_cluster["silhouette"]), 4), "davies_bouldin": round(float(best_cluster["davies_bouldin"]), 4), "calinski_harabasz": round(float(best_cluster["calinski_harabasz"]), 4)}, "prediction_models": results_df.to_dict(orient="records"), "selected_prediction_model": best_name, "classification_report": classification_report(y_test, best_model.predict(x_test), output_dict=True)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return metrics


def main() -> None:
    print(json.dumps(train_all(), indent=2, default=float))


if __name__ == "__main__":
    main()
