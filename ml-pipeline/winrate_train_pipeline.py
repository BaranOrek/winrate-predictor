import argparse
import kfp
from kfp import dsl, compiler
from kfp.dsl import Dataset, Input, Output, Model, Metrics
import google.cloud.aiplatform as aip


# Pipeline Component: Data Ingestion
@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image="python:3.11.9-slim"
)
def download_data(project_id: str, bucket: str, file_name: str, dataset: Output[Dataset]):
    """Download a CSV file from Google Cloud Storage."""
    from google.cloud import storage
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    client = storage.Client(project=project_id)
    storage_bucket = client.bucket(bucket)
    blob = storage_bucket.blob(file_name)
    blob.download_to_filename(dataset.path + ".csv")
    logging.info("Downloaded gs://%s/%s", bucket, file_name)


# Pipeline Component: Data Preprocessing
@dsl.component(
    packages_to_install=["pandas==2.1.4", "numpy==1.26.4"],
    base_image="python:3.11.9-slim"
)
def preprocess_data(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset]
):
    """Select the same model inputs used in the local notebook."""
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    feature_columns = [
        "league", "elo_home", "elo_away", "crowd_present",
        "stadium_capacity", "attendance_pct", "travel_distance_km",
        "timezones_crossed", "is_derby", "venue_familiarity"
    ]
    selected_columns = feature_columns + ["home_result"]

    train_df = pd.read_csv(train_data.path + ".csv")
    test_df = pd.read_csv(test_data.path + ".csv")

    # Keep the original fixed holdout; do not randomly split it again.
    train_df = train_df[selected_columns].copy()
    test_df = test_df[selected_columns].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training and test datasets must not be empty.")
    if train_df[selected_columns].isna().any().any() or test_df[selected_columns].isna().any().any():
        raise ValueError("Selected model columns contain missing values.")

    train_df.to_csv(train_dataset.path + ".csv", index=False)
    test_df.to_csv(test_dataset.path + ".csv", index=False)
    logging.info("Prepared %d training rows and %d test rows", len(train_df), len(test_df))


# Pipeline Component: Logistic Regression Training
@dsl.component(
    packages_to_install=["numpy==1.26.4", "pandas==2.1.4", "scikit-learn==1.9.1"],
    base_image="python:3.11.9-slim"
)
def train_logistic_regression(train_dataset: Input[Dataset], out_model: Output[Model]):
    """Fit and serialize preprocessing plus Logistic Regression together."""
    import pandas as pd
    import pickle
    import logging
    import sys
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    df = pd.read_csv(train_dataset.path + ".csv")
    feature_columns = [
        "league", "elo_home", "elo_away", "crowd_present",
        "stadium_capacity", "attendance_pct", "travel_distance_km",
        "timezones_crossed", "is_derby", "venue_familiarity"
    ]
    numeric_columns = [column for column in feature_columns if column != "league"]

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["league"]),
            ("numeric", StandardScaler(), numeric_columns)
        ]
    )
    model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(C=0.01, class_weight=None, max_iter=1000, random_state=42))
    ])
    model.fit(df[feature_columns], df["home_result"])

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "logistic_regression"
    with open(out_model.path + ".pkl", "wb") as file:
        pickle.dump(model, file)
    logging.info("Trained Logistic Regression on %d rows", len(df))


# Pipeline Component: Model Evaluation
@dsl.component(
    packages_to_install=["numpy==1.26.4", "pandas==2.1.4", "scikit-learn==1.9.1"],
    base_image="python:3.11.9-slim"
)
def evaluate_model(
    model: Input[Model],
    test_dataset: Input[Dataset],
    evaluation_metrics: Output[Metrics]
) -> float:
    """Evaluate multiclass predictions on the fixed holdout set."""
    import pandas as pd
    import pickle
    import logging
    import sys
    from sklearn.metrics import accuracy_score, f1_score

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    df = pd.read_csv(test_dataset.path + ".csv")
    feature_columns = [
        "league", "elo_home", "elo_away", "crowd_present",
        "stadium_capacity", "attendance_pct", "travel_distance_km",
        "timezones_crossed", "is_derby", "venue_familiarity"
    ]
    with open(model.path + ".pkl", "rb") as file:
        trained_model = pickle.load(file)

    predictions = trained_model.predict(df[feature_columns])
    accuracy = float(accuracy_score(df["home_result"], predictions))
    macro_f1 = float(f1_score(df["home_result"], predictions, average="macro", zero_division=0))
    evaluation_metrics.log_metric("accuracy", accuracy)
    evaluation_metrics.log_metric("macro_f1", macro_f1)
    logging.info("Fixed test accuracy: %.4f; macro-F1: %.4f", accuracy, macro_f1)
    return accuracy


# Pipeline Component: Smoke Test
@dsl.component(
    packages_to_install=["numpy==1.26.4", "pandas==2.1.4", "scikit-learn==1.9.1"],
    base_image="python:3.11.9-slim"
)
def smoke_test(model: Input[Model], test_dataset: Input[Dataset]):
    """Check that the serialized model predicts and returns valid probabilities."""
    import pandas as pd
    import pickle
    import logging
    import sys
    import numpy as np

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    feature_columns = [
        "league", "elo_home", "elo_away", "crowd_present",
        "stadium_capacity", "attendance_pct", "travel_distance_km",
        "timezones_crossed", "is_derby", "venue_familiarity"
    ]
    df = pd.read_csv(test_dataset.path + ".csv")
    sample = df.iloc[:3][feature_columns]
    if len(sample) != 3:
        raise ValueError("Smoke test requires at least three test rows.")

    with open(model.path + ".pkl", "rb") as file:
        trained_model = pickle.load(file)
    predictions = trained_model.predict(sample)
    probabilities = trained_model.predict_proba(sample)

    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Smoke test produced invalid class probabilities.")
    if not set(predictions).issubset({"home_win", "draw", "away_win"}):
        raise ValueError("Smoke test produced an unexpected class label.")
    logging.info("Smoke test predictions: %s", predictions.tolist())


# Pipeline Component: Upload Model
@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image="python:3.11.9-slim"
)
def upload_model_to_gcs(project_id: str, model_repo: str, model: Input[Model]):
    """Publish the model at the path consumed by the existing CI/CD pipeline."""
    from google.cloud import storage
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    client = storage.Client(project=project_id)
    bucket = client.bucket(model_repo)
    blob = bucket.blob("model.pkl")
    blob.upload_from_filename(model.path + ".pkl")
    logging.info("Uploaded model to gs://%s/model.pkl", model_repo)


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(name="winrate-predictor-training-pipeline")
def ml_pipeline(
    project_id: str,
    data_bucket: str,
    trainset_filename: str,
    testset_filename: str,
    model_repo: str,
    accuracy_threshold: float = 0.49
):
    training_data = download_data(
        project_id=project_id, bucket=data_bucket, file_name=trainset_filename
    )
    test_data = download_data(
        project_id=project_id, bucket=data_bucket, file_name=testset_filename
    )
    prepared_data = preprocess_data(
        train_data=training_data.outputs["dataset"],
        test_data=test_data.outputs["dataset"]
    )
    training_job = train_logistic_regression(
        train_dataset=prepared_data.outputs["train_dataset"]
    )
    evaluation_job = evaluate_model(
        model=training_job.outputs["out_model"],
        test_dataset=prepared_data.outputs["test_dataset"]
    )
    smoke_test_job = smoke_test(
        model=training_job.outputs["out_model"],
        test_dataset=prepared_data.outputs["test_dataset"]
    ).after(evaluation_job)

    # Publish only when the model passes the accuracy gate and smoke test.
    with dsl.If(evaluation_job.outputs["Output"] >= accuracy_threshold):
        upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_job.outputs["out_model"]
        ).after(smoke_test_job)


def compile_pipeline(target_file):
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path=target_file)


def _run_pipeline():
    PROJECT_ID = "winrate-project-512916"
    REGION = "us-central1"
    PIPELINE_ROOT = "gs://winrate-temp-2171997/vertex-pipelines"

    aip.init(project=PROJECT_ID, location=REGION)
    job = aip.PipelineJob(
        display_name="winrate-predictor-training-pipeline",
        enable_caching=False,
        template_path="winrate_training_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        location=REGION,
        parameter_values={
            "project_id": PROJECT_ID,
            "data_bucket": "winrate-data-2171997",
            "trainset_filename": "home_advantage_initial.csv",
            "testset_filename": "home_advantage_test.csv",
            "model_repo": "winrate-models-2171997",
            "accuracy_threshold": 0.49
        }
    )
    job.run()


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_file", type=str, default="winrate_training_pipeline.yaml",
        help="Compiled pipeline YAML path"
    )
    parser.add_argument("--run", action="store_true", help="Submit the compiled pipeline to Vertex AI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    compile_pipeline(args.target_file)
    if args.run:
        _run_pipeline()
