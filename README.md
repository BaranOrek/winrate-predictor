# DE Assignment 1 – playground

Starting point for Assignment 1 (MLOps for an ML application). Everything in this folder is copied **verbatim**
from the lab repository <https://github.com/IndikaKuma/DE2026> (labs 2, 3, 4 and 5). Nothing has been adapted yet:
the adaptation is the assignment.

The diabetes model (`model.pkl`) and the diabetes data (CSV files) were left out on purpose: the diabetes
application may not be used in the assignment.

## Where each file comes from

| Playground path | Lab source | Purpose |
|---|---|---|
| `prediction-api/` | `lab2/prediction-api` | Flask API + Dockerfile + pytest example |
| `prediction-ui/` | `lab2/prediction-ui` | Flask UI (HTML form) + Dockerfile |
| `ml-pipeline/dt_lr_train_pipeline.py` | `lab4/ml-pipeline` | Vertex AI (Kubeflow) training pipeline |
| `custom_cloud_build_builder_tools/` | `lab4/custom_cloud_build_builder_tools` | Cloud Build tool that runs a compiled pipeline |
| `cicd/cloud_build_model_deployment.json` | `lab3` | CI-CD: test, build, deploy API + UI to Cloud Run |
| `cicd/cicd-model-deployment-dt.json`, `-lr.json` | `lab3` | Variants triggered by Pub/Sub model uploads |
| `cicd/ml_pipeline_operationalization_cicd.json` | `lab4` | CI-CD: compile, publish and run the training pipeline |
| `parameters.json` | `lab4` | Pipeline parameters (goes to the config bucket) |
| `notebooks/Lab4_1.ipynb` | `lab4` | Pipeline development notebook |
| `optional-lab5/` | `lab5` | Container-based pipeline components (optional) |

## Paths inside the build files that must be changed

The Cloud Build files still use the folder layout of the lab repo. In this playground the folders are one level
higher:

| In the build files (lab layout) | In this playground |
|---|---|
| `./lab2/prediction-api` | `./prediction-api` |
| `./lab2/prediction-ui` | `./prediction-ui` |
| `./lab4/ml-pipeline` | `./ml-pipeline` |
| `./lab4/custom_cloud_build_builder_tools/pipeline_executor` | `./custom_cloud_build_builder_tools/pipeline_executor` |
| `./lab5/components/...` | `./optional-lab5/components/...` |

## Still diabetes-specific (to be replaced)

* `prediction-api/diabetes_predictor.py`, `app.py` (route `/diabetes_predictor/`), `tests/`, `testResources/`
* `prediction-ui/app.py` (route `/checkdiabetes`), `templates/`
* `ml-pipeline/dt_lr_train_pipeline.py` (features, labels, names), `parameters.json`
* names such as `diabetes_predictor_dt_lr_pipeline.yaml` in the CI-CD files
