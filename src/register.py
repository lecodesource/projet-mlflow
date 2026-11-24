import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(
    experiment_names=["cancer-sein"],
    filter_string="attributes.run_name LIKE 'logreg-%'",
    order_by=["metrics.recall DESC", "metrics.accuracy DESC"],
)
best = runs.iloc[0]
model_id = mlflow.get_run(best.run_id).outputs.model_outputs[0].model_id
v = mlflow.register_model(f"models:/{model_id}", "cancer-sein")

MlflowClient().set_registered_model_alias("cancer-sein", "champion", v.version)
print(
    f"champion = version {v.version} : {best['tags.mlflow.runName']}, "
    f"recall {best['metrics.recall']}, accuracy {best['metrics.accuracy']}"
)


champion = mlflow.sklearn.load_model("models:/cancer-sein@champion")
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
patientes = X.sample(8, random_state=0)
print("prédit :", champion.predict(patientes).tolist())
print("réel :", (1 - y[patientes.index]).tolist())
