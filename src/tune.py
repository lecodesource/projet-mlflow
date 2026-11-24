"""Recherche deC:autologcréeunrun parentetunrunenfantparvaleur."""

import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
y = 1 - y
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cancer-sein")
mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False)
modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
grille = {"logisticregression__C": [0.01, 0.1, 1.0, 3.0, 10.0]}
recherche = GridSearchCV(modele, grille, cv=5, scoring="recall")
mlflow.start_run(run_name="recherche-C")
recherche.fit(X_tr, y_tr)
mlflow.end_run()
print(
    f"meilleurC:{recherche.best_params_['logisticregression__C']}|"
    f"recallmoyen(validationcroisée):{recherche.best_score_:.4f}"
)
