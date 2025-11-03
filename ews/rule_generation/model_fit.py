import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif


def fit_hybrid_model(
    X, y,
    plot: bool = True,
    random_state: int = 42,
    params_grid: dict = None,
    cv_splits: int = 5
):
    """
    Fit interpretable decision tree (Hybrid Model):
    SPM patterns + contextual/demographic features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset (after preprocessing)
    y : pd.Series
        Target labels (risk_label)
    plot : bool
        Show plots (feature importance, MI scatter)
    random_state : int
        Random seed
    params_grid : dict
        Parameter grid for GridSearchCV
    cv_splits : int
        Number of CV folds

    Returns
    -------
    dict
        {
          "base_model": base_clf,
          "best_model": best_clf,
          "reports": {
              "base": base_report_df,
              "best": best_report_df
          }
        }
    """

    # --- One-hot encode categorical columns ---
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X = X.reindex(sorted(X.columns), axis=1)

    # --- Identify feature groups ---
    has_cols = [c for c in X.columns if c.startswith("has_")]
    context_cols = [c for c in X.columns if not c.startswith("has_")]
    print(f"SPM cols: {len(has_cols)}, context cols: {len(context_cols)}")

    # --- Define final feature set (Hybrid) ---
    X_hybrid = X[context_cols]

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # --- Base model ---
    base_clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state
    )
    base_clf.fit(X_train, y_train)

    y_pred_base = base_clf.predict(X_test)

    base_report = pd.DataFrame(classification_report(y_test, y_pred_base, output_dict=True)).T
    print("\n=== Base Model Performance ===")
    print(base_report)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_base))

    # --- Grid Search for Hyperparameter Tuning ---
    if params_grid is None:
        params_grid = {
            "max_depth": [3, 5, 7, 9],
            "min_samples_leaf": [2, 3, 4, 5],
            "criterion": ["gini", "entropy"],
        }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        param_grid=params_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    print("\nBest Parameters:", grid_search.best_params_)

    y_pred_best = best_clf.predict(X_test)
    best_report = pd.DataFrame(classification_report(y_test, y_pred_best, output_dict=True)).T
    print("\n=== Tuned Model Performance ===")
    print(best_report)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_best))

    # --- Feature importance plot ---
    if plot:
        imp = pd.DataFrame({
            "feature": X_hybrid.columns,
            "importance": best_clf.feature_importances_
        }).sort_values("importance", ascending=False)
        sns.barplot(y="feature", x="importance", data=imp.head(15))
        plt.title("Top 15 Features (Best Model)")
        plt.tight_layout()
        plt.show()

    # --- Rule Extraction ---
    print("\n=== Decision Tree Rules (Best Model) ===")
    print(export_text(best_clf, feature_names=list(X_hybrid.columns)))

    # --- Return results ---
    return {
        "base_model": base_clf,
        "best_model": best_clf,
        "reports": {
            "base": base_report,
            "best": best_report
        }
    }