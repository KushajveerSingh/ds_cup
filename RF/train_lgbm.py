from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn import metrics
import pandas as pd
from data import get_data
import numpy as np
import pickle
import hydra


@hydra.main(config_path="config", config_name="config")
def random_forest(cfg):
    # Load data
    train_df, valid_df, test_df = get_data(cfg)
    train_df = train_df[['avg_card_debt', 'card_age', 'non_mtg_acc_past_due_12_months_num', 'inq_12_month_num', 'uti_card', 'Default_ind']]
    valid_df = valid_df[['avg_card_debt', 'card_age', 'non_mtg_acc_past_due_12_months_num', 'inq_12_month_num', 'uti_card', 'Default_ind']]
    test_df = test_df[['avg_card_debt', 'card_age', 'non_mtg_acc_past_due_12_months_num', 'inq_12_month_num', 'uti_card', 'Default_ind']]

    train_x = train_df.drop("Default_ind", axis=1)
    train_y = train_df["Default_ind"]
    valid_x = valid_df.drop("Default_ind", axis=1)
    valid_y = valid_df["Default_ind"]
    test_x = test_df.drop("Default_ind", axis=1)
    test_y = test_df["Default_ind"]

    # Normalize data
    df_min = train_x.min()
    df_max = train_x.max()

    train_x = (train_x-df_min)/(df_max-df_min)
    valid_x = (valid_x-df_min)/(df_max-df_min)
    test_x = (test_x-df_min)/(df_max-df_min)
    X = pd.concat([train_x, valid_x])
    X = X.values
    y = pd.concat([train_y, valid_y])
    y = y.values
    test_X = test_x.values
    test_y = test_y.values

    # Below 2 lines needed for cross-validation in RandomizedSearchCV
    split_index = [-1]*len(train_df) + [0]*len(valid_df)
    pds = PredefinedSplit(test_fold=split_index)

    # Create classifier and the hyperparameter search space
    classifier = LGBMClassifier(objective="binary", n_jobs=-1)
    param_grid = {
        "num_leaves": [31, 63, 127, 255, 511],
        "boosting_type": ["gbdt", "dart", "goss"],
        "learning_rate": [0.1, 0.5, 0.001],
        "max_depth": np.arange(1, 30),
        "n_estimators": [100, 400, 700, 900],
        "importance_type": ["split", "gain"],
    }

    model = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        scoring="f1",
        n_iter=200,
        verbose=2,
        n_jobs=1,
        cv=pds,
    )

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
    with open("lgbm.pkl", "wb") as f:
        pickle.dump([model.best_estimator_, df_min, df_max], f)

    clf = model.best_estimator_
    y_pred = clf.predict(test_X)

    print(f"f1 = {metrics.f1_score(test_y, y_pred)}")
    print(f"roc auc = {metrics.roc_auc_score(test_y, y_pred)}")


if __name__ == "__main__":
    random_forest()
