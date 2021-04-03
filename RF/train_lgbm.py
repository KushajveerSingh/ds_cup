from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
from data import get_data
import numpy as np
import pickle
import hydra


@hydra.main(config_path="config", config_name="config")
def random_forest(cfg):
    # Load data
    train_df, valid_df, test_df = get_data(cfg)

    # Normalize data
    df_min = train_df.min()
    df_max = train_df.max()
    train_df = (train_df-df_min)/(df_max-df_min)
    valid_df = (valid_df-df_min)/(df_max-df_min)
    test_df = (test_df-df_min)/(df_max-df_min)
    df = pd.concat([train_df, valid_df])

    # Remove columns and split data into (X,y)
    df = df.drop(['State_AL', 'State_NC', 'isNaN_rep_income', 'State_FL', 'State_LA',
       'isNaN_uti_card_50plus_pct', 'State_SC', 'State_GA', 'State_MS',
       'auto_open_36_month_num', 'card_open_36_month_num', 'ind_acc_XYZ'], axis=1)
    X = df.drop("Default_ind", axis=1).values
    y = df["Default_ind"].values

    test_df = test_df.drop(['State_AL', 'State_NC', 'isNaN_rep_income', 'State_FL', 'State_LA',
        'isNaN_uti_card_50plus_pct', 'State_SC', 'State_GA', 'State_MS',
        'auto_open_36_month_num', 'card_open_36_month_num', 'ind_acc_XYZ'], axis=1)
    test_X = test_df.drop('Default_ind', axis=1).values
    test_y = test_df['Default_ind'].values

    # Below 2 lines needed for cross-validation in RandomizedSearchCV
    split_index = [-1]*len(train_df) + [0]*len(valid_df)
    pds = PredefinedSplit(test_fold=split_index)

    # Create classifier and the hyperparameter search space
    classifier = LGBMClassifier(objective="binary", n_jobs=-1)
    param_grid = {
        "num_leaves": [31, 63, 127, 255, 511],
        "boosting_type": ["gbdt", "dart", "goss"],
        "learning_rate": [0.1],
        "max_depth": np.arange(1, 30),
        "n_estimators": [100, 400, 700],
        "importance_type": ["split", "gain"],
    }

    model = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        scoring="f1",
        n_iter=100,
        verbose=1,
        n_jobs=1,
        cv=pds,
    )

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
    with open("lgbm.pkl", "wb") as f:
        pickle.dump(model.best_estimator_, f)

    clf = model.best_estimator_
    y_pred = clf.predict(test_X)

    print(f"f1 = {metrics.f1_score(test_y, y_pred)}")
    print(f"roc auc = {metrics.roc_auc_score(test_y, y_pred)}")


if __name__ == "__main__":
    random_forest()
