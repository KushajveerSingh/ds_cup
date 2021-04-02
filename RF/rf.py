from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from data import get_data
import numpy as np
import pickle
import hydra


@hydra.main(config_path="config", config_name="config")
def random_forest(cfg):
    train_df, valid_df, test_df = get_data(cfg)
    df = pd.concat([train_df, valid_df])
    df = df.drop(['State_AL', 'State_NC', 'isNaN_rep_income', 'State_FL', 'State_LA',
       'isNaN_uti_card_50plus_pct', 'State_SC', 'State_GA', 'State_MS',
       'auto_open_36_month_num', 'card_open_36_month_num', 'ind_acc_XYZ'], axis=1)
    
    X = df.drop("Default_ind", axis=1).values
    y = df["Default_ind"].values
    
    split_index = [-1]*len(train_df) + [0]*len(valid_df)
    pds = PredefinedSplit(test_fold=split_index)

    classifier = RandomForestClassifier(n_jobs=-1, verbose=1)
    param_grid = {
        "n_estimators": np.arange(50, 1000, 100),
        "max_depth": np.arange(1, 20),
        "criterion": ["gini", "entropy"],
        "min_samples_split": np.arange(2,10),
        "max_features": [0.8, "sqrt", "log2"],
        "min_samples_leaf": np.arange(1, 5),
        "bootstrap": [True, False],
    }
    model = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        scoring="f1",
        n_iter=700,
        verbose=1,
        n_jobs=1,
        cv=pds,
    )

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
    with open("rf.pkl", "wb") as f:
        pickle.dump(model.best_estimator_, f)


if __name__ == "__main__":
    random_forest()
