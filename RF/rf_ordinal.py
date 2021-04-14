from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from data import get_data
import numpy as np
import pickle
import hydra


def convert_to_ordinal(df, enc=None):
    x_df = df.drop(['State_AL', 'State_NC', 'isNaN_rep_income', 'State_FL', 'State_LA',
       'isNaN_uti_card_50plus_pct', 'State_SC', 'State_GA', 'State_MS',
       'auto_open_36_month_num', 'card_open_36_month_num', 'ind_acc_XYZ', 'Default_ind'], axis=1)
    y_df = df['Default_ind']

    if enc is None:
        save_enc = True
        enc = OrdinalEncoder()
        enc.fit(x_df)
    else:
        save_enc = False

    x_df = enc.transform(x_df)

    if save_enc:
        return x_df, y_df, enc
    return x_df, y_df


@hydra.main(config_path="config", config_name="config")
def random_forest(cfg):
    # Load data
    train_df, valid_df, test_df = get_data(cfg)

    train_x, train_y, enc = convert_to_ordinal(train_df)
    valid_x, valid_y = convert_to_ordinal(valid_df, enc)
    test_x, test_y = convert_to_ordinal(test_df, enc)

    fit_x = pd.concat([train_x, valid_y]).values
    fit_y = pd.concat([train_y, valid_y]).values

    test_x = test_x.values
    test_y = test_y.values

    # Below 2 lines needed for cross-validation in RandomizedSearchCV
    split_index = [-1]*len(train_x) + [0]*len(valid_x)
    pds = PredefinedSplit(test_fold=split_index)

    # Create classifier and the hyperparameter search space
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
        n_iter=1,
        verbose=1,
        n_jobs=1,
        cv=pds,
    )

    model.fit(fit_x, fit_y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
    with open("rf.pkl", "wb") as f:
        pickle.dump(model.best_estimator_, f)

    y_pred = model.best_estimator_.predict(test_x)

    print(f"f1 = {metrics.f1_score(test_y, y_pred)}")
    print(f"roc auc = {metrics.roc_auc_score(test_y, y_pred)}")

if __name__ == "__main__":
    random_forest()
