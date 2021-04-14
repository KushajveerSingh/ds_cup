import pandas as pd
from lightgbm import LGBMClassifier
import pickle
from sklearn import metrics
import numpy as np


if __name__ == "__main__":
    with open('lgbm.pkl', 'rb') as f:
        clf, df_min, df_max = pickle.load(f)

    test_df = pd.read_csv('../processed_data/test.csv')
    test_df = test_df[['avg_card_debt', 'card_age', 'non_mtg_acc_past_due_12_months_num', 'inq_12_month_num', 'uti_card', 'Default_ind']]
    test_x = test_df.drop("Default_ind", axis=1)
    test_y = test_df["Default_ind"]
    test_x = (test_x-df_min)/(df_max-df_min)
    X = test_x.values
    y = test_y.values

    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    y_pos = y_prob[:, 1]
    precision, recall, threshold = metrics.precision_recall_curve(y, y_pos)
    fscore = (2*precision*recall) / (precision + recall)
    index = np.argmax(fscore)

    print(f"f1 = {metrics.f1_score(y, y_pred)}")
    print(f"roc auc = {metrics.roc_auc_score(y, y_pred)}")
    print(f'Best f1 score = {fscore[index]}')
    print(f'Best threshold = {threshold[index]}')
