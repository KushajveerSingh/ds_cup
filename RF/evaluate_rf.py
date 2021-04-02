import pandas as pd
import pickle
from sklearn import metrics


if __name__ == "__main__":
    with open('rf.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_df = pd.read_csv('../processed_data/test.csv')
    test_df = test_df.drop(['State_AL', 'State_NC', 'isNaN_rep_income', 'State_FL', 'State_LA',
        'isNaN_uti_card_50plus_pct', 'State_SC', 'State_GA', 'State_MS',
        'auto_open_36_month_num', 'card_open_36_month_num', 'ind_acc_XYZ'], axis=1)
    X = test_df.drop('Default_ind', axis=1).values
    y = test_df['Default_ind'].values

    y_pred = clf.predict(X)

    print(metrics.f1_score(y, y_pred))
