import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import plot_partial_dependence


if __name__ == "__main__":
    with open('rf.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_df = pd.read_csv('../processed_data/test.csv')
    test_df = test_df.drop(['State_AL', 'State_NC', 'isNaN_rep_income', 'State_FL', 'State_LA',
        'isNaN_uti_card_50plus_pct', 'State_SC', 'State_GA', 'State_MS',
        'auto_open_36_month_num', 'card_open_36_month_num', 'ind_acc_XYZ'], axis=1)
    X = test_df.drop('Default_ind', axis=1)
    cols = test_df.drop('Default_ind', axis=1).columns
    importance = clf.feature_importances_

    # Feature importance
    # plt.figure(figsize=(20, 15), dpi=300)
    # plt.tight_layout()
    # feat_importances = pd.Series(importance, index=cols).sort_values()
    # feat_importances.plot(kind='barh')
    # plt.show()

    # # Partial dependence
    # plot_partial_dependence(clf, X, [0, 1, 2, 3, 4, 5])
    # plt.show()
    # plot_partial_dependence(clf, X, [6,7,8,9,10,11], percentiles=(0,1))
    # plt.show()
    # plot_partial_dependence(clf, X, [12,13,14,15,16])
    # plt.show()
    # print(cols[8])