import hydra
import pandas as pd
from sklearn import preprocessing
import os
from pathlib import Path


def handle_columns(cfg, df, median_dict=None):
    # Get median values
    if median_dict is None:
        return_median_dict = True
        median_rep_income = df['rep_income'].median()
        median_uti_card_50plus_pct = df['uti_card_50plus_pct'].median()

        median_dict = {}
        median_dict['rep_income'] = median_rep_income
        median_dict['uti_card_50plus_pct'] = median_uti_card_50plus_pct
    else:
        return_median_dict = False
        median_rep_income = median_dict['rep_income']
        median_uti_card_50plus_pct = median_dict['uti_card_50plus_pct']

    # Add column for missing values
    if cfg.data.add_missing_col_rep_income:
        df['isNaN_rep_income'] = (df['rep_income'].isnull()).astype(int)
    if cfg.data.add_missing_col_uti_card_50plus_pct:
        df['isNaN_uti_card_50plus_pct'] = (df['uti_card_50plus_pct'].isnull()).astype(int)

    # Replace missing values with median
    df['rep_income'] = df['rep_income'].fillna(median_rep_income)
    df['uti_card_50plus_pct'] = df['uti_card_50plus_pct'].fillna(median_uti_card_50plus_pct)

    # Add 'isOld' col
    if cfg.data.add_isOld_col:
        df['isOld'] = (df['credit_age'] > cfg.data.isOld_value).astype(int)

    # Balance the training data
    if return_median_dict:
        default_ind_1 = df[df['Default_ind'] == 1]

        default_ind_0 = df[df['Default_ind'] == 0].sample(frac=1)
        default_ind_0 = default_ind_0.iloc[:len(default_ind_1)]

        df = pd.concat([default_ind_0, default_ind_1])
        df = df.sample(frac=1)

    if return_median_dict:
        return df, median_dict
    else:
        return df


@hydra.main(config_path='config', config_name='config')
def get_data(cfg):
    orig_cwd = hydra.utils.get_original_cwd()

    # Check if processed data already exists. If 'yes' then return it,
    # else, preprocess the data and save it
    if cfg.data.processed_data_exists:
        path = Path(f'{orig_cwd}/{cfg.data.save_path}')

        train_df = pd.read_csv(path/'train.csv')
        valid_df = pd.read_csv(path/'valid.csv')
        test_df = pd.read_csv(path/'test.csv')

        return train_df, valid_df, test_df
    else:
        path = Path(f'{orig_cwd}/{cfg.data.path}')

        train_df = pd.read_csv(path/'train.csv')
        valid_df = pd.read_csv(path/'valid.csv')
        test_df = pd.read_csv(path/'test.csv')

        # Handle missing columns
        # Replace 'rep_income' with median and (optionally) add missing column
        # Replace 'uti_card_50plus_pct` with median and (optionally) add missing column
        train_df, median_dict = handle_columns(cfg, train_df)
        valid_df = handle_columns(cfg, valid_df, median_dict)
        test_df = handle_columns(cfg, test_df, median_dict)

        # Label 'DefaultInd' using LabelEncoder
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(train_df['Default_ind'].values)
        train_df.loc[:, 'Default_ind'] = label_encoder.transform(train_df['Default_ind'].values)
        valid_df.loc[:, 'Default_ind'] = label_encoder.transform(valid_df['Default_ind'].values)
        test_df.loc[:, 'Default_ind'] = label_encoder.transform(test_df['Default_ind'].values)

        # One-hot encode 'State' column
        train_df = pd.get_dummies(train_df, prefix='State', columns=['States'])
        valid_df = pd.get_dummies(valid_df, prefix='State', columns=['States'])
        test_df = pd.get_dummies(test_df, prefix='State', columns=['States'])

        # Save the data
        save_dir = Path(f'{orig_cwd}/{cfg.data.save_path}')
        os.makedirs(save_dir, exist_ok=True)

        train_df.to_csv(save_dir/'train.csv', index=False)
        valid_df.to_csv(save_dir/'valid.csv', index=False)
        test_df.to_csv(save_dir/'test.csv', index=False)

        return train_df, valid_df, test_df


if __name__ == "__main__":
    get_data()
