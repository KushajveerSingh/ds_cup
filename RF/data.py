import hydra
import pandas as pd
from sklearn import preprocessing
import os
from pathlib import Path


def handle_columns(cfg, df, median_dict, label_encoder, is_train=False):
    # Add column for missing values
    if cfg.data.add_missing_col_rep_income:
        df['isNaN_rep_income'] = (df['rep_income'].isnull()).astype(int)
    if cfg.data.add_missing_col_uti_card_50plus_pct:
        df['isNaN_uti_card_50plus_pct'] = (df['uti_card_50plus_pct'].isnull()).astype(int)

    # Replace missing values with median
    df['rep_income'] = df['rep_income'].fillna(median_dict['rep_income'])
    df['uti_card_50plus_pct'] = df['uti_card_50plus_pct'].fillna(median_dict['uti_card_50plus_pct'])

    # Add 'isOld' col
    if cfg.data.add_isOld_col:
        df['isOld'] = (df['credit_age'] > cfg.data.isOld_value).astype(int)

    # Remove white space from "auto_open_ 36_month_num"
    df = df.rename(columns={"auto_open_ 36_month_num": "auto_open_36_month_num"})

    # One-hot encode State column
    df = pd.get_dummies(df, prefix='State', columns=['States'])

    # Label 'Default_ind' using LabelEncoder
    df.loc[:, 'Default_ind'] = label_encoder.transform(df['Default_ind'].values)

    # # Balance the training data
    if is_train:
        default_ind_1 = df[df['Default_ind'] == 1]

        default_ind_0 = df[df['Default_ind'] == 0].sample(frac=1).reset_index(drop=True)
        default_ind_0 = default_ind_0.iloc[:len(default_ind_1)]

        df = pd.concat([default_ind_0, default_ind_1])
        df = df.sample(frac=1).reset_index(drop=True)

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

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(train_df['Default_ind'].values)

        median_dict = {}
        median_dict['rep_income'] = train_df['rep_income'].median()
        median_dict['uti_card_50plus_pct'] = train_df['uti_card_50plus_pct'].median()

        train_df = handle_columns(cfg, train_df, median_dict, label_encoder, is_train=True)
        valid_df = handle_columns(cfg, valid_df, median_dict, label_encoder)
        test_df  = handle_columns(cfg, test_df,  median_dict, label_encoder)

        # Save the data
        save_dir = Path(f'{orig_cwd}/{cfg.data.save_path}')
        os.makedirs(save_dir, exist_ok=True)

        train_df.to_csv(save_dir/'train.csv', index=False)
        valid_df.to_csv(save_dir/'valid.csv', index=False)
        test_df.to_csv(save_dir/'test.csv', index=False)

        return train_df, valid_df, test_df


if __name__ == "__main__":
    get_data()
