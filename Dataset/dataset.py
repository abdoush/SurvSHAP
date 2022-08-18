'''
Fixed dataset classes (SurvEvo + adding Eve)
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from itertools import permutations
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, dataset_file_path=None, number_of_splits=5,
                 drop_percentage=0, events_only=True, drop_feature=None,
                 random_seed=20, drop_corr_level=None):

        self.random_seed = random_seed
        self.dataset_file_path = dataset_file_path
        self.number_of_splits = number_of_splits
        self.drop_percentage = drop_percentage
        self.events_only = events_only
        self.drop_feature = drop_feature
        self.drop_corr_level = drop_corr_level
        self._load_data()
        self._get_n_splits(seed=random_seed)

        self.features_names = list(self.df.columns.drop(['T', 'E']))


        self.print_dataset_summery()

    @staticmethod
    def _get_corrolated_columns(ohdf, corr_level=0.999):
        # Dropping corrolated features
        cor = ohdf.corr()
        cor = cor[(cor >= corr_level) | (cor <= -corr_level)]
        cor = cor.dropna(thresh=2).dropna(how='all', axis=1).fillna(0)

        cols_to_drop = []
        for col in cor.columns:
            # print('Col:', col)
            # print('Dropped Cols', cols_to_drop)
            # print(not (col in cols_to_drop))
            if not (col in cols_to_drop):
                cor_cols = cor.loc[cor[col] != 0, [col]].index.to_list()
                print('Col:', col)
                print('Corrolated with:', cor_cols)
                cor_cols.remove(col)
                # print(cor_cols)
                cols_to_drop.extend(cor_cols)
        for col in ['T', 'E']:
            if col in cols_to_drop: cols_to_drop.remove(col)
        return cols_to_drop

    def get_dataset_name(self):
        pass

    def _preprocess_x(self, x_df):
        pass

    def _preprocess_y(self, y_df, normalizing_val=None):
        pass

    def _preprocess_e(self, e_df):
        pass

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        pass

    def _load_data(self):
        pass

    def get_x_dim(self):
        return self.df.shape[1]-2

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        if (x_test_df is not None) & (x_tune_df is not None):
            return x_train_df.to_numpy(), x_val_df.to_numpy(), x_test_df.to_numpy(), x_tune_df.to_numpy()
        elif x_test_df is not None:
            return x_train_df.to_numpy(), x_val_df.to_numpy(), x_test_df.to_numpy()
        else:
            return x_train_df.to_numpy(), x_val_df.to_numpy()

    def print_dataset_summery(self):
        s = 'Dataset Description =======================\n'
        s += 'Dataset Name: {}\n'.format(self.get_dataset_name())
        s += 'Dataset Shape: {}\n'.format(self.df.shape)
        s += 'Events: %.2f %%\n' % (self.df['E'].sum()*100 / len(self.df))
        s += 'NaN Values: %.2f %%\n' % (self.df.isnull().sum().sum()*100 / self.df.size)
        s += f'Events % in splits: '
        for split in self.n_splits:
            s += '{:.2f}, '.format((split["E"].mean()*100))
        s += '\n'
        s += '===========================================\n'
        print(s)
        return s

    @staticmethod
    def max_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = ((df_transformed[col]) / df_transformed[col].max()) ** powr
        return df_transformed

    @staticmethod
    def log_transform(df, cols):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = np.abs(np.log(df_transformed[col] + 1e-8))
        return df_transformed

    @staticmethod
    def power_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = df_transformed[col] ** powr
        return df_transformed

    def _get_n_splits(self, seed=20):
        k = self.number_of_splits
        train_df = self.df
        df_splits = []
        for i in range(k, 1, -1):
            train_df, test_df = train_test_split(train_df, test_size=(1 / i), random_state=seed, shuffle=True,
                                                 stratify=train_df['E'])
            df_splits.append(test_df)
            if i == 2:
                df_splits.append(train_df)
        self.n_splits = df_splits
        #return df_splits

    def get_train_val_test_from_splits(self, val_id, test_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_val_test_train_exclude_one_from_splits(self, val_id, test_id, excluded_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id, excluded_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_val_test_sampled_train_from_splits(self, val_id, test_id, frac=0.8, replace=True, seed=20):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        sampled_train_df = train_df.sample(frac=frac, replace=replace, random_state=seed)

        x_train_df, y_train_df, e_train_df = self._split_columns(sampled_train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_test_tune_from_splits(self, val_id, test_id, tune_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        tune_df = df_splits_temp[tune_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id, tune_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)
        x_tune_df, y_tune_df, e_tune_df = self._split_columns(tune_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df, x_tune_df)

        x_train, x_val, x_test, x_tune = self._preprocess_x(x_train_df), \
                                         self._preprocess_x(x_val_df), \
                                         self._preprocess_x(x_test_df), \
                                         self._preprocess_x(x_tune_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test, y_tune = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_tune_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test, e_tune = self._preprocess_e(e_train_df), \
                                         self._preprocess_e(e_val_df), \
                                         self._preprocess_e(e_test_df), \
                                         self._preprocess_e(e_tune_df)

        ye_train, ye_val, ye_test, ye_tune = np.array(list(zip(y_train, e_train))), \
                                             np.array(list(zip(y_val, e_val))), \
                                             np.array(list(zip(y_test, e_test))), \
                                             np.array(list(zip(y_tune, e_tune)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test,
                x_tune, ye_tune, y_tune, e_tune)


    @staticmethod
    def get_shuffled_pairs(x, y, e, seed=None):
        x_sh, y_sh, e_sh = shuffle(x, y, e, random_state=seed)
        y_diff = y_sh - y
        fltr = (e == 1) & (y_diff > 0)  # choose the first item in the pair to be an event
        return x[fltr], y[fltr], x_sh[fltr], y_sh[fltr], y_diff[fltr]

    def get_train_val_from_splits(self, val_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)

        self._fill_missing_values(x_train_df, x_val_df)

        x_train, x_val = self._preprocess_x(x_train_df), self._preprocess_x(x_val_df)

        x_train, x_val = self._scale_x(x_train, x_val)

        y_train, y_val = self._preprocess_y(y_train_df), self._preprocess_y(y_val_df)

        e_train, e_val = self._preprocess_e(e_train_df), self._preprocess_e(e_val_df)

        ye_train, ye_val = np.array(list(zip(y_train, e_train))), np.array(list(zip(y_val, e_val)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val)

    @staticmethod
    def _split_columns(df):
        y_df = df['T']
        e_df = df['E']
        x_df = df.drop(['T', 'E'], axis=1)
        return x_df, y_df, e_df

    def test_dataset(self):
        combs = list(permutations(range(self.number_of_splits), 2))
        for i, j in combs:
            (x_train, ye_train, y_train, e_train,
             x_val, ye_val, y_val, e_val,
             x_test, ye_test, y_test, e_test) = self.get_train_val_test_from_splits(i, j)
            assert np.isnan(x_train).sum() == 0
            assert np.isnan(x_val).sum() == 0
            assert np.isnan(x_test).sum() == 0


class Flchain(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df.drop('chapter', axis=1, inplace=True)
        df['sample.yr'] = df['sample.yr'].astype('category')
        df['flc.grp'] = df['flc.grp'].astype('category')
        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df)
        self.df = ohdf
        #return ohdf

    def get_dataset_name(self):
        return 'flchain'

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        m = x_train_df['creatinine'].median()
        x_train_df['creatinine'].fillna(m, inplace=True)
        x_val_df['creatinine'].fillna(m, inplace=True)
        if x_test_df is not None:
            x_test_df['creatinine'].fillna(m, inplace=True)
        if x_tune_df is not None:
            x_tune_df['creatinine'].fillna(m, inplace=True)

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class FlchainSub1(Flchain):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df['SigmaFLC'] = df['kappa'] + df['lambda']
        df.drop(['chapter', 'sample.yr', 'flc.grp', 'kappa', 'lambda', 'mgus'], axis=1, inplace=True)

        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        ohdf = pd.get_dummies(df)
        self.df = ohdf

    def get_dataset_name(self):
        return 'flchain_sub1'
