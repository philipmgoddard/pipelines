import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from scipy import sparse

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Select columns from pandas dataframe by specifying a list of column names
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.attribute_names].values


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    '''
    Wrapper around sklearn.preprocess.LabelEncoder to perform encoding
    on multiple columns.
    WARNING: if you want to perform one-hot encoding, I really suggest
    using pandas.DataFrame.get_dummies()
    '''
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


class ZeroVariance(BaseEstimator, TransformerMixin):
    '''
    Transformer to identify zero variance and optionally low variance features
    for removal

    This works similarly to the R caret::nearZeroVariance function
    '''
    def __init__(self, near_zero=False, freq_cut=95/5, unique_cut=10):
        '''
        near zero: boolean. False: remove only zero var. True: remove near zero as well
        freq_cut: cutoff frequency ratio of most frequent to second most frequent
        unique_cut: unique cut: cutoff for percentage unique values
        '''
        self.near_zero = near_zero
        self.freq_cut = freq_cut
        self.unique_cut = unique_cut

    def fit(self, X, y=None): 
        self.zero_var = np.zeros(X.shape[1], dtype=bool)
        self.near_zero_var = np.zeros(X.shape[1], dtype=bool)
        n_obs = X.shape[0]

        for i, col in enumerate(X.T):
            # obtain values, counts of values and sort counts from
            # most to least frequent
            val_counts = np.unique(col, return_counts= True)
            counts = val_counts[1]
            counts_len = counts.shape[0]
            counts_sort = np.sort(counts)[::-1]

            # if only one value, is ZV
            if counts_len == 1:
                self.zero_var[i] = True
                self.near_zero_var[i] = True
                continue

            # ratio of most frequent / second most frequent
            freq_ratio = counts_sort[0] / counts_sort[1]
            # percent unique values
            unique_pct = (counts_len / n_obs) * 100

            if (unique_pct < self.unique_cut) and (freq_ratio > self.freq_cut):
                 self.near_zero_var[i] = True

        return self

    def transform(self, X, y=None):
        if self.near_zero:
            return X.T[~self.near_zero_var].T
        else:
            return X.T[~self.zero_var].T

    def get_feature_names(self, input_features=None):
        if self.near_zero:
            return input_features[~self.near_zero_var]
        else:
            return input_features[~self.zero_var]


class FindCorrelation(BaseEstimator, TransformerMixin):
    '''
    Remove pairwise correlations beyond threshold.
    This is not 'exact': it does not recalculate correlation
    after each step, and is therefore less expensive.

    This works similarly to the R caret::findCorrelation function
    with exact = False
    '''
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        '''
        Produce binary array for filtering columns in feature array.
        Remember to transpose the correlation matrix so is
        column major.

        Loop through columns in (n_features,n_features) correlation matrix.
        Determine rows where value is greater than threshold.
        For the candidate pairs, one must be removed. Determine which feature
        has the larger average correlation with all other features and remove it.

        Remember, matrix is symmetric so shift down by one row per column as
        iterate through.
        ''' 
        self.correlated = np.zeros(X.shape[1], dtype=bool)
        self.corr_mat =  np.corrcoef(X.T)
        abs_corr_mat = np.abs(self.corr_mat)

        for i, col in enumerate(abs_corr_mat.T):
            corr_rows = np.where(col[i+1:] > self.threshold)[0]
            avg_corr = np.mean(col)

            if len(corr_rows) > 0:
                for j in corr_rows:
                    if np.mean(abs_corr_mat.T[:, j]) > avg_corr:
                        self.correlated[j] = True
                    else:
                        self.correlated[i] = True

        return self

    def transform(self, X, y=None):
        '''
        Mask the array with the features flagged for removal
        '''
        return X.T[~self.correlated].T

    def get_feature_names(self, input_features=None):
        return input_features[~self.correlated]


class OptionalStandardScaler(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around sklearn.Preprocessing to allow scaling
    to be toggled as optional transformation
    '''
    def __init__(self, scale=True, with_mean=True, with_std=True, copy=True):
        self.scale = scale
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.scale_obj = StandardScaler(with_mean = self.with_mean,
                                       with_std = self.with_std,
                                       copy = self.copy)

    def fit(self, X, y=None):
        self.scale_obj.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if self.scale:
            return self.scale_obj.transform(X)
        else:
            return X


class ManualDropper(BaseEstimator, TransformerMixin):
    '''
    Manually specify columns to drop. Could be useful in
    the context of dropping fully correlated columns after
    creating dummy variables, etc
    '''
    def __init__(self, drop_ix=[], optional_drop_ix=None):
        self.drop_ix = drop_ix
        self.optional_drop_ix = optional_drop_ix

    def fit(self, X, y=None):
        # transformer is fully dependend on column indices provided.
        # however we can add some checking here to ensure array being
        # transformed is same number of columns as that which was
        # fitted
        # REMOVE THIS FROM HERE - is in pipeline checker

        self.n_cols = X.shape[1]
        return self

    def transform(self, X, y=None):
        if self.n_cols != X.shape[1]:
            raise ValueError('Array different n_cols to that fitted')

        self.drop_array = np.zeros(X.shape[1], dtype='bool')

        for i in self.drop_ix:
            self.drop_array[i] = True

        if self.optional_drop_ix:
            for i in self.optional_drop_ix:
                self.drop_array[i] = True
        return X.T[~self.drop_array].T

    def get_feature_names(self, input_features=None):
        return input_features[~self.drop_array]


class PipelineChecker(BaseEstimator, TransformerMixin):
    '''
    purpose: to do some error checking,
    e.g. number of columns, sense checks, missing values, extreme values etc
    perhaps store min, max, is negative, is binary, std (flag if gt max + 3 std)

    At the moment only checks to see that number of columns in train the same
    as anything else ran through the pipeline.
    '''

    def fit(self, X, y=None):
        self.n_cols = X.shape[1]
        return self

    def transform(self, X, y=None):
        if X.shape[1] != self.n_cols:
            raise ValueError('Inconsistent columns')
        return X
