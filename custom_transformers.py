import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

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

class NullsColRemoval(BaseEstimator, TransformerMixin):
    '''
    Transformer to remove columns with high percentage of null values
    '''

    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def fit(self, X, y=None):
        if X.dtype == 'object':
            self.null_bool = np.sum(pd.isnull(X), axis=0) / X.shape[0] >= self.threshold
        else:
            self.null_bool = np.sum(np.isnan(X), axis=0) / X.shape[0] >= self.threshold
        return self

    def transform(self, X, y=None):
        return X.T[~self.null_bool].T

    def get_feature_names(self, input_features=None):
        return input_features[~self.null_bool]

    def get_feature_names_drop(self, input_features=None):
        return input_features[self.null_bool]

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    '''
    Encode the categorical variables either as Label or Hot Encode.
    '''

    def __init__(self, OneHot=True, sparse=False, drop_ix = [], optional_drop_ix = None):
        self.OneHot = OneHot
        self.sparse = sparse
        self.OneHot_obj = OneHotEncoder(sparse=self.sparse)
        self.drop_ix = drop_ix
        self.optional_drop_ix = optional_drop_ix

    def fit(self, X, y=None):
        if self.OneHot:
            self.OneHot_obj.fit(X, y=None)
        return self

    def transform(self,X):
        '''
        Transforms  X using either ΟneHot or LabelEncoder().
        '''
        if self.OneHot:
            #self.one_hot_features = self.OneHot_obj.get_feature_names(self.col_names)
            return self.OneHot_obj.transform(X)

            '''
            Manually specify columns to drop: used for fully correlated columns after
            creating dummy variables.
               '''
            self.n_cols = X.shape[1]
            if self.n_cols != X.shape[1]:
                raise ValueError('Array different n_cols to that fitted')

            self.drop_array = np.zeros(X.shape[1], dtype='bool')

            for i in self.drop_ix:
                self.drop_array[i] = True

            if self.optional_drop_ix:
                for i in self.optional_drop_ix:
                    self.drop_array[i] = True
            return X

        else:
            le = LabelEncoder()
            output = np.apply_along_axis(le.fit_transform, 0, X)
            return output

    def get_feature_names(self, input_features=None):
        if self.OneHot:
            one_hot_features = self.OneHot_obj.get_feature_names_out(input_features)
            return one_hot_features
        else:
            return input_features

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

        for i, col in enumerate(np.array(X.T)):
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
            return np.array(input_features)[~self.near_zero_var]
        else:
            return np.array(input_features)[~self.zero_var]

class OptionalSimpleImputer(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around sklearn.impute SimpleImputer to allow imputation of missing values.
    '''
    def __init__(self, SimpleImpute=True, missing_values=np.nan, strategy='median', copy=True):
        self.SimpleImpute = SimpleImpute
        self.missing_values = missing_values
        self.strategy = strategy
        self.copy = copy
        self.simple_imputed_obj = SimpleImputer(missing_values = self.missing_values,
                                                strategy = self.strategy,
                                                copy = self.copy)

    def fit(self, X, y=None):
        self.simple_imputed_obj.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if self.SimpleImpute:
            return self.simple_imputed_obj.transform(X)
        else:
            return X

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

class OptionalPowerTransformer(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around sklearn.Preprocessing PowerTransformer to allow power transform featurewise to
    make data more Gaussian-like.
    '''
    def __init__(self, PowerTransform=True, method='yeo-johnson', standardize=True, copy=True):
        self.PowerTransform = PowerTransform
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.power_transformed_obj = PowerTransformer(method=self.method, standardize=self.standardize, copy=self.copy)

    def fit(self, X, y=None):
        self.power_transformed_obj.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if self.PowerTransform:
            return self.power_transformed_obj.transform(X)
        else:
            return X
        
class OptionalPCA(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around from sklearn.decomposition import PCA to construct uncorrelated features and do feature
    selection
    '''
    def __init__(self, PCATransform=True, copy=True):
        self.PCATransform = PCATransform
        self.copy = copy
        self.pca_transformed_obj = PCA(copy=self.copy)

    def fit(self, X, y=None):
        self.pca_transformed_obj.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if self.PCATransform:
            return self.pca_transformed_obj.transform(X)
        else:
            return X

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
