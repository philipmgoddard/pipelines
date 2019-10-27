import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import inspect


def sample(X, y, target_ratio=1.0, upsample=True, random_state=1234):
    """
    sample (up or down) from observations
    X: np array of features
    y: np array of outcomes (binary 0/1)
    target_ratio: majority / minority proportion. If = 1, classes balanced, 0.5 = twice as many in majority class as minorty, etc.
    upsample: True if upsample minority class, False if downsample majority
    random_state: seed for np.random.seed
    """
    np.random.seed(random_state)
    n_positive_class = X[y==1].shape[0]
    n_negative_class = X[y==0].shape[0]
    n_obs = X.shape[0]
    
    majority_class = 1 if n_positive_class / n_negative_class >=1 else 0
    minority_class = int(not majority_class)
    X_majority = X[y==majority_class]
    X_minority = X[y!=majority_class]
    n_majority_class = max(n_positive_class, n_negative_class)
    n_minority_class = min(n_positive_class, n_negative_class)
    n_to_sample = int((n_majority_class - n_minority_class) * target_ratio)
    
    if upsample is False and target_ratio > 1.0:
        raise ValueError("this wont work")
        
    if upsample:
        # up sample from minority with replacement 
        X_sampled = resample(
            X_minority,
            replace=True,
            n_samples=n_to_sample
        )
        # combine upsampled minority class with all other samples
        X_new = np.concatenate((X_sampled, X_minority, X_majority), axis=0)
        print(n_minority_class + n_to_sample, n_majority_class)
        y_new = np.array([minority_class] * (n_minority_class + n_to_sample) + [majority_class] * n_majority_class)
        shuffle_index = resample(list(range(X_new.shape[0])), replace = False)
        return X_new[shuffle_index], y_new[shuffle_index]
    else:
        # down sample majority class by sampling without replacement
        X_sampled = resample(
            X_majority,
            replace=False,
            n_samples= n_to_sample
        )
        X_new = np.concatenate((X_sampled, X_minority), axis=0)
        y_new = np.array([majority_class] * (n_to_sample) + [minority_class] * n_minority_class)
        shuffle_index = resample(list(range(X_new.shape[0])), replace = False)
        return X_new[shuffle_index], y_new[shuffle_index]
    
    
class SampleMixin(object):
    def fit(self, X, y, sample_weight = None):
        sampling_random_state = self.sampling_random_state
        target_ratio = self.target_ratio
        upsample = self.upsample
        X_sample, y_sample = sample(X, y, target_ratio, upsample, sampling_random_state)
        return super().fit(X_sample, y_sample, sample_weight)


def sample_clf_factory(SklearnClassifier):
    class ClassifierWithSampling(SampleMixin, SklearnClassifier):
        """ We ignore sklearn convention and pass arguments to the superclass as kwargs :) """
        def __init__(self, target_ratio = 1.0, upsample=True, sampling_random_state=1234,  **kwargs):
            self.sampling_random_state = sampling_random_state
            self.target_ratio = target_ratio
            self.upsample = upsample
            super().__init__(**kwargs)

        """override in the class BaseEstimator to allow kwargs"""

        @classmethod
        def _get_param_names(cls):
            """Get parameter names for the estimator"""
            init = getattr(cls.__init__, 'deprecated_original', cls.__init__)

            # hmmm what to do here
            if init is object.__init__:
                # No explicit constructor to introspect
                return []

            init_signature = inspect.signature(init)
            super_init = getattr(super().__init__, 'deprecated_original', super().__init__)

            # hmmm what to do here
            if super_init is object.__init__:
                # No explicit constructor to introspect
                return []

            super_init_signature = inspect.signature(super_init)

            cls_parameters = [p.name for p in init_signature.parameters.values()
                              if p.name != 'self' and p.kind != p.VAR_KEYWORD]

            super_cls_parameters = [p.name for p in super_init_signature.parameters.values()
                                    if p.name != 'self' and p.kind != p.VAR_KEYWORD]

            return sorted(cls_parameters + super_cls_parameters)

    return ClassifierWithSampling