import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import inspect


def sample(X, y, target_class, sample_prop, random_state):
    """
    sample (up or down) from observations
    X: np array of features
    y: np array of outcomes (binary 0/1)
    target_class: which class to sample
    sample_prop: if > 1 will (up)sample with replacement (sample_prop-1)% of records of target class and append observations to target class
                 if < 1 will (down)sample without replacement sample_prop% of target class
                 if = 1 will return data
    random_state: seed for np.random.seed
    """
    np.random.seed(random_state)
    # split data into positive and negative class
    X_target = X[y==target_class]
    X_non_target = X[y!=target_class]
    non_target = int(not target_class)
    n_target = X_target.shape[0]
    n_non_target = X_non_target.shape[0]
    
    if sample_prop <= 0.0:
        raise ValueError("nope")
    
    # just return
    if sample_prop == 1.0:
        return X, y
       
    # upsample: sample with replacement (sample_prop-1)% of records of target class
    # and append observations to target class
    elif sample_prop > 1.0:
        n_to_sample = int(n_target * (sample_prop-1))
        X_sampled = resample(
            X_target,
            replace=True,
            n_samples=n_to_sample
        )
        # combine upsampled positive samples with all negative samples
        X_up = np.concatenate((X_target, X_sampled, X_non_target), axis=0)
        y_up = np.array([target_class] * (X_sampled.shape[0] + n_target) + [non_target] * n_non_target)
        shuffle_index = resample(list(range(X_up.shape[0])), replace = False)
        return X_up[shuffle_index], y_up[shuffle_index]
    
    # downsample: sample without replacement sample_prop% of target class
    else:
        n_to_sample = int(n_target * sample_prop)
        X_sampled = resample(
            X_target,
            replace=False,
            n_samples=n_to_sample
        )
        X_down = np.concatenate((X_sampled, X_non_target), axis=0)
        y_down = np.array([target_class] * (X_sampled.shape[0]) + [non_target] * n_non_target)
        shuffle_index = resample(list(range(X_down.shape[0])), replace = False)
        return X_down[shuffle_index], y_down[shuffle_index]
    
class SampleMixin(object):
    def fit(self, X, y, sample_weight = None):
        sample_prop = self.sample_prop
        sampling_random_state = self.sampling_random_state
        target_class = self.target_class
        X_sample, y_sample = sample(X, y, target_class, sample_prop, sampling_random_state)
        return super().fit(X_sample, y_sample, sample_weight)


def sample_clf_factory(SklearnClassifier):
    class ClassifierWithSampling(SampleMixin, SklearnClassifier):
        """ We ignore sklearn convention and pass arguments to the superclass as kwargs :) """
        def __init__(self, target_class = 1, sample_prop=0.5, sampling_random_state=1234,  **kwargs):
            self.sample_prop = sample_prop
            self.sampling_random_state = sampling_random_state
            self.target_class = target_class
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