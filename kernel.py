
from typing import Union, Callable, Iterable, Optional, Any, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb

class ProductKernel:

    """
    Implements a product kernel given a list of kernel functions
    """
    
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, kernel_functions: Optional[Iterable[Callable]], kernel_coefficients: Optional[Union[Iterable[float], str]] = None):
        self.X = X
        if kernel_functions is None:
            kernel_functions = self.infer_kernel_from_data(X)

        if isinstance(kernel_coefficients, str):
            if kernel_coefficients == 'silverman':
                kernel_coefficients = self.infer_kernel_coefficients_from_data(X)
            else:
                kernel_coefficients = None
        kernel = lambda x: np.average(np.exp(sum([np.log((kf((x-X[:,i])/kernel_coefficients[i]))/kernel_coefficients[i]) for i, kf in enumerate(kernel_functions)])))
        self._kernel = kernel
    
    def infer_kernel_coefficients_from_data(X: Union['pd.DataFrame',np.ndarray]) -> List[Callable]:
        types = X.dtypes
        X = np.array(X)
        coeffs = []
        for i, t in enumerate(types):
            if pd.api.types.is_integer_dtype(t):
                coeffs.append(1)
            elif pd.api.types.is_float_dtype(t):
                coeffs.append(1.06*np.var(X[:,i])*(len(X))**(-0.2)) # Silverman rule of thumb
            else:
                raise TypeError(f'{t} not supported')
        return coeffs

    def infer_kernel_from_data(X: Union['pd.DataFrame',np.ndarray]) -> List[Callable]:
        types = X.dtypes
        kfs = []
        for t in types:
            if pd.api.types.is_integer_dtype(t):
                kfs.append(lambda x: 1 if x==0 else 0)
            elif pd.api.types.is_float_dtype(t):
                kfs.append(lambda x: np.power(2*np.pi, -0.5)*np.exp(-np.power(x, 2)/2))
            else:
                raise TypeError(f'{t} not supported')
        return kfs

    def predict(self, predt: np.ndarray):
        return self._kernel(predt)
    
    def gradient(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared error for estimated probabilities.'''
        y = dtrain.get_label()
        return self.predict(predt)-y
    
    def hessian(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared error for estimated probabilities.'''
        return np.ones_like(predt)

    def __call__(self, x) -> Any:
        return self._kernel(x)