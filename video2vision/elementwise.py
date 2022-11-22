from functools import partial
from math import log
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from .operators import Operator, OPERATOR_REGISTRY
from .utils import _coerce_to_2dim

__all__ = ['build_linearizer', 'ElementwiseOperator', 'Polynomial', 'PowerLaw']


class ElementwiseOperator(Operator):
    '''
    This is a superclass for an :class:`Operator` that applies an elementwise
    operation.
    '''
    def __init__(self, funcs: List[Callable]):
        '''
        Args:
            funcs (list of callables): The functions to apply, listed per band.
        '''
        self.funcs = funcs

    def apply(self, x: Dict) -> Dict:
        if len(self.funcs) != x['image'].shape[-1]:
            raise ValueError(
                f"Number of functions ({len(self.funcs)}) does not match shape"
                f" of image ({x['image'].shape})"
            )

        with _coerce_to_2dim(x):
            out = [
                func(x['image'][..., band].flatten()).astype(np.float32)
                for band, func in enumerate(self.funcs)
            ]
            x['image'] = np.stack(out, axis=1)

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts

    def apply_values(self, values: np.ndarray) -> np.ndarray:
        '''
        This is a convenience function that applies the functions to a set of
        values of shape (N, B), where N indexs the value and B indexs the band.
        '''
        return np.stack([f(s) for f, s in zip(self.funcs, values.T)], axis=1)


@OPERATOR_REGISTRY.register
class Polynomial(ElementwiseOperator):
    '''
    This applies a polynomial to each band, e.g. for linearization.
    '''
    def __init__(self,
                 polys: List[Union[np.polynomial.Polynomial, List[float]]]):
        '''
        Args:
            polys (list of :class:`numpy.polynomial.Polynomial`): The
            polynomials to apply.
        '''
        # Need to be able to pass the polynomials as coefficients to support
        # deserialization from JSON.
        if isinstance(polys[0], (np.ndarray, list, tuple)):
            polys = [np.polynomial.Polynomial(coef) for coef in polys]
        # Coerce to np.float32
        for poly in polys:
            poly.coef = poly.coef.astype(np.float32)

        super().__init__(funcs=polys)

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'polys': [p.coef.tolist() for p in self.funcs],
        }

    @classmethod
    def fit(cls, input_samples: np.ndarray, target_samples: np.ndarray,
            order: int = 3) -> np.polynomial.Polynomial:
        '''
        Fits a polynomial to a given set of inputs and desired outputs.
        '''
        # NOTE: Domain and window need to be specified explicitly or else the
        # polynomial does this internal scaling thing that makes it hard to
        # interpret the coefficients...
        return np.polynomial.Polynomial.fit(
            input_samples,
            target_samples,
            order,
            domain=[0, 255],
            window=[0, 255],
        )


class PowerLawFormula:
    '''
    This encapsulates a power law:

    .. math::

        f(x) = cb^x + d

    There is an optional shoulder component, that replaces part of the low end
    with a linear fit:

    .. math:

        f(x) = \\left{\\{\\begin{array}{ll}
            cb^x + d & x > x_0 \\\\
            c(\\lob b)b^{x_0}(x - x_0) + cb^{x_0} & x \\leq x_0
        \\end{array}\\right}
    '''
    def __init__(self, scale: float, base: float, shift: float,
                 shoulder: Optional[float] = None):
        '''
        Args:
            scale (float): Multiplier of the exponent.

            base (float): Base of the exponent.

            shift (float): Added to the result.

            shoulder (optional, float): Location of the shoulder.
        '''
        self.scale, self.base, self.shift = scale, base, shift
        self.shoulder = shoulder

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Calculate power law component. We do these operations in place
        # where possible to minimize memory use.
        pow_out = self.base**x
        pow_out *= self.scale
        pow_out += self.shift
        # Calculate linear component
        if self.shoulder is not None:
            cbx = self.scale * self.base**self.shoulder
            slope = cbx * log(self.base)
            intercept = cbx + self.shift - (self.shoulder * slope)
            linear_out = x * slope
            linear_out += intercept
            return np.where(x > self.shoulder, pow_out, linear_out)
        else:
            return pow_out

    def __repr__(self) -> str:
        return f'PowerLawFormula(scale={self.scale}, base={self.base}, ' \
               f'shift={self.shift}, shoulder={self.shoulder})'

    @classmethod
    def fit(cls, input_samples: np.ndarray, target_samples: np.ndarray,
            init_coeffs: Optional[List[float]] = None):
        '''
        Fits a power law to a given set of inputs and desired outputs.
        '''
        use_shoulder = (init_coeffs is not None) and (len(init_coeffs) == 4)

        # f needs to have explicit coefficients, no *coeffs, because the
        # curve_fit function uses signature inspection to figure out how many
        # coefficients there are.
        if use_shoulder:
            def f(x, c_0, c_1, c_2, c_3):
                return cls(c_0, c_1, c_2, c_3)(x).astype(np.float64)
            bounds = ((-np.inf, 0, -np.inf, -np.inf),
                      (np.inf, np.inf, np.inf, np.inf))
        else:
            def f(x, c_0, c_1, c_2):
                return cls(c_0, c_1, c_2, None)(x).astype(np.float64)
            bounds = ((-np.inf, 0, -np.inf,), (np.inf, np.inf, np.inf))

        # Inputs to and outputs of f need to be float64, or else optimization
        # may return incorrect results; see:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        # Also, this will raise an OptimizeWarning, because the covariance
        # cannot be estimated. We catch and suppress that.

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            # We add explicit constraints to the coefficient values to keep the
            # curve_fit algorithm from jumping into negatives.
            coeffs, _ = curve_fit(
                f,
                input_samples.T.astype(np.float64),
                target_samples.T.astype(np.float64),
                init_coeffs,
                bounds=bounds,
            )

        return cls(*coeffs)


@OPERATOR_REGISTRY.register
class PowerLaw(ElementwiseOperator):
    '''
    This applies a power law to each band, e.g. for linearization:

    .. math::
        c \\cdot b^x + d
    '''
    def __init__(self, coefs: List[Union[PowerLawFormula, List[float]]]):
        '''
        Args:
            funcs (list of list of floats): The coefficients to apply. Each
            entry is expected to be a trple of scale, base, shift.
        '''
        if isinstance(coefs[0], (np.ndarray, list, tuple)):
            coefs = [PowerLawFormula(*coef) for coef in coefs]

        super().__init__(funcs=coefs)

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'coefs': [(f.scale, f.base, f.shift) for f in self.funcs],
        }

    @classmethod
    def fit(cls, input_samples: np.ndarray, target_samples: np.ndarray,
            init_coeffs: Optional[Tuple[float, float, float]] = None) \
            -> PowerLawFormula:
        '''
        Fits a power law to a given set of inputs and desired outputs.
        '''
        return PowerLawFormula.fit(input_samples, target_samples, init_coeffs)


def build_linearizer(input_samples: np.ndarray, target_samples: np.ndarray,
                     method: str = 'poly', order: int = 3,
                     init_coeffs: Optional[Tuple[float, float, float]] = None)\
        -> ElementwiseOperator:
    '''
    Builds an :class:`ElementwiseOperator` that acts as a linearizer. That is,
    given a set of samples received by the sensors, and a set of known target
    values, this fits a function to each band that attempts to map the
    input samples to the target samples, with the aim of undoing whatever
    post-processing occurred in the camera.

    Args:
        input_samples (:class:`numpy.ndarray`): The samples from the
        sensor. This should be of shape (num_samples, num_bands).

        target_samples (:class:`numpy.ndarray`): The expected values for
        the samples. This should be of shape (num_samples, num_bands).

        method (str): The method to use. Choices: 'poly', 'power'.

        order (int): The order of the polynomial, if that method is being used.

        init_coeffs (optional, 4-tuple of float): The initial estimates for the
        coefficients used in power law fitting.
    '''
    input_samples = np.array(input_samples)
    target_samples = np.array(target_samples)
    if input_samples.shape != target_samples.shape:
        raise ValueError(
            f'Mismatch in sample shapes: {input_samples.shape} vs '
            f'{target_samples.shape}'
        )
    if input_samples.ndim != 2:
        raise ValueError(
            f'Samples must be 2-dimensional, not {input_samples.shape}'
        )

    n_bands = input_samples.shape[1]

    if method == 'poly':
        fit, cls = partial(Polynomial.fit, order=order), Polynomial
    elif method == 'power':
        fit, cls = partial(PowerLaw.fit, init_coeffs=init_coeffs), PowerLaw
    else:
        raise ValueError(f'Did not recognize method {method}')

    funcs = [
        fit(input_samples[:, band], target_samples[:, band])
        for band in range(n_bands)
    ]

    return cls(funcs)
