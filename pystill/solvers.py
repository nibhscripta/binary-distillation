def poly_fit(x_data, y_data, deg: int=10):
    from numpy import zeros, asarray
    from numpy.polynomial import Polynomial
    from scipy.optimize import curve_fit

    x_data = asarray(x_data)
    y_data = asarray(y_data)

    if x_data.ndim != 1 or y_data.ndim != 1:
        raise TypeError("Input arrays must both be 1-dimensional.")

    if x_data.shape != y_data.shape:
        raise TypeError(f"Input arrays must be of the same shape. Input shapes are {x_data.shape} and {y_data.shape}")

    # generic polynomial function
    f = lambda x, *coeffs: Polynomial([0, *coeffs])(x)

    # fit to polynomial
    # zeros(deg) - f does not have a fixed number of coefficients
    # scipy.optimize.curvefit requires that the input function
    # contain a fixed number of coefficients
    # zeros(deg) forces a polynomial of 'deg' degree by 
    # supplying a guess of the coefficients to scipy.optimize.curvefit
    coeffs, _ = curve_fit(f, x_data, y_data, zeros(deg))

    return Polynomial([0, *coeffs])

def r2_score(y_true, y_reg) -> float:
    '''
    Implemention of an algorithm for calculating the coefficient
    of determination from two arrays representing the actual data
    and data calculated from regression.

    Parameters
    ----------

    y_true: array-like
        f(x), y, data to compare regression against.

    y_reg: array-like
        f(x) values calculated from regressed function where
        f is the function which was regressed. x is the x data.

    Returns
    ----------

    float
        Coefficient of determination
    '''
    from numpy import asarray, sum, mean
    y_true = asarray(y_true)
    y_reg = asarray(y_reg)

    if y_true.ndim != 1 or y_reg.ndim != 1:
        raise TypeError("Input arrays must both be 1-dimensional.")

    if y_true.shape != y_reg.shape:
        raise TypeError(f"Input arrays must be of the same shape. Input shapes are {y_true.shape} and {y_reg.shape}")

    ss_res = sum((y_true - y_reg)**2)
    ss_tot = sum((y_true - mean(y_true))**2)

    r2 = 1 - ss_res / ss_tot

    return r2