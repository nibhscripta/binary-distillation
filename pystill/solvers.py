def poly_fit(x_data, y_data, deg=10):
    from numpy import zeros, asarray
    from numpy.polynomial import Polynomial
    from scipy.optimize import curve_fit

    x_data = asarray(x_data)
    y_data = asarray(y_data)

    f = lambda x, *coeffs: Polynomial([0, *coeffs])(x)

    coeffs, _ = curve_fit(f, x_data, y_data, zeros(deg))

    return Polynomial([0, *coeffs])

def r2_score(y_true, y_reg) -> float:
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