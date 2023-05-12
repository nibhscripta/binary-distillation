def poly_fit(x_data, y_data, deg=10):
    from numpy import zeros
    from numpy.polynomial import Polynomial
    from scipy.optimize import curve_fit

    f = lambda x, *coeffs: Polynomial([0, *coeffs])(x)

    coeffs, _ = curve_fit(f, x_data, y_data, zeros(deg))

    return Polynomial([0, *coeffs])