import numpy, scipy, sklearn, dataclasses, typing

'''
Functions for fitting data to polynomials
'''



def _evaluate_polynomials(x_data, y_data, max_deg=11):
    p = {}
    r2 = numpy.zeros(max_deg-1)
    f = lambda x, *coeffs: numpy.polynomial.Polynomial([0, *coeffs])(x)

    for i in range(max_deg-1):
        deg = i + 2

        covs, _ = scipy.optimize.curve_fit(f, x_data, y_data, numpy.zeros(deg))

        r2[i] = sklearn.metrics.r2_score(y_data, f(x_data, *covs))

        p[str(deg)] = {
            'coeffs': (0, *covs),
            'r^2': r2[i]
        }

    return p



@dataclasses.dataclass
class OptimalPolynomial():
    coeffs: list
    r2: float
    f: typing.Callable = dataclasses.field(init=False)

    def __post_init__(self):
        self.f = numpy.polynomial.Polynomial(self.coeffs)



def _select_best_polynomial(polynomials):
    r2 = []
    for key in polynomials:
        r2.append(polynomials[key]['r^2'])
    
    r2 = numpy.array(r2)
    max_r2 = r2.max()

    optimal_order = list(polynomials.keys())[numpy.where(r2==max_r2)[0][0]]

    return OptimalPolynomial(polynomials[optimal_order]['coeffs'], max_r2)



def polynomial_fit(x_data, y_data, max_deg=11):
    '''
    Take in x and y data and try to find an optimal polynomial to fit the data.
    '''
    p = _select_best_polynomial(_evaluate_polynomials(x_data, y_data))

    return p.f