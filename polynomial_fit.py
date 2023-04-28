import numpy, scipy, sklearn

'''
Functions for fitting data to polynomials
'''

def _2_order_poly(x, A, B):
    return A*x+B*x**2
def _3_order_poly(x, A, B, C):
    return A*x+B*x**2+C*x**3
def _4_order_poly(x, A, B, C, D):
    return A*x+B*x**2+C*x**3+D*x**4
def _5_order_poly(x, A, B, C, D, E):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5
def _6_order_poly(x, A, B, C, D, E, F):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6
def _7_order_poly(x, A, B, C, D, E, F, G):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7
def _8_order_poly(x, A, B, C, D, E, F, G, H):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8
def _9_order_poly(x, A, B, C, D, E, F, G, H, I):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9
def _10_order_poly(x, A, B, C, D, E, F, G, H, I, J):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10
def _11_order_poly(x, A, B, C, D, E, F, G, H, I, J, K):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10+K*x**11
def _12_order_poly(x, A, B, C, D, E, F, G, H, I, J, K, L):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10+K*x**11+L*x**12
def _13_order_poly(x, A, B, C, D, E, F, G, H, I, J, K, L, M):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10+K*x**11+L*x**12+M*x**13
def _14_order_poly(x, A, B, C, D, E, F, G, H, I, J, K, L, M, N):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10+K*x**11+L*x**12+M*x**13+N*x**14
def _15_order_poly(x, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10+K*x**11+L*x**12+M*x**13+N*x**14+O*x**15
def _16_order_poly(x, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P):
    return A*x+B*x**2+C*x**3+D*x**4+E*x**5+F*x**6+G*x**7+H*x**8+I*x**9+J*x**10+K*x**11+L*x**12+M*x**13+N*x**14+O*x**15+P*x**16



polynomials = {
    '2nd-order': _2_order_poly,
    '3rd-order': _3_order_poly,
    '4th-order': _4_order_poly,
    '5th-order': _5_order_poly,
    '6th-order': _6_order_poly,
    '7th-order': _7_order_poly,
    '8th-order': _8_order_poly,
    '9th-order': _9_order_poly,
    '10th-order': _10_order_poly,
    '11th-order': _11_order_poly,
    '12th-order': _12_order_poly,
    '13th-order': _13_order_poly,
    '14th-order': _14_order_poly,
    '15th-order': _15_order_poly,
    '16th-order': _16_order_poly,
}



def polynomial_fit(x_data, y_data):
    '''
    Take in x and y data and try to find an optimal polynomial to fit the data.
    '''
    p = {}
    n = len(polynomials.keys())
    r2 = numpy.zeros(n)

    for i, key in enumerate(polynomials):
        f = polynomials[key]

        covs, _ = scipy.optimize.curve_fit(f, x_data, y_data)

        r2[i] = sklearn.metrics.r2_score(y_data, f(x_data, *covs))

        p[key] = {
            'fit_coefficients': covs.tolist(),
            'r^2': r2[i],
        }

    max_r2 = r2.max()

    optimal_order = list(p.keys())[numpy.where(r2==max_r2)[0][0]]

    optimal_coeffs = p[optimal_order]['fit_coefficients']

    return lambda x: polynomials[optimal_order](x, *optimal_coeffs)