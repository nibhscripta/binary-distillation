import typing

from dataclasses import dataclass, field

R = 8.31446261815324

@dataclass
class XYLine():
    '''
    Base class for representing vapor/liquid composition lines.

    x/y are arrays representing the liquid/vapor compositions.
    '''
    x: list = field(repr=False)
    y: list = field(repr=False)

    def __init_check__(self):
        '''
        Function which performs validation on the x/y arrays.
        '''
        from numpy import asarray
        self.x = asarray(self.x)
        self.y = asarray(self.y)

        if self.x.shape != self.y.shape:
            raise TypeError("x and y arrays must be the same shape.")

        if self.x.ndim != 1:
            raise TypeError("x and y arrays must be 1-dimensional.")

        # all composition values must be between 0 and 1
        if (self.x.min() < 0) or (self.y.min() < 0) or (self.x.max() > 1) or (self.y.max() > 1):
            raise TypeError("Vapor/Liquid composition must be between 0 and 1.")
        
    def __post_init__(self):
        self.__init_check__()


def _one_d_intersections(x1, y1, x2, y2, iterations=10, tol=1e-8):
    '''
    Find the intersections between two lines represented by a set 
    of x/y arrays.
    '''
    from numpy import interp
    from scipy.optimize import fsolve
    # interpolated function
    # equal to zero at an intersection
    f = lambda x: interp(x, x1, y1) - interp(x, x2, y2)
    # lines may have multiple intersections
    # return a list of x/y pairs representing the intersections.
    sols = []
    for i in range(iterations):
        x = fsolve(f, x1[int(x1.shape[0] * (i/iterations))])[0]
        y = interp(x, x1, y1)
        # do not append duplicate solutions
        if [x, y] not in sols:
            sols.append([x, y])
    tolerant_sols = []
    tolerant_val = None
    for i, val in enumerate(sols):
        # y values from each line at an x solution point
        y_1i = interp(val[0], x1, y1)
        y_2i = interp(val[0], x2, y2)
        # determine if a solution is tolerant
        meets_tol = abs(y_2i - y_1i) <= tol
        # a solution cannot exist outside the range of x values of either line
        within_x1 = (val[0] <= x1.max()) and (val[0] >= x1.min())
        within_x2 = (val[0] <= x2.max()) and (val[0] >= x2.min())
        # return only tolerant solutions
        not_already_found = tolerant_val not in tolerant_sols
        if meets_tol and within_x1 and within_x2 and not_already_found:
            tolerant_sols.append(val)
    return tuple(tolerant_sols)


def _is_azeotrope(x, y, tol=1e-8):
    '''
    An azeotrope will occur when an equilibrium line intersects
    the identity line except when the composition is 0 or 1.
    '''
    from numpy import array
    # identity line
    identity_x = array([0, 1])
    identity_y = identity_x
    # find intersections between the equilibrium line
    intersections = _one_d_intersections(x, y, identity_x, identity_y, tol=tol)
    if intersections == []:
        return False
    for val in intersections:
        # if intersection point is between 0 and 1s
        middle = (val[0] >= tol) and (val[0] <= (1 - tol))
        if middle:
            return val[0]
    return False


@dataclass
class EquilibriumLine(XYLine):
    '''
    A class that defines a binary vapor/liquid equilibrium line 
    for use in designing a binary distillation column. Either 
    specify vapor/liquid equilibrium data or a function that 
    represents the vapor/liquid equilibrium relationship.

    Parameters
    -----------

    x: array-like
        1-dimensional array containing the liquid compositions of a 
        binary vapor/liquid equilibrium line. Values must be between
        0 and 1.

    y: array-like
        1-dimensional array containing the vapor compositions of a 
        binary vapor/liquid equilibrium line. Values must be between
        0 and 1.

    Attributes
    -----------

    azeo_x: float
        The azeotrope composition of a binary vapor/liquid 
        equilibrium line. Defaults to None if no azeotrope
        exists.

    Example
    -----------

    >>> from pystill.equilibrium import EquilibriumLine
    >>> x = [0, 0.5, 1]
    >>> y = [0, 0.7, 1]
    >>> eq = EquilibriumLine(x, y)
    '''
    azeo_x: float = field(init=False, default=None)

    def __post_init__(self): 
        self.__init_check__()

        azeo_x = _is_azeotrope(self.x, self.y)

        if azeo_x:
            self.azeo_x = azeo_x

    def from_function(f):
        '''
        Parameters
        -----------

        f: function

        
        Returns
        -----------

        class
            Returns an EquilibriumLine object.

        Example
        -----------

        >>> from pystill.equilibrium import EquilibriumLine
        >>> f = lambda x: x**0.5
        >>> eq = EquilibriumLine.from_function(f)
        '''
        from numpy import linspace
        x = linspace(0, 1, 1000)
        y = f(x)

        return EquilibriumLine(x, y)

    def fit_curve(self, function=None, deg=10):
        '''
        Fits a function to the vapor/liquid equilibrium data
        defined by an instance of `EquilibriumLine`. Specifiying
        an objective function is optional. `EquilibriumLine.fit_curve`
        replaces the vapor/liquid equilibrium data defined by the 
        instance of `EquilibriumLine`.

        Parameters
        -----------

        function: function, Optional
            The function to which the vapor/liquid data 
            will be fit. Must be in the form f(x, *coeffs)
            where x is the liquid composition and *coeffs
            are a series of coefficients A, B, C, etc. 
            `EquilibriumLine.fit_curve` calls on 
            `scipy.optimize.curve_fit`, and so the specified
            function must be a valid input to 
            `scipy.optimize.curve_fit`. When a function is not
            specified, a polynomial is fit by default.

        deg: int, Optional
            The degree of polynomial to which the data will
            be fit. A polynomial is fit to the data when
            a function is not specified. Defaults to 10.

        Example
        -----------

        >>> from pystill.equilibrium import EquilibriumLine
        >>> x = [0, 0.5, 1]
        >>> y = [0, 0.7, 1]
        >>> eq = EquilibriumLine(x, y)
        >>> eq.fit_curve()
        >>> f = lambda x: x**0.5
        >>> eq.fit_curve(function=f)
        '''
        from numpy import linspace

        if function is not None:
            from scipy.optimize import curve_fit
            covs, _ = curve_fit(function, self.x, self.y)

            f = lambda x: function(x, *covs)
        else:
            from pystill.solvers import poly_fit
            f = poly_fit(self.x, self.y, deg)

        self.x = linspace(0, 1, 1000)
        self.y = f(self.x)


@dataclass 
class AntoineEquation():
    '''
    Define the vapor pressure relationship of a molecule
    using Antoine's equation.

    Parameters
    -----------

    A: float
        A parameter in Antoine's equation.

    B: float

        B parameter in Antoine's equation.
    C: float

        C parameter in Antoine's equation.
    
    log_type: string, Optional
        Either "ln" or "log". Determines the logarithm
        type used by the specific Antoine's Equation
        paramters. Defaults to "ln".

        A = AntoineEquation(16.5785, 3638.27, 239.500)

    Example
    -----------

    >>> from pystill.equilibrium import AntoineEquation
    >>> A = AntoineEquation(16.5785, 3638.27, 239.500)
    '''
    A: float
    B: float
    C: float
    log_type: typing.Literal["ln", "log"] = "ln"

    def P(self, T):
        '''
        Return the vapor pressure of a molecule at 
        a specific temperature. Temperature units
        must match those defined by the Antoine's
        Equation Parameters.

        Parameters
        -----------

        float
            Vapor pressures

        Example
        -----------

        >>> from pystill.equilibrium import AntoineEquation
        >>> A = AntoineEquation(16.5785, 3638.27, 239.500)
        >>> A.P(101.325)
        366.35697271372584
        '''
        if self.log_type == "ln":
            from numpy import exp
            return exp(self.A - self.B / (T + self.C))
        elif self.log_type == "log":
            return 10**(self.A - self.B / (T + self.C))


_ideal_gamma = lambda x, T: 1

@dataclass 
class RaultsLawEquilibrium():
    '''
    Class representing a binary system of molecules.

    Parameters
    -----------

    P_1: function
        Function that returns the vapor pressure of 
        the first molecule. Must be in the form
        P(T).

    P_2: function
        Function that returns the vapor pressure of 
        the second molecule. Must be in the form
        P(T).

    gamma_1: function
        Function that returns the activity coefficient
        of the first molecule. Must be in the form 
        gamma(x, T) where x is the liquid composition
        of the first molecule and T is the temperature.

    gamma_2: function
        Function that returns the activity coefficient
        of the seoond molecule. Must be in the form 
        gamma(x, T) where x is the liquid composition
        of the first molecule and T is the temperature.
    '''
    P_1: typing.Callable
    P_2: typing.Callable
    gamma_1: typing.Callable = _ideal_gamma
    gamma_2: typing.Callable = _ideal_gamma

    def _equilibrium(self, x, P):
        from scipy.optimize import fsolve

        # pressure from Rauolt's Law, P(T)
        def P_T(T):
            return x * self.gamma_1(x, T) * self.P_1(T) + (1 - x) * self.gamma_2(x, T) * self.P_2(T)

        # find where the temperature where pressure from Rauolt's Law, P(T)
        # is equal to the specified pressure
        T = fsolve(lambda T: P - P_T(T), 298)[0]
        y = x * self.gamma_1(x, T) * self.P_1(T) / P 
        return y
    
    def equilibrium_line(self, P):
        '''
        Creates an `EquilibriumLine` object from the vapor pressure
        realtionship defined by an instance of the `RaultsLawEquilibrium`
        class.

        See `EquilibriumLine`

        Parameters
        -----------

        P: float
            The pressure at which the equilibrium line will be
            evaluated. Must be in the same units that the vapor
            pressure relationships are defined.

        Returns
        -----------

        class  
            `EquilibriumLine` object 
        '''
        from numpy import linspace, zeros

        x = linspace(0, 1, 100)
        y = zeros(x.shape[0])
  
        for i in range(x.shape[0]):
            y[i] = self._equilibrium(x[i], P)

        line = EquilibriumLine(x, y)

        return line


class NRTL(RaultsLawEquilibrium):
    '''
    Class used for solving the NRTL equation for 
    a binary system. Extends `RaultsLawEquilibrium`.

    See `equilibrium.RaultsLawEquilibrium`

    Parameters
    -----------

    P_1: function
        Function that returns the vapor pressure of 
        the first molecule. Must be in the form
        P(T).

    P_2: function
        Function that returns the vapor pressure of 
        the second molecule. Must be in the form
        P(T).

    b_12: float
        b_12 parameter in the NRTL equation.

    b_21: float
        b_21 parameter in the NRTL equation.

    alpha: float
        b_12 parameter in the NRTL equation.
    '''
    b_12: float
    b_21: float
    alpha: float

    def __init__(self, P_1, P_2, b_12, b_21, alpha):
        self.P_1 = P_1
        self.P_2 = P_2
        self.b_12 = b_12
        self.b_21 = b_21
        self.alpha = alpha

        from numpy import exp

        # NRTL temperature dependent parameters
        tau_12 = lambda T: self.b_12 / R / T
        tau_21 = lambda T: self.b_21 / R / T
        # NRTL temperature independent parameters
        G_12 = lambda T: exp(-self.alpha * tau_12(T))
        G_21 = lambda T: exp(-self.alpha * tau_21(T))

        # GE/RT for the first molecule from NRTL
        ln_gamma_1 = lambda x, T: (1 - x)**2 * (tau_21(T) * (G_21(T) / (x + (1 - x) * G_21(T)))**2 + G_12(T) * tau_12(T) / ((1 - x) + x * G_12(T))**2)
        # GE/RT for the second molecule from NRTL
        ln_gamma_2 = lambda x, T: x**2 * (tau_12(T) * (G_12(T) / ((1 - x) + x * G_12(T)))**2 + G_21(T) * tau_21(T) / (x + (1 - x) * G_21(T))**2)

        # ln(GE/RT) = acticity coefficient (gamma)
        self.gamma_1 = lambda x, T: exp(ln_gamma_1(x, T))
        self.gamma_2 = lambda x, T: exp(ln_gamma_2(x, T))