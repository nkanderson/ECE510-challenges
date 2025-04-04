# differential_equation_solver.py
import numpy as np
from scipy.integrate import solve_ivp

def exponential_decay(t, y, k):
    """
    ODE: dy/dt = -ky, representing exponential decay.

    Args:
        t: Time (not used directly in this simple ODE, but required by solve_ivp).
        y: The current value of y (the dependent variable).
        k: Decay constant.
    """
    return -k * y

def solve_exponential_decay(initial_condition, t_span, t_eval, k):
    """
    Solves the exponential decay ODE.

    Args:
        initial_condition: The initial value of y at t_span[0].
        t_span:  Tuple (t_start, t_end) representing the time interval.
        t_eval:  Array of time points at which to evaluate the solution.
        k: Decay constant.

    Returns:
        The solution y(t) evaluated at t_eval.
    """
    # Wrap the ODE function to pass the constant 'k'
    def ode_function(t, y):
        return exponential_decay(t, y, k)

    solution = solve_ivp(ode_function, t_span, initial_condition, t_eval=t_eval)
    return solution.y[0]  # Return only the y values

