def numerical_derivation(f, x):
    delta_x = 1e-10 # delta x is Limit!
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)

