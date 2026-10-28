"""Theta series with bounded tails at arbitrary nomes inside the unit disk."""
from math import ceil, comb, log, sqrt


def theta_series(q_series, *, full=False):
    """Return reduced theta2, theta3 and theta4, composed with ``q_series``.

    Bound the omitted coefficients in the native q Taylor series before
    composition. For exponent E=n(n+a), coefficient k has magnitude at most
    binom(E,k)*r**(E-k), where r bounds |q0|. The ratio of successive terms
    is bounded by ((E_next-k+1)/(E-k+1))**k*r**(2*n+a+1), decreasing with n.
    Once this ratio is below one, a geometric series bounds the entire tail.
    """
    from flint import arb, acb, acb_series, ctx

    count = ctx.cap
    k = count-1
    coeffs = q_series.coeffs()
    q0 = coeffs[0] if coeffs else acb(0)
    r = q0.abs_upper()
    if not r < 1:
        raise ValueError('theta series requires a resolved nome with abs(q) < 1')
    tolerance = arb(2)**(-ctx.prec-16)

    def tail(n, a):
        exponent = n*(n+a)
        following = (n+1)*(n+1+a)
        ratio = (arb(following-k+1)/(exponent-k+1))**k*r**(following-exponent)
        if not ratio < 1:
            return None
        return (arb(comb(exponent, k))*r**(exponent-k)/(1-ratio)).upper()

    if r == 0:
        n = max(1, ceil(sqrt(k+1)))
        tails = (arb(0), arb(0))
    else:
        # Start with an estimate, then check the actual ball bounds. E>=2k
        # makes coefficient k dominate every lower coefficient's tail.
        rate = float(-r.log())
        n = max(2, ceil(sqrt(2*k+1)), ceil(sqrt((ctx.prec+32)*log(2)/rate))+2)
        while True:
            tails = (tail(n, 0), tail(n, 1) if full else arb(0))
            if all(t is not None and t < tolerance for t in tails):
                break
            n += max(1, n//20)

    native_q = acb_series([q0, 1])
    theta2 = theta3 = theta4 = acb_series([acb(1)])
    for j in range(1, n):
        term = 2*native_q**(j*j)
        theta3 += term
        if full:
            theta2 += native_q**(j*(j+1))
            theta4 += (-1)**j*term

    def bounded_composition(series, error):
        # Real centers have real native Taylor coefficients. Keeping their
        # imaginary radius exactly zero preserves the real batch fast path.
        real_error = arb(0, error)
        ball = acb(real_error, 0 if q0.imag == 0 else real_error)
        values = series.coeffs()
        native = acb_series([(values[j] if j < len(values) else acb(0))+ball
                             for j in range(count)])
        shift = acb_series([acb(0), *coeffs[1:]])
        return native(shift) if coeffs[1:] else acb_series([native.coeffs()[0]])

    return (bounded_composition(theta2, tails[1]) if full else None,
            bounded_composition(theta3, 2*tails[0]),
            bounded_composition(theta4, 2*tails[0]) if full else None)
