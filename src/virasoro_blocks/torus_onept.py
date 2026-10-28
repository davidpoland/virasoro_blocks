"""Torus one-point Virasoro blocks: the second recursion kernel.

Implements the h-recursion for the chiral torus one-point block, the N = 1 case
of the necklace recursion of
Cho-Collier-Yin arXiv:1703.09805 Section 3 (originally Hadasz-Jaskolski-
Suchanek for N = 1).

Conventions (CCY eqs. 3.1-3.2, 3.19-3.20 specialized to N = 1):

    <Phi>_tau  =  sum_P C_{P Phi P} q^{h_P - c/24} qbar^{hbar_P - c/24}
                     * F(q, h_P, d_Phi, c) * Fbar(qbar, hbar_P, dbar_Phi, c)

with q = e^{2 pi i tau} (modular convention, matching `modular.py`) and
d_Phi the external chiral weight. F factorizes on the non-degenerate torus
character (its large-h limit):

    F(q, h, d, c) = [ prod_{n>=1} 1/(1 - q^n) ] * f(q, h, d, c),
    f = 1 + O(q),

so that q^{h - c/24} F = q^{h - (c-1)/24} eta(tau)^{-1} f, exactly the
character prefactor times the recursion object f. The recursion
((3.20) at N = 1, all a_i = 0, both necklace neighbors the same primary):

    f(q, h, d, c) = 1 + sum_{r,s >= 1} q^{rs} R_{rs}(d, c) / (h - d_{rs})
                          * f(q, d_{rs} + rs, d, c)

    R_{rs}(d, c) = A^c_{rs} * P^{rs}_c[ d ; d_{rs} ] * P^{rs}_c[ d ; d_{rs} + rs ]

(CCY eq. 4.27; the N = 1 residue involves a 3-point function with a PAIR of
null states, giving two fusion polynomials at the distinct weights d_{rs}
and d_{rs} + rs -- NOT a square. Verified against the level-1 closed form
f_1 = d(d-1)/(2h) and the Gram-matrix oracle.) A_{rs}, d_{rs}, and the
fusion polynomial P^{rs}_c are exactly the sphere kernel's
(`params.A_rs`, `params.d_rs`, `params.fusion_polynomial`).
Structural checks built into the algebra:

  * identity external d = 0: P^{rs}_c[0; d_{rs}] = 0 for all rs >= 1
    (the factor (Q - lambda_{rs} + (r-1) b + (s-1)/b)/2 vanishes), so
    f == 1 and the block reduces to the character.
  * per-order closure: the q^{rs} factor in the residue means [q^n] f(h)
    needs the shifted-weight table only through order n - rs, so the table
    fills order by order like the sphere H table.

Arithmetic: the plain kernel is generic-c only; at a singular c,
`build_torus_onept_state` (and so `torus_onept_delta_table`) routes to the exact Laurent kernel
(`build_torus_onept_laurent_state`). The plain kernel is the generic level-indexed pole recursion of
`virasoro_blocks.poles` (one sector, seed 1), filled and evaluated in python-flint with the mpmath form as
fallback. For 1 < c < 25 the Kac data is complex (unit-circle b) but f itself is
real for real (c, h, d); we accumulate in mpc when b is complex and coerce
the final coefficients back to mpf with a relative tolerance check.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

from . import poles as _poles
from .params import A_rs, b_from_c, d_rs, fusion_polynomial
from .polynomial import Polynomial
from .serialization import SerializedDerivTable
from .numeric_input import ExactRational, literal_components, literal_number, real_input_label


def _integer_control(name, value, minimum=0):
    from .requests import _integer_field

    value = _integer_field(name, value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _onept_input(value, name):
    label = literal_number(value)
    return label if isinstance(label, (complex, mp.mpc)) else real_input_label(label, name=name)


def _onept_charge(c):
    """The central charge as :func:`_onept_input` reads it, a Python float first snapped to the singular locus it
    rounds from (a float 0.7 means 7/10, 6/7 the minimal model M(6,7); :func:`singularity.snap_float_c`)."""
    from .singularity import snap_float_c

    return _onept_input(snap_float_c(c), "c")


def _check_kernel_inputs(kernel, c, d_ext, q_order, *, what="supplied kernel"):
    if (literal_components(_kernel_c(kernel)) != literal_components(c)
            or literal_components(kernel.d_ext) != literal_components(d_ext)):
        raise ValueError(f"{what} was built for different c or d_ext")
    if kernel.q_order < q_order:
        raise ValueError(f"{what} has lower q_order than requested")


def _kernel_precision(kernel):
    # Laurent evaluations round back to base_dps even after a guarded build.
    return kernel.base_dps if isinstance(kernel, TorusOnePtLaurentState) else kernel.work_dps


def _onept_exponent(kernel):
    """d_ext + 1/2, retaining the input's real and complex components."""
    from .numeric_input import _exchange_weight
    from mpmath.libmp import mpf_shift

    half = _exchange_weight(literal_number(kernel.d_ext), ExactRational(1, 2))
    if isinstance(half, mp.mpc):
        return mp.make_mpc(tuple(mpf_shift(part, 1) for part in half._mpc_))
    return 2*half


def _rs_pairs(q_order: int) -> list:
    """All (r, s) with r*s <= q_order, sorted by rs then (r, s)."""
    pairs = [
        (r, s)
        for r in range(1, q_order + 1)
        for s in range(1, q_order // r + 1)
    ]
    pairs.sort(key=lambda pair: (pair[0] * pair[1], pair))
    return pairs


def _coerce_real_coeffs(values, what: str, *, guard: bool = True) -> list:
    """Return real f-series coefficients, checking imaginary roundoff when requested.

    Complex Kac data can leave small imaginary parts in coefficients that should
    be real. The guard compares each imaginary part with
    ``10**(-max(12, dps // 2)) * max(abs(value), 1)``. The scale floor of one
    matches the leading coefficient f_0 = 1 and accommodates coefficients that
    vanish by cancellation. A larger imaginary part raises an error because it
    indicates insufficient precision or inconsistent data.

    ``guard=False`` skips this check for callers that measure the error separately,
    such as :func:`torus_onept_verify_table`.
    """
    tol = mp.mpf(10) ** (-max(12, mp.mp.dps // 2))
    out = []
    for value in values:
        value = mp.mpc(value)
        if guard and abs(value.imag) > tol * max(abs(value), 1):
            raise ValueError(
                f"{what} has a non-negligible imaginary part {mp.nstr(value.imag, 5)} (value "
                f"{mp.nstr(value.real, 5)}) for real inputs: the kernel is precision-starved (raise the "
                "precision) or its data is inconsistent"
            )
        out.append(value.real)
    return out


class _TorusOnePtFamily:
    """The torus one-point h-recursion as a `virasoro_blocks.poles` pole family: one sector, seed f = 1,
    level unit q, every pole coupling to every other."""

    name = "torus-onepoint"
    level_unit = Fraction(1)
    n_sectors = 1
    couples = None

    @staticmethod
    def seed(sector: int, order: int, field) -> list:
        out = [field(0)] * (order + 1)
        out[0] = field(1)
        return out


_TORUS_FAMILY = _TorusOnePtFamily()


@dataclass(frozen=True)
class TorusOnePtKernel:
    """Order-by-order data for the torus one-point h-recursion.

    `pairs[k] = (r, s)`; `poles[k] = d_{r,s}`; `residues[k] = R_{r,s}(d, c)`;
    `f_table[k][n] = [q^n] f(q, d_{r,s} + rs, d, c)` for n <= q_order - rs (the shifted-weight
    blocks the recursion recurses into, only as deep as the truncation reads them). `real_output` records
    whether f coefficients at real h may be coerced to mpf (real c, d). `level` is the
    `virasoro_blocks.poles.LevelTable` the evaluation runs on (flint form cached on it).
    """

    q_order: int
    b: object
    c: object
    d_ext: object
    pairs: tuple
    poles: tuple
    residues: tuple
    f_table: tuple
    real_output: bool
    level: object = None
    work_dps: int = 0


def build_torus_onept_kernel(q_order: int, *, d_ext, b=None, c=None) -> TorusOnePtKernel:
    """Build the (r, s) data and shifted-weight f table.

    Exactly one of `b`, `c` must be given. Cost is O(#pairs^2 * q_order)
    scalar operations with #pairs ~ q_order * log(q_order) (one flint matrix-vector product per level); at the
    torus crossing point q = e^{-2 pi} each q-order buys ~2.7 digits, so q_order
    stays modest (~45 at dps 120).
    """
    q_order = _integer_control("q_order", q_order)
    if (b is None) == (c is None):
        raise ValueError("pass exactly one of b, c")
    d_ext = _onept_input(d_ext, "d_ext")
    if c is not None:
        c = _onept_charge(c)
    return _build_torus_onept_kernel(q_order, d_ext=d_ext, b=b, c=c)


def _build_torus_onept_kernel(q_order, *, d_ext, b, c):
    if b is None:
        # an mpc c stays mpc (the contour kernels sample c0 + r e^{i theta}, real points included);
        # anything else via convert (Fraction / str / int / float)
        c_in = mp.convert(c)
        if not isinstance(c_in, mp.mpc):
            from .singularity import recognized_singular_locus

            locus = recognized_singular_locus(c)
            if locus is not None:
                raise ValueError(
                    f"c = {mp.nstr(c_in, 10)} is the singular locus "
                    f"[{locus}]: Kac weights collide and the generic-c torus "
                    "one-point kernel silently loses all precision there "
                    "(observed at c = 26, b^2 = 2/3). Use "
                    "build_torus_onept_laurent_state (or torus_onept_delta_table, "
                    "which routes singular c to it), or perturb c or supply b "
                    "explicitly to override."
                )
        b = b_from_c(c)
    elif not isinstance(b, (mp.mpf, mp.mpc, Fraction)):
        # Freeze an explicitly numerical parameter once; rebuilds and exact
        # membership checks must use the same value as the Kac products.
        b = mp.convert(b)
    from .params import _c_input_of_b, c_origin_of_b

    c_val = c if c is not None else literal_number(_c_input_of_b(b))

    pairs = _rs_pairs(q_order)

    def kac_data(b_value):
        pole_list, residue_list = [], []
        for r, s in pairs:
            pole = d_rs(r, s, b_value)
            P_lower = fusion_polynomial(r, s, b_value, d_ext, pole)
            P_upper = fusion_polynomial(r, s, b_value, d_ext, pole + r * s)
            residue_list.append(A_rs(r, s, b_value) * P_lower * P_upper)
            pole_list.append(pole)
        return pole_list, residue_list

    poles, residues = kac_data(b)

    # An imaginary source component is deliberate, however small. Only
    # metadata derived from an explicit b needs the computed-noise tolerance.
    complex_c = _is_complex(c_val) if c is not None or c_origin_of_b(b) is not None else (
        _is_complex(c_val) and abs(mp.im(c_val)) > mp.mpf(10) ** (-max(1, mp.mp.dps - 12)))
    real_output = not (complex_c or _is_complex(d_ext))

    # The shifted-weight table f(q, w_k), w_k = d_k + rs_k, filled level by level: f_k[n] reads f_j[n - rs_j],
    # already known because rs_j >= 1 -- the generic pole recursion (Virasoro/NS/Ramond share it).
    level_poles = [
        dict(r=r, s=s, rs=r * s, d=pole, v=pole + r * s, K=residue)
        for (r, s), pole, residue in zip(pairs, poles, residues)
    ]
    field = mp.mpc if any(isinstance(x, mp.mpc) for x in (*poles, *residues)) else mp.mpf
    seeds = {0: _TORUS_FAMILY.seed(0, q_order, field)}

    def collision(outer, inner):
        return ValueError(
            "Kac-weight collision d_{r,s} + rs == d_{r',s'} at "
            f"pairs {(outer['r'], outer['s'])} -> {(inner['r'], inner['s'])}: singular central "
            "charge; this kernel supports generic c only (use "
            "build_torus_onept_laurent_state at a singular c)"
        )

    def rebuild(prec_bits):
        # Propagate the source uncertainty through the Kac products themselves.
        from flint import acb
        from .params import c_origin_of_b
        from .polynomial import _mpf_to_acb
        from .recursion import _b_ball_from_c

        origin = c_origin_of_b(b)
        b_value = _b_ball_from_c(origin, b) if origin is not None else _mpf_to_acb(b)
        inverse = 1 / b_value
        external = _mpf_to_acb(d_ext)
        out = []
        for r, s in pairs:
            pole = ((1-r)*b_value+(1-s)*inverse)*((1+r)*b_value+(1+s)*inverse)/4
            if d_ext == 0 or d_ext == 1:
                residue = acb(0)
            else:
                lower = fusion_polynomial(r, s, b_value, external, pole)
                upper = fusion_polynomial(r, s, b_value, external, pole+r*s)
                residue = A_rs(r, s, b_value)*lower*upper
            out.append(dict(r=r, s=s, rs=r*s, d=pole, K=residue))
        return out

    from .recursion import _kac_input_bits

    collision_check = _poles._kac_collision_checker(b, c=c_origin_of_b(b))
    built = _poles.fill_level_table_flint(level_poles, q_order, seeds, field, rebuild=rebuild,
                                         collision_check=collision_check, input_bits=_kac_input_bits(b, (d_ext,)))
    if built[0] == "collision":
        raise collision(level_poles[built[1][0]], level_poles[built[1][1]])
    level = _poles.LevelTable(poles=level_poles, table=built[0], order=q_order, family=_TORUS_FAMILY,
                              flint_table=built[1], flint_kd=built[2])

    return TorusOnePtKernel(
        q_order=q_order,
        b=b,
        c=c_val,
        d_ext=d_ext,
        pairs=tuple(pairs),
        poles=tuple(poles),
        residues=tuple(residues),
        f_table=tuple(tuple(row) for row in level.table),
        real_output=real_output,
        level=level,
        work_dps=mp.mp.dps,
    )


def torus_onept_f_coeffs(kernel: TorusOnePtKernel, h, *, coerce_real: bool = True, guard_imag: bool = True) -> list:
    """[q^n] f(q, h, d, c) for n = 0..q_order at the internal weight h.

    Rational in h with simple poles at the Kac weights d_{r,s} (rs <=
    q_order). Real h with real (c, d) returns mpf coefficients (coerced with
    a relative imaginary-part guard, :func:`_coerce_real_coeffs`, when the internal Kac data is complex, i.e.
    1 < c < 25; ``guard_imag=False`` drops the imaginary parts unchecked). Pass ``coerce_real=False`` to keep raw
    mpc output -- required by the singular-c contour average, whose
    individual samples carry large (canceling) imaginary artifacts.
    """
    h_val = _onept_input(h, "h")
    if not isinstance(kernel, TorusOnePtLaurentState):
        from .conditioning import pole_loss, report

        report(pole_loss(h_val, _kernel_poles(kernel)), what="torus one-point", source=h_val)
    work = max(mp.mp.dps, kernel.work_dps)
    if work > mp.mp.dps:
        with mp.workdps(work):
            out = _onept_f_coeffs_resolved(kernel, h_val, coerce_real=coerce_real, guard_imag=guard_imag)
        return [+value for value in out]
    return _onept_f_coeffs_resolved(kernel, h_val, coerce_real=coerce_real, guard_imag=guard_imag)


def _torus_kernel_at_work(kernel):
    if kernel.work_dps >= mp.mp.dps:
        return kernel
    from .params import c_origin_of_b

    cache = getattr(kernel, '_guarded_kernels', None)
    if cache is None:
        cache = {}
        object.__setattr__(kernel, '_guarded_kernels', cache)
    work = mp.mp.dps
    if work not in cache:
        origin = c_origin_of_b(kernel.b)
        b = b_from_c(origin) if origin is not None else kernel.b
        cache[work] = build_torus_onept_kernel(kernel.q_order, d_ext=kernel.d_ext, b=b)
    return cache[work]


def _onept_f_coeffs_resolved(kernel, h_val, *, coerce_real, guard_imag):
    labels = getattr(kernel, '_pole_labels', None)
    if labels is None:
        labels = {}
        for index, (pair, pole) in enumerate(zip(kernel.pairs, _kernel_poles(kernel))):
            if kernel.level.flint_kd[0][index].is_zero():
                continue
            labels.setdefault(literal_components(pole), pair)
        object.__setattr__(kernel, '_pole_labels', labels)
    pair = labels.get(literal_components(h_val))
    if pair is None:
        from .params import c_origin_of_b
        from .numeric_input import _evaluation_pole_probes

        matches = None
        probe = complex(mp.convert(h_val))
        for index, ((r, s), pole) in enumerate(zip(kernel.pairs, _evaluation_pole_probes(kernel.poles))):
            if kernel.level.flint_kd[0][index].is_zero():
                continue
            if pole is not None and abs(probe-pole) > .02*max(1, abs(probe), abs(pole)):
                continue
            if matches is None:
                matches = _poles._kac_weight_checker(kernel.b, c=c_origin_of_b(kernel.b))
            if matches(h_val, r, s):
                pair = r, s
                break
    if pair is not None:
        raise ValueError(
            f"internal weight h = {mp.nstr(mp.mpc(mp.convert(h_val)), 8)} sits on the "
            f"Kac pole d_{pair}"
        )
    order = kernel.q_order
    seed = _TORUS_FAMILY.seed(0, order, mp.mpc)
    # The source-aware check and guard are complete. Convert once here;
    # mixed Fraction/mpc equality in the inner pole loop is very expensive.
    h_numeric = mp.mpf(h_val) if isinstance(h_val, ExactRational) else mp.convert(h_val)
    from .recursion import _kac_input_bits

    values = _poles.certified_level_series(lambda: _torus_kernel_at_work(kernel).level,
                                           h_numeric, order, seed, what='the torus one-point f coefficients',
                                           input_bits=_kac_input_bits(kernel.b, (kernel.d_ext, h_val)))
    if values is not None:
        from .polynomial import _acb_to_mpc

        coeffs = [_acb_to_mpc(value) for value in values]
    else:
        level = _torus_kernel_at_work(kernel).level
        coeffs = [mp.mpc(value) for value in _poles.eval_level_series(
            h_numeric, level.poles, level.table, order, seed, mp.mpc)]

    real_input = not _is_complex(h_val)
    if coerce_real and kernel.real_output and real_input:
        return _coerce_real_coeffs(coeffs, "torus one-point f coefficient", guard=guard_imag)
    return coeffs


def torus_onept_F(kernel: TorusOnePtKernel, h, q):
    """Numeric chiral block F(q, h, d, c) = [prod 1/(1-q^n)] * f(q, h, d, c).

    The infinite product is truncated at n = q_order, matching the series
    truncation of f (both carry O(q^{q_order+1}) tails). The caller supplies
    the prefactor q^{h - c/24} (or the reduced-character convention of
    `modular.py`).
    """
    coeffs = torus_onept_f_coeffs(kernel, h)
    q_val = mp.convert(literal_number(q))
    f_val = mp.mpf(0)
    for coeff in reversed(coeffs):
        f_val = f_val * q_val + coeff
    prod = mp.mpf(1)
    for n in range(1, kernel.q_order + 1):
        prod *= 1 - q_val**n
    return f_val / prod


# ---------------------------------------------------------------------------
# Reduced one-point tables at the crossing point tau = i
# ---------------------------------------------------------------------------
#
# The S-invariant reduced one-point function for a SCALAR external Phi
# (d = dbar = Delta_Phi / 2) is
#
#     Zhat_Phi = tau_2^{(Delta_Phi + 1)/2} |eta(tau)|^2 <Phi>_tau,
#
# whose primary contribution factorizes as (the eta factors cancel as in the reduced character)
#
#     C_{P Phi P} * tau_2^{(Delta_Phi+1)/2} * q^{hhat} f(q, h, d, c)
#                                          * qbar^{hbarhat} f(qbar, hbar, d, c)
#
# in the disk coordinates (v, vbar) of `modular.py`. Entry (m, n) at fixed
# spin j is prefactor e^{-2 pi (Delta - shift)} times a function of Delta
# that is rational up to the q_order truncation, with Kac poles at
# Delta = 2 d_{rs} - j (h side) and Delta = 2 d_{rs} + j (hbar side).

_QPOW_CACHE: dict = {}


def _qpow_matrix(lam: int, q_order: int) -> list:
    """Q[p][n] = [v^p] of (q(v)/q(0))^n scaled by q(0)^n, i.e.
    Q[p][n] = e^{-2 pi n} C_p(n): the v-Taylor coefficients of q(v)^n.

    Composing any q-series through the disk map is then a matrix multiply:
    [v^p] sum_n f_n q(v)^n = sum_n Q[p][n] f_n.
    """
    from .modular import _chiral_coeff_polys, torus_exp_base

    key = (lam, q_order, mp.mp.prec)
    cached = _QPOW_CACHE.get(key)
    if cached is not None:
        return cached
    C = _chiral_coeff_polys(lam + 1)
    base = torus_exp_base()
    base_pows = [mp.mpf(1)]
    for _ in range(q_order):
        base_pows.append(base_pows[-1] * base)
    Q = []
    for p in range(lam + 1):
        coeffs = C[p]
        row = []
        for n in range(q_order + 1):
            acc = mp.mpf(0)
            for coef in reversed(coeffs):
                acc = acc * n + coef
            row.append(base_pows[n] * acc)
        Q.append(row)
    _QPOW_CACHE[key] = Q
    return Q


def _kernel_c(kernel) -> object:
    """Central charge of a plain kernel or a Laurent state."""
    if isinstance(kernel, TorusOnePtLaurentState):
        return kernel.c0
    if kernel.c is None:
        raise ValueError("kernel carries no central charge metadata")
    return kernel.c


def _kernel_f_coeffs(kernel, h, *, guard_imag: bool = True) -> list:
    """f-series coefficients from a plain kernel (generic c) or a Laurent
    state (singular c). Uniform interface for the table builders; ``guard_imag=False`` for a caller that measures
    the precision itself (:func:`_coerce_real_coeffs`)."""
    if isinstance(kernel, TorusOnePtLaurentState):
        return torus_onept_f_coeffs_laurent(kernel, h, guard_imag=guard_imag)
    return torus_onept_f_coeffs(kernel, h, guard_imag=guard_imag)


def _flint_prec() -> int:
    """flint working bits for the current mpmath precision (the repository's convention)."""
    return int(mp.mp.dps * 3.33) + 20


def _is_complex(value) -> bool:
    return isinstance(value, (mp.mpc, complex)) and mp.im(value) != 0


def _flint_scalar(value, field):
    """An mpmath number as a flint ``arb`` (``field`` real) or ``acb``, exactly (no string round-trip)."""
    from .polynomial import _mpf_to_acb, _mpf_to_arb

    if isinstance(value, Fraction):
        from flint import acb, arb

        real = arb(value.numerator)/value.denominator
        return real if field == "arb" else acb(real)
    if field == "arb":
        return _mpf_to_arb(mp.re(value) if isinstance(value, (mp.mpc, complex)) else value)
    return _mpf_to_acb(value)


def _mp_scalar(value, field):
    from .polynomial import _acb_to_mpc, _arb_to_mpf

    if field == "arb":
        return _arb_to_mpf(value)
    value = _acb_to_mpc(value)
    return mp.mpf(value.real) if value.imag == 0 else value


_FLINT_STATIC_CACHE: dict = {}


def _flint_static(lam: int, q_order: int, field: str) -> tuple:
    """The Delta-independent matrices of the table, in flint at the current precision (cached): ``C[p][k]`` the
    a^k coefficient of the chiral prefactor's C_p(a) and ``Q[p][n]`` the q-power matrix (:func:`_qpow_matrix`)."""
    from flint import acb_mat, arb_mat

    key = (lam, q_order, field, mp.mp.prec)
    cached = _FLINT_STATIC_CACHE.get(key)
    if cached is not None:
        return cached
    from .modular import _chiral_coeff_polys

    matrix = arb_mat if field == "arb" else acb_mat
    n = lam + 1
    C = _chiral_coeff_polys(n)
    C_f = matrix([[_flint_scalar(C[p][k], field) if k < len(C[p]) else 0 for k in range(n)] for p in range(n)])
    Q = _qpow_matrix(lam, q_order)
    Q_f = matrix([[_flint_scalar(value, field) for value in row] for row in Q])
    _FLINT_STATIC_CACHE[key] = (C_f, Q_f)
    return C_f, Q_f


def _chiral_v_coeffs_flint(kernel, h, lam: int, order: int, field: str, *, guard_imag: bool):
    """[v^p] of q(v)^{hhat} f(q(v), h) for p = 0..lam as a flint (lam + 1) x 1 matrix (:func:`torus_onept_chiral_v_coeffs`).

    The prefactor e^{-2 pi hhat} C_p(hhat) is ``C`` times the powers of hhat, the f part the q-power matrix times
    the f-series, and the product their truncated convolution."""
    from flint import acb, acb_mat, arb, arb_mat

    number, matrix = (arb, arb_mat) if field == "arb" else (acb, acb_mat)
    n = lam + 1
    C_f, Q_f = _flint_static(lam, order, field)
    h_real, h_imag = literal_components(h)
    c_real, c_imag = literal_components(_kernel_c(kernel))
    # Derive both components before ball conversion: complex arithmetic
    # otherwise rounds away a tiny real offset from (c-1)/24.
    hhat = _flint_scalar(h_real - (c_real-1)/24, "arb")
    if field == "acb":
        hhat = acb(hhat, _flint_scalar(h_imag - c_imag/24, "arb"))
    powers = [number(1)]
    for _ in range(lam):
        powers.append(powers[-1] * hhat)
    pref = (C_f * matrix([[value] for value in powers])) * (-2 * number.pi() * hhat).exp()
    f_coeffs = _kernel_f_coeffs(kernel, h, guard_imag=guard_imag)[: order + 1]
    f_v = Q_f * matrix([[_flint_scalar(value, field)] for value in f_coeffs])
    return matrix([[sum((pref[i, 0] * f_v[p - i, 0] for i in range(1, p + 1)), pref[0, 0] * f_v[p, 0])]
                   for p in range(n)])


def _table_field(kernel, *values) -> str:
    """``"acb"`` when any input is complex (the kernel's c or external, whose f coefficients are then complex, or
    one of ``values``), else ``"arb"`` (real f coefficients: the kernels coerce them for real inputs)."""
    if isinstance(kernel, TorusOnePtLaurentState):
        complex_kernel = _is_complex(kernel.c0) or _is_complex(kernel.d_ext)
    else:
        complex_kernel = not kernel.real_output
    return "acb" if complex_kernel or any(_is_complex(value) for value in values) else "arb"


def torus_onept_chiral_v_coeffs(kernel, h, lam: int, *, q_order: int | None = None,
                                guard_imag: bool = True) -> list:
    """[v^p] of q(v)^{h - (c-1)/24} * f(q(v), h, d, c) for p = 0..lam.

    The reduced-character chiral factor of the one-point contribution at the
    crossing point. q(v) = exp(-2 pi (1+v)/(1-v)); the prefactor part is the
    reduced-character e^{-2 pi hhat} C_p(hhat) series, the f part composes through the
    cached q-power matrix. `kernel` is a `TorusOnePtKernel` (generic c) or a
    `TorusOnePtLaurentState` (singular c). `q_order` truncates the f-series below the kernel's own order (the
    recursion is triangular in the q-power, so a deeper kernel truncated at N is the order-N block exactly).
    Computed in flint at the current precision.
    """
    lam = _integer_control("lam", lam)
    h = _onept_input(h, "h")
    order = kernel.q_order if q_order is None else _integer_control("q_order", q_order)
    if not 0 <= order <= kernel.q_order:
        raise ValueError(f"q_order {order} outside the kernel's 0..{kernel.q_order}")
    from flint import ctx

    field = _table_field(kernel, h)
    old_prec = ctx.prec
    ctx.prec = _flint_prec()
    try:
        out = _chiral_v_coeffs_flint(kernel, h, lam, order, field, guard_imag=guard_imag)
        return [_mp_scalar(out[p, 0], field) for p in range(lam + 1)]
    finally:
        ctx.prec = old_prec


def torus_onept_table_values(
    kernel: TorusOnePtKernel, spin: int, lam: int, Delta, *, symmetrize: bool = True,
    q_order: int | None = None, guard_imag: bool = True,
) -> list:
    """Compute the reduced one-point derivative table at (Delta, j) directly.

    The output matches the layout of `SerializedDerivTable.evaluate`.

    Scalar external Phi with chiral weight d = kernel.d_ext, so
    Delta_Phi = 2 d and the tau_2 exponent is (Delta_Phi + 1)/2 = d + 1/2.
    `symmetrize` adds the (j, -j) pair as for character tables; `q_order` truncates the f-series
    (:func:`torus_onept_chiral_v_coeffs`); ``guard_imag=False`` skips the f coefficients' imaginary-part guard
    (the verified tables measure their precision against a reference instead).

    Entry (m, n) = sum_{a <= m, b <= n} T[a][b] u[m - a] w[n - b] (T the tau_2 power table, u and w the chiral
    factors): the matrix product U T W^t with U, W the lower-triangular Toeplitz matrices of u and w, in flint at the
    current precision (the table is sampled at every node of every entry's own node set, so this is the tables'
    hot loop).
    """
    from flint import acb_mat, arb_mat, ctx

    from .modular import tau2_power_table

    spin = _integer_control("spin", spin)
    lam = _integer_control("lam", lam)
    order = kernel.q_order if q_order is None else _integer_control("q_order", q_order)
    if not 0 <= order <= kernel.q_order:
        raise ValueError(f"q_order {order} outside the kernel's 0..{kernel.q_order}")
    from .numeric_input import _exchange_weight

    Delta = _onept_input(Delta, "Delta")
    h = _exchange_weight(Delta, spin)
    hbar = _exchange_weight(Delta, -spin)
    exponent = _onept_exponent(kernel)
    field = _table_field(kernel, Delta, exponent)
    matrix = arb_mat if field == "arb" else acb_mat
    n = lam + 1
    P2D = tau2_power_table(lam, exponent)
    old_prec = ctx.prec
    ctx.prec = _flint_prec()
    try:
        u = _chiral_v_coeffs_flint(kernel, h, lam, order, field, guard_imag=guard_imag)
        w = u if spin == 0 else _chiral_v_coeffs_flint(kernel, hbar, lam, order, field, guard_imag=guard_imag)
        zero = u[0, 0] * 0

        def toeplitz(vector):
            return matrix([[vector[m - a, 0] if a <= m else zero for a in range(n)] for m in range(n)])

        P_f = matrix([[_flint_scalar(value, field) for value in row] for row in P2D])
        table = toeplitz(u) * P_f * toeplitz(w).transpose()
        if symmetrize and spin != 0:
            table = table + table.transpose()
        return [[_mp_scalar(table[m, nn], field) for nn in range(n)] for m in range(n)]
    finally:
        ctx.prec = old_prec


def _halfline_distance(p) -> mp.mpf:
    """Distance from a pole to the positivity half-line x in [0, infinity).

    The selection metric: poles nearest the interpolation support dominate
    the function there and must be kept; large-|Im| strip poles (1 < c < 25)
    can have large POSITIVE real parts yet be irrelevant -- selecting by
    'rightmost Re' keeps those and destroys the interpolation at large
    Delta (observed at c = 9.4: reconstruction error growing to 1e17).
    """
    p = mp.mpc(p)
    if mp.re(p) >= 0:
        return abs(mp.im(p))
    return abs(p)


def _conjugate_grouped_poles(poles_x: list) -> list:
    """Group poles into conjugate-closed units sorted by relevance
    (ascending distance to the positivity half-line), so a keep-N cut never
    splits a pair (DampedRational enforces conjugate symmetry). Repeated
    entries (multiplicity, e.g. both chiral halves at j = 0) stay distinct.

    Each pole, last first, takes the first unused earlier entry that is its conjugate; the candidates are looked
    up by their float (Re, |Im|) (a linear scan only when that misses), not by an O(n^2) scan of the pool."""
    tol = mp.mpf(10) ** (-max(1, mp.mp.dps - 10))
    reals = []
    pairs = []
    pool = [mp.mpc(p) for p in poles_x]
    used = [False] * len(pool)

    def is_real(p):
        return abs(mp.im(p)) <= tol * max(1, abs(p))

    def key(p):
        return (float(mp.re(p)), float(abs(mp.im(p))))

    buckets: dict = {}
    for i, p in enumerate(pool):
        if not is_real(p):
            buckets.setdefault(key(p), []).append(i)
    for i in reversed(range(len(pool))):
        if used[i]:
            continue
        used[i] = True
        p = pool[i]
        if is_real(p):
            reals.append(mp.mpf(mp.re(p)))
            continue
        target, reach = mp.conj(p), tol * max(1, abs(p))

        def partner(indices):
            return next((j for j in indices if j < i and not used[j] and abs(pool[j] - target) <= reach), None)

        best = partner(buckets.get(key(p), ()))
        if best is None:
            best = partner(range(i))
        if best is None:
            raise ValueError(f"unpaired complex pole {p}")
        used[best] = True
        pairs.append((p, target))
    groups = [(_halfline_distance(p), [p]) for p in reals] + [
        (_halfline_distance(p), [p, q]) for (p, q) in pairs
    ]
    groups.sort(key=lambda g: g[0])
    return groups


def _kernel_poles(kernel, q_order: int | None = None) -> list:
    """Kac-weight pole positions d_{r,s}(c), rs <= q_order (default the kernel's own), of a plain kernel or
    Laurent state (the latter reads the t^0 coefficient = value at the singular c0)."""
    order = kernel.q_order if q_order is None else _integer_control("q_order", q_order)
    if isinstance(kernel, TorusOnePtLaurentState):
        return [p["d_const"] if p.get("d_const") is not None else p["d"].s.coeff(0)
                for p in kernel.levels.poles if p["rs"] <= order]
    cached = getattr(kernel, '_source_poles', None)
    if cached is None:
        from .params import c_origin_of_b
        from .numeric_input import exact_b_squared

        source = c_origin_of_b(kernel.b)
        c_real, c_imag = literal_components(source) if source is not None else (None, None)
        x = exact_b_squared(c_real) if c_imag == 0 else None
        cached = []
        for (r, s), pole in zip(kernel.pairs, kernel.poles):
            if c_imag == 0 and r == s:
                pole = (1-r*r)*(c_real-1)/24
            elif x is not None:
                pole = ((1-r*r)*x + 2*(1-r*s) + (1-s*s)/x)/4
            cached.append(pole)
        cached = tuple(cached)
        object.__setattr__(kernel, '_source_poles', cached)
    return [pole for (r, s), pole in zip(kernel.pairs, cached) if r*s <= order]


class TorusOnePtTableError(ValueError, NotImplementedError):
    """A one-point table could not be constructed.

    Reported as one line by the CLI. Inherits from ``NotImplementedError``
    as well as ``ValueError`` for compatibility with existing callers.
    """


def _is_singular_c(c) -> bool:
    from .singularity import recognized_singular_locus

    real, imaginary = literal_components(c)
    return imaginary == 0 and recognized_singular_locus(real) is not None


def build_torus_onept_state(q_order: int, *, d_ext, c, singular: bool | None = None, work_dps: int | None = None):
    """The one-point recursion state for (c, d_ext) at ``q_order``: the plain kernel at a generic c, the exact
    Laurent kernel (:func:`build_torus_onept_laurent_state`) at a recognized singular c (c = 1, 25, minimal
    models, the rational-b^2 dual loci), where the plain kernel's Kac weights collide.  ``singular`` forces the
    choice.  Every table, evaluation and verification path builds its kernel here."""
    q_order = _integer_control("q_order", q_order)
    if work_dps is not None:
        work_dps = _integer_control("work_dps", work_dps, mp.mp.dps)
    c = _onept_charge(c)
    d_ext = _onept_input(d_ext, "d_ext")
    if singular is None:
        singular = _is_singular_c(c)
    if singular:
        return build_torus_onept_laurent_state(q_order, d_ext=d_ext, c0=c, work_dps=work_dps)
    with mp.workdps(mp.mp.dps if work_dps is None else work_dps):
        return build_torus_onept_kernel(q_order, d_ext=d_ext, c=c)


def torus_onept_reference_order(q_order: int) -> int:
    """q-order of the verification reference: deep enough that its own truncation is far below the table's (at
    tau = i one order is a factor ~e^{-2 pi}, less the derivative growth), so ``ref(N) - ref`` measures the table's
    truncation."""
    q_order = _integer_control("q_order", q_order)
    return q_order + max(4, q_order // 6)


def torus_onept_reference_dps(dps: int) -> int:
    """Working digits of the verification reference: enough above the table's that the reference's own rounding
    (the kernel's cancellation loses the same number of digits at any precision) is ~10^-10 of the table's."""
    dps = _integer_control("dps", dps, 1)
    return dps + max(10, dps // 5)


def build_torus_onept_reference(kernel):
    """The verification reference of a one-point kernel: the same (c, d_ext) recursion, deeper in q
    (:func:`torus_onept_reference_order`) and built at more digits (:func:`torus_onept_reference_dps`).  Its
    samples are evaluated at that precision; truncated back to ``kernel.q_order`` they are the table's own block
    at more digits."""
    singular = isinstance(kernel, TorusOnePtLaurentState)
    c = kernel.c0 if singular else kernel.c
    with mp.workdps(torus_onept_reference_dps(mp.mp.dps)):
        return build_torus_onept_state(
            torus_onept_reference_order(kernel.q_order), d_ext=kernel.d_ext, c=c, singular=singular,
        )


# Multiples of the last node where a one-point table is checked past its nodes (:func:`torus_onept_check_points_x`).
_TAIL_CHECK_FACTORS = (mp.mpf(5) / 4, mp.mpf(2), mp.mpf(8), mp.mpf(64), mp.mpf(1024))


def _onept_away_from_poles(points, poles):
    """Move a node or verification point off a real pole before evaluating the raw kernel."""
    real_poles = tuple(sorted({mp.re(p) for p in poles if mp.im(p) == 0}))
    if not real_poles:
        return list(points)
    out = []
    for point in points:
        value = mp.mpf(point)
        # Only displace unresolved collisions. A fixed absolute clearance
        # would erase probes on one side of poles near the anchor or each other.
        while any(abs(value - pole) <= 32 * mp.eps * max(abs(value), abs(pole)) for pole in real_poles):
            value += max(1, abs(value)) * mp.mpf("0.0001")
        out.append(value)
    return out


def torus_onept_check_points_x(nodes, poles_x=()) -> list:
    """The x = Delta - shift where a one-point table is verified: between its nodes (``table_backends.
    _check_points_x``), on both sides of representative positive poles, and
    out to 1024 times the larger of the last node and largest positive pole.

    A consumer reads the table at every Delta >= shift (a bootstrap imposes positivity on all of them), so its
    digits are what it delivers there, not on the node range alone.  Each entry's reduced rational has the block's
    large-Delta power (:func:`torus_onept_delta_table`), so past the nodes the relative error settles to the
    constant set by the fitted leading coefficient; the farthest points measure that asymptote.  (A table fitted at
    one uniform degree was 2e3 x the block at x = 50 while recording 13 digits on its nodes.)
    """
    from .table_backends import _check_points_x

    last = max((mp.re(mp.mpc(x)) for x in nodes), default=mp.mpf(0))
    last = max(last, mp.mpf(1))
    positive = sorted({mp.re(p) for p in poles_x if mp.im(p) == 0 and mp.re(p) > 0})
    inner = list(_check_points_x(nodes, beyond_last=False))
    if positive:
        # Check both sides of representative split poles as well as the
        # remainder's node interval. Far probes cover the entire pole scale.
        last = max(last, positive[-1])
        picks = sorted({min(len(positive)-1, k * len(positive) // 8) for k in range(8)} | {len(positive)-1})
        for i in picks:
            pole = positive[i]
            gap = min(mp.mpf(1), pole,
                      pole-positive[i-1] if i else pole,
                      positive[i+1]-pole if i+1 < len(positive) else mp.mpf(1)) / 10000
            inner.extend((pole-gap, pole+gap))
    return _onept_away_from_poles(inner + [factor * last for factor in _TAIL_CHECK_FACTORS], poles_x)


def torus_onept_verify_table(table, kernel, reference, spin: int, lam: int, *, target: int | None = None,
                            q_order: int | None = None) -> dict:
    """Measure how many digits the fitted one-point ``table`` delivers, and why not more.

    Check points sit between the interpolation nodes and far past the last one (:func:`torus_onept_check_points_x`).
    At each the stored table (evaluated at twice the working precision, as a consumer would) is compared with the
    deep, high-precision ``reference`` (:func:`build_torus_onept_reference`); the three sources of error are
    separated as well:

    * ``fit``: table vs. the block sampled with ``kernel`` itself -- the order reduction (``npoles``) and the
      monomial-form cancellation;
    * ``precision``: ``kernel`` vs. the reference truncated at the table's ``q_order`` -- the recursion's cancellation at
      the working precision (large ``d_ext`` loses many digits);
    * ``truncation``: the reference truncated at ``q_order`` vs. the full reference -- the ``--order`` tail.

    ``q_order`` defaults to the kernel's order; pass the table's requested order when reusing a deeper kernel.

    Sets ``table.verified_digits`` (the overall per-entry agreement, what the table delivers) and
    ``table.verified_digits_largest_entry`` (relative to the largest entry at each point), records them with the
    three parts and the verified Delta range in ``table.metadata``, and returns that dict (its ``"verification"``
    is the :class:`~virasoro_blocks.verified_build.Verification` the shared build loop reads).
    """
    from .modular import TORUS_COORDINATE
    from .table_backends import _agreement_digits
    from .verified_build import point_agreement, table_values, verification_from_points

    target = int(mp.mp.dps if target is None else target)
    order = kernel.q_order if q_order is None else _integer_control("q_order", q_order)
    if order > kernel.q_order:
        raise ValueError("supplied kernel has lower q_order than requested")
    nodes = table.sample_points_x or ()
    _check_kernel_inputs(reference, _kernel_c(kernel), kernel.d_ext,
                         torus_onept_reference_order(order), what="supplied reference")
    shift = real_input_label(table.shift, name="shift")
    kept = [(m, n) for m in range(lam + 1) for n in range(min(m, lam - m) + 1)]
    positive = [p for p in table.pole_list_x if mp.im(p) == 0 and mp.re(p) > 0]
    # A deeper reference can have poles beyond this truncation. Avoid those
    # too without treating them as poles of the fitted block.
    reference_poles = [2 * literal_number(p) + sign * spin - shift
                       for p in _kernel_poles(reference) for sign in (1, -1)] if positive else ()
    points = _onept_away_from_poles(torus_onept_check_points_x(nodes, positive), reference_poles)
    deltas = [shift + literal_number(x) for x in points]
    # (no imaginary-part guard: the imaginary noise of complex Kac data sits in the real part too, and this
    # comparison is what measures it)
    own = [torus_onept_table_values(kernel, spin, lam, D, q_order=order, guard_imag=False) for D in deltas]
    got = table_values(table, TORUS_COORDINATE, deltas, kept, dps=2 * mp.mp.dps)
    with mp.workdps(torus_onept_reference_dps(mp.mp.dps)):
        ref_same = [torus_onept_table_values(reference, spin, lam, D, q_order=order, guard_imag=False)
                    for D in deltas]
        ref_deep = [torus_onept_table_values(reference, spin, lam, D, guard_imag=False) for D in deltas]
    ceiling = float(torus_onept_reference_dps(mp.mp.dps))

    def digits(gots, refs):
        # (an exact match only says the difference is below what the reference resolves)
        value = _agreement_digits(gots, refs, kept, target)
        return None if value is None else min(value, ceiling)

    per_point = []
    for g, r in zip(got, ref_deep):
        measured = point_agreement(g, r, kept, target)
        per_point.append(None if measured is None else (min(measured[0], ceiling), min(measured[1], ceiling)))
    n_inner = len(points) - len(_TAIL_CHECK_FACTORS)
    last = max((mp.re(mp.mpc(x)) for x in nodes), default=None)
    verification = verification_from_points(per_point, points, n_inner=n_inner, last_node=last, shift=shift)
    report = {
        "verified_digits": None if verification is None else verification.digits,
        "verified_digits_largest_entry": None if verification is None else verification.largest_digits,
        "fit_digits": digits(got, own),
        "precision_digits": digits(own, ref_same),
        "q_truncation_digits": digits(ref_same, ref_deep),
        "verified_delta_range": (shift, shift + max(points)),
    }
    table.verified_digits = report["verified_digits"]
    table.verified_digits_largest_entry = report["verified_digits_largest_entry"]
    table.metadata = {**(table.metadata or {}), **report}
    return {**report, "verification": verification}


def torus_onept_diagnose(table, report: dict, verification, *, digits: int, work: int):
    """Name what limits a one-point table short of ``digits`` (a :class:`~virasoro_blocks.verified_build.
    Verification` with its ``cause``), from the parts :func:`torus_onept_verify_table` separated: the q-series
    truncation ("order": raise --order), the recursion's cancellation ("precision"), or the fit -- whose own
    shortfall is told apart as for the sphere tables (:func:`~virasoro_blocks.verified_build.diagnose_monomial`:
    the coefficients' measured cancellation, or the kept poles).  ``detail`` lists the three parts."""
    from dataclasses import replace

    from .verified_build import diagnose_monomial, verification_margin

    parts = {name: report.get(key) for name, key in (("fit", "fit_digits"), ("precision", "precision_digits"),
                                                     ("q-truncation", "q_truncation_digits"))}
    detail = "(Parts: " + ", ".join(f"{name} {value:.1f}" for name, value in parts.items() if value is not None) + ".)"
    measured = [(value, name) for name, value in parts.items() if value is not None]
    limiting = min(measured)[1] if measured else "fit"
    if limiting == "q-truncation":
        return replace(verification, cause="order", detail=detail)
    if limiting == "precision":
        lost = float(int(work) - parts["precision"])
        return replace(verification, cause="precision", mechanism="recursion", lost=lost,
                       needed_precision=int(mp.ceil(digits + lost + verification_margin(digits))), detail=detail)
    found = diagnose_monomial(table, verification, digits=digits, work=work, order_reduced=True)
    return replace(found, detail=detail)


def kac_hit_weight(kernel, h):
    """The exact rational Kac weight ``h`` sits on, at the singular c of a Laurent state, or None (generic c, or ``h``
    off every Kac weight with rs <= the kernel's order)."""
    if not isinstance(kernel, TorusOnePtLaurentState) or isinstance(h, mp.mpc) and mp.im(h) != 0:
        return None
    from .laurent_levels import recognized_rational

    real, imaginary = literal_components(_onept_input(h, "h"))
    with mp.workdps(kernel.work_dps):
        exact = recognized_rational(real) if imaginary == 0 else None
    if exact is None:
        return None
    return next((p["d_const"] for p in kernel.levels.poles if p.get("d_const") == exact), None)


def onept_point_f_coeffs(kernel, h, q_order: int) -> list:
    """f-series coefficients of the one-point block a POINT value uses: on a Kac weight of a singular c the
    irreducible-module trace (`irreducible_onept_F_series`; neither limit of the recursion is it), else the
    kernel's.  The Delta tables keep the kernel's continuous function."""
    q_order = _integer_control("q_order", q_order)
    if q_order > kernel.q_order:
        raise ValueError("supplied kernel has lower q_order than requested")
    hit = kac_hit_weight(kernel, h)
    if hit is None:
        return _kernel_f_coeffs(kernel, h, guard_imag=False)[: q_order + 1]
    from .singularity import exact_locus_c

    c_exact = _recognized_fraction(exact_locus_c(_kernel_c(kernel), q_order=kernel.q_order))
    return _f_irr_from_F_irr(irreducible_onept_F_series(hit, c_exact, kernel.d_ext, q_order))


def _onept_point(kernel, h, hbar, spin: int, v, vbar, q_order: int):
    """The reduced one-point contribution tau_2^{d+1/2} q^{h - (c-1)/24} qbar^{hbar - (c-1)/24} f(h, q) f(hbar, qbar)
    (+ the (hbar, h) orientation for spin != 0) at the disk point (v, vbar), f truncated at ``q_order``, at the
    current precision."""
    from .modular import torus_disk_point

    _tau, _taubar, tau2, q, qbar = torus_disk_point(v, vbar)
    cshift = (_kernel_c(kernel) - 1) / 24

    def f(weight, nome):
        value = mp.mpc(0)
        # (unguarded: torus_onept_point_value measures the precision against its reference)
        for coeff in reversed(onept_point_f_coeffs(kernel, weight, q_order)):
            value = value * nome + coeff
        return value

    power = tau2 ** _onept_exponent(kernel)
    value = power * q ** (h - cshift) * qbar ** (hbar - cshift) * f(h, q) * f(hbar, qbar)
    if spin != 0:
        value += power * q ** (hbar - cshift) * qbar ** (h - cshift) * f(hbar, q) * f(h, qbar)
    return value


def torus_onept_point_value(c, d_ext, h, hbar, spin: int, v, vbar, *, q_order: int, precision: int | None = None,
                            target_digits: int | None = None, max_precision: int | None = None, kernel=None):
    """The one-point block at (v, vbar), truncated at ``q_order``, at working precision ``precision`` (default the
    current one), verified against its reference.

    The recursion cancels (a large ``d_ext`` loses tens of digits, the same number at any precision), so the value is
    compared with a deeper reference at more digits (:func:`build_torus_onept_reference`); a shortfall is reported
    (:func:`~virasoro_blocks.verified_build.run_verified_build`).  With ``target_digits`` the working precision grows
    by the measured deficit until the two agree to the target (up to ``max_precision``).  The same reference,
    untruncated, measures the ``q_order`` tail.  Returns ``(value, info)`` with ``info = {"precision_actual": the
    working precision, "verified_digits": its agreement with the reference, "q_truncation_digits": digits the
    truncation leaves}``; the value is the order-``q_order`` block, as asked.  ``kernel`` is an optional state at
    the working precision.
    """
    from .table_backends import _agreement_digits
    from .verified_build import PrecisionRequest, Verification, run_verified_build, \
        verification_margin

    spin = _integer_control("spin", spin)
    q_order = _integer_control("q_order", q_order)
    precision = _integer_control("precision", mp.mp.dps if precision is None else precision, 1)
    c, d_ext = _onept_charge(c), _onept_input(d_ext, "d_ext")
    h, hbar = _onept_input(h, "h"), _onept_input(hbar, "hbar")
    if kernel is not None:
        _check_kernel_inputs(kernel, c, d_ext, q_order)
    request = PrecisionRequest.make(precision, target_digits, max_precision)
    start = int(request.precision)

    def build(work):
        with mp.workdps(work):
            reuse = kernel is not None and work == start and _kernel_precision(kernel) >= work
            state = kernel if reuse else build_torus_onept_state(q_order, d_ext=d_ext, c=c)
            value = _onept_point(state, h, hbar, spin, v, vbar, q_order)
            reference = build_torus_onept_reference(state)
        with mp.workdps(torus_onept_reference_dps(work)):
            same = _onept_point(reference, h, hbar, spin, v, vbar, q_order)
            deep = _onept_point(reference, h, hbar, spin, v, vbar, reference.q_order)
        return value, same, deep

    def verify(built, work):
        value, same, deep = built
        digits = _agreement_digits([[[value]]], [[[same]]], [(0, 0)], request.goal)
        if digits is not None:
            digits = min(digits, float(torus_onept_reference_dps(work)))
        return Verification(digits=digits, largest_digits=digits, where_text="at this point")

    def diagnose(built, verification, work):
        from dataclasses import replace

        lost = float(int(work) - verification.digits)
        return replace(verification, cause="precision", mechanism="recursion", lost=lost,
                       needed_precision=int(mp.ceil(request.digits + lost + verification_margin(request.digits))))

    (value, same, deep), verification, work = run_verified_build(
        build, verify, request=request, what="the torus one-point value", order_reduced=False, diagnose=diagnose,
    )
    truncation = _agreement_digits([[[same]]], [[[deep]]], [(0, 0)], request.digits)
    if truncation is not None:
        # (an exact match only says the tail is below the reference's digits)
        truncation = min(truncation, float(torus_onept_reference_dps(work)))
    with mp.workdps(work):
        value = +value
    return value, {"precision_actual": work, "verified_digits": None if verification is None else verification.digits,
                   "q_truncation_digits": truncation}


def onept_point_chiral_v_coeffs(kernel, h, lam: int, q_order: int) -> list:
    """[v^p] (p <= lam) of the chiral factor q(v)^{h - (c-1)/24} f(q(v)) of a one-point POINT value: on a Kac weight
    of a singular c the irreducible-module trace (:func:`onept_point_f_coeffs`), else the kernel's
    (:func:`torus_onept_chiral_v_coeffs`)."""
    if kac_hit_weight(kernel, h) is None:
        return torus_onept_chiral_v_coeffs(kernel, h, lam, q_order=q_order, guard_imag=False)
    return _chiral_irr_v_coeffs(onept_point_f_coeffs(kernel, h, q_order), h, _kernel_c(kernel), lam)


def torus_onept_point_values(kernel, h, hbar, spin: int, lam: int, *, q_order: int) -> list:
    """The (m, n) Taylor table at the crossing point of the reduced one-point contribution of the module (h, hbar),
    paired with its mirror (hbar, h) for spin != 0 (the convention of the Delta tables), f truncated at ``q_order``,
    at the current precision: the point-table counterpart of :func:`torus_onept_table_values`, with the irreducible
    trace on a Kac weight of a singular c, as :func:`torus_onept_point_value` evaluates it."""
    from .modular import tau2_power_table

    spin = _integer_control("spin", spin)
    lam = _integer_control("lam", lam)
    n = lam + 1
    u = onept_point_chiral_v_coeffs(kernel, h, lam, q_order)
    w = u if spin == 0 else onept_point_chiral_v_coeffs(kernel, hbar, lam, q_order)
    P2D = tau2_power_table(lam, _onept_exponent(kernel))
    real = all(literal_components(_onept_input(value, "weight"))[1] == 0
               for value in (h, hbar, kernel.d_ext, _kernel_c(kernel)))
    table = []
    for m in range(n):
        row = []
        for nn in range(n):
            acc = mp.mpc(0)
            for a in range(m + 1):
                for b in range(nn + 1):
                    if P2D[a][b] != 0:
                        acc += P2D[a][b] * u[m - a] * w[nn - b]
            row.append(acc)
        table.append(row)
    if spin != 0:
        table = [[table[m][nn] + table[nn][m] for nn in range(n)] for m in range(n)]
    # real inputs give a real block (the Kac data are complex for 1 < c < 25; the verification measures the rest)
    return [[mp.re(value) if real else value for value in row] for row in table]


def torus_onept_point_table(c, d_ext, delta, spin: int, lam: int, *, q_order: int, precision: int | None = None,
                            target_digits: int | None = None, max_precision: int | None = None, kernel=None):
    """The frozen one-point point table at Delta (spin ``spin``): :func:`torus_onept_point_values` as degree-0
    numerators, ``shift = Delta``, base e^{-2 pi}, no poles, built at working precision ``precision`` and verified
    as :func:`torus_onept_point_value` is: against the same order-``q_order`` table from a deeper, higher-precision
    reference kernel (``verified_digits``, escalated for ``target_digits``), the reference itself untruncated
    measuring the order tail (``metadata["q_truncation_digits"]``).  ``metadata["regularization_policy"]`` is
    ``"irreducible"`` on a Kac weight of a singular c, ``"laurent"`` elsewhere at a singular c, else ``"direct"``.
    A Kac weight off the singular loci is a pole of the block (a ``BlockPoleError``, as for a value)."""
    from .modular import TORUS_COORDINATE, torus_exp_base
    from .numeric_input import _exchange_weight
    from .truncation import kept_pairs
    from .verified_build import (
        PrecisionRequest, Verification, point_agreement, record_verification, run_verified_build,
        verification_margin,
    )

    spin = _integer_control("spin", spin)
    lam = _integer_control("lam", lam)
    q_order = _integer_control("q_order", q_order)
    precision = _integer_control("precision", mp.mp.dps if precision is None else precision, 1)
    c, d_ext = _onept_charge(c), _onept_input(d_ext, "d_ext")
    delta = _onept_input(delta, "Delta")
    h, hbar = _exchange_weight(delta, spin), _exchange_weight(delta, -spin)
    if kernel is not None:
        _check_kernel_inputs(kernel, c, d_ext, q_order)
    request = PrecisionRequest.make(precision, target_digits, max_precision)
    start = int(request.precision)
    kept = kept_pairs(TORUS_COORDINATE, lam, basis="torus_onept_G", ell=spin)

    def build(work):
        with mp.workdps(work):
            reuse = kernel is not None and work == start and _kernel_precision(kernel) >= work
            state = kernel if reuse else build_torus_onept_state(q_order, d_ext=d_ext, c=c)
            _check_onept_point_poles(state, h, hbar)
            values = torus_onept_point_values(state, h, hbar, spin, lam, q_order=q_order)
            reference = build_torus_onept_reference(state)
        with mp.workdps(torus_onept_reference_dps(work)):
            same = torus_onept_point_values(reference, h, hbar, spin, lam, q_order=q_order)
            deep = torus_onept_point_values(reference, h, hbar, spin, lam, q_order=reference.q_order)
        return values, same, deep, state

    def verify(built, work):
        values, same, _deep, _state = built
        with mp.workdps(torus_onept_reference_dps(work)):
            # (per entry, an entry more than the asked digits below the largest compared at that level, as the
            # other point tables are checked)
            found = point_agreement(values, same, kept, request.digits)
        if found is None:
            return None
        ceiling = float(torus_onept_reference_dps(work))
        digits, largest = min(found[0], ceiling), min(found[1], ceiling)
        return Verification(digits=digits, largest_digits=largest, where_text="at its point")

    def diagnose(built, verification, work):
        from dataclasses import replace

        lost = float(int(work) - verification.digits)
        # (the irreducible trace is exact: what it loses is the rounding of the assembly, not the recursion's)
        irreducible = any(kac_hit_weight(built[3], weight) is not None for weight in (h, hbar))
        return replace(verification, cause="precision", mechanism="exact" if irreducible else "recursion", lost=lost,
                       needed_precision=int(mp.ceil(request.digits + lost + verification_margin(request.digits))))

    (values, same, deep, state), verification, work = run_verified_build(
        build, verify, request=request, what=f"spin {spin}: the torus one-point point table", order_reduced=False,
        diagnose=diagnose,
    )
    with mp.workdps(torus_onept_reference_dps(work)):
        tail = point_agreement(same, deep, kept, request.digits)
    truncation = None if tail is None else min(tail[0], float(torus_onept_reference_dps(work)))
    if kac_hit_weight(state, h) is not None or kac_hit_weight(state, hbar) is not None:
        policy = "irreducible"
    else:
        policy = "laurent" if isinstance(state, TorusOnePtLaurentState) else "direct"
    with mp.workdps(work):
        numerators = [[Polynomial([+value], field=mp.mpc if isinstance(value, mp.mpc) else mp.mpf)
                       for value in row] for row in values]
        table = SerializedDerivTable(
            shift=delta, base=torus_exp_base(), pole_list_x=(), coordinates=(TORUS_COORDINATE,),
            numerators={TORUS_COORDINATE: numerators}, sample_points_x=(mp.mpf(0),),
            metadata={"regularization_policy": policy, "q_truncation_digits": truncation},
        )
    record_verification(table, verification, work, request)
    table.q_truncation_digits = truncation
    return table


def _check_onept_point_poles(kernel, h, hbar) -> None:
    """A Kac weight is a pole of the one-point block except where the irreducible trace replaces it (a Kac weight
    of a singular c); the common :class:`~virasoro_blocks.requests.BlockPoleError` there, as for a value."""
    from .evaluation import _check_torus_onepoint_pole

    for weight, side in ((h, "holomorphic"), (hbar, "antiholomorphic")):
        if kac_hit_weight(kernel, weight) is None:
            _check_torus_onepoint_pole(kernel, weight, basis="torus_onept_G", side=side)


def _onept_rational_pole_data(kernel, order):
    """The truncated f-series as ``seed + M K / (h - d)^a``, retaining Arb error bounds.

    At a singular charge the merged columns give the surviving orders, rather
    than counting colliding Kac labels. A deeper supplied kernel is rebuilt at
    the requested order so poles beyond that truncation cannot enter the table.
    """
    from flint import acb, acb_mat

    from . import poles
    from .laurent_levels import WindowDeficit, merged_columns

    if kernel.q_order != order or _kernel_precision(kernel) < mp.mp.dps:
        kernel = build_torus_onept_state(order, d_ext=kernel.d_ext, c=_kernel_c(kernel))
    seed = [1] + [0] * order
    if isinstance(kernel, TorusOnePtLaurentState):
        while True:
            try:
                # An earlier spin can have raised this shared state's guard
                # precision. Laurent arithmetic caches its flint series, so
                # reading it at the output precision would demote those caches
                # while work_dps still described the higher-precision state.
                with mp.workdps(max(mp.mp.dps, kernel.work_dps)):
                    locations, rows, count = merged_columns(
                        kernel.levels, sector=0, couples=lambda _s, _p: True, as_acb=True,
                    )
                break
            except WindowDeficit as exc:
                _rebuild(kernel, kernel.work_dps, half=kernel.levels.half + exc.deficit)
        weights = [exact if exact is not None else loc for loc, _a, exact in locations]
        return (weights, [acb(1)] * count, [_flint_scalar(h, "acb") for h in weights],
                acb_mat(rows) if count else None, [acb(v) for v in seed],
                [a for _h, a, _exact in locations])
    active = poles._evaluation_rows(kernel.level, 0, order, None, 0)
    _d, K, d, M, seed = poles._evaluation_data(kernel.level, 0, order, seed, None, 0, active)
    weights = _kernel_poles(kernel, order)
    return [weights[j] for j in active], K, d, M, seed, [1] * len(active)


def _onept_chiral_pole_rows(data, h, lam, order, principal_order):
    """Laurent rows in Delta of the chiral v-series with ``exp(-2 pi hhat)`` removed.

    The rational f-series is composed with q(v), then multiplied by the
    polynomial C_p(hhat). The weight is ``h + (Delta - loc)/2``: negative
    Laurent powers acquire 2**a and Taylor powers acquire 2**(-a).
    """
    from flint import acb, acb_mat

    from .modular import _chiral_coeff_polys
    from .poles import laurent_rows_acb, location_tolerance

    weights, K, d, M, seed, powers = data
    tol = location_tolerance(h)
    at = [j for j, weight in enumerate(weights) if abs(mp.convert(weight) - h) <= tol]
    regular_order = principal_order - 1
    principal, regular = laurent_rows_acb(
        M, K, d, powers, at, _flint_scalar(h, "acb"), seed, principal_order, regular_order,
    )
    f_rows = {-(a + 1): [v * 2 ** (a + 1) for v in row] for a, row in enumerate(principal)}
    f_rows.update({a: [v / 2 ** a for v in row] for a, row in enumerate(regular)})
    Q = _flint_static(lam, order, "acb")[1]
    f_v = {a: Q * acb_mat([[v] for v in row]) for a, row in f_rows.items()}
    C = _chiral_coeff_polys(lam + 1)
    return f_v, C


def _onept_positive_principal_parts(kernel, spin, lam, order, shift, entries):
    """All genuine poles on x > 0 and their exact prefactor-stripped principal parts.

    This uses the same Laurent rows and polynomial recombination as the sphere
    tables. The torus-specific factors are C_p(hhat), q(v), and tau_2**(d+1/2).
    """
    from math import comb
    from flint import acb, ctx

    from .modular import tau2_power_table, torus_exp_base
    from .poles import _certified_ladder, _laurent_deficit
    from .polyderivs import _group_roots_by_location, _principal_parts_record

    def compute():
        ctx.prec = _flint_prec()
        data = _onept_rational_pole_data(kernel, order)
        weights, _K, _d, _M, _seed, powers = data
        groups = _group_roots_by_location([mp.convert(w) for w in weights], mp.eps * 2 ** 20)
        pool = [2 * loc + sign * spin for loc, idxs in groups
                for sign in (1, -1) for _ in range(max(powers[i] for i in idxs))]
        positive = [p for p in pool if mp.im(p) == 0 and mp.re(p) > shift]
        if not positive:
            return pool, [], []
        top = max(powers)
        cshift = (literal_number(_kernel_c(kernel)) - 1) / 24
        scale = _flint_scalar(2 * cshift, "acb") * (2 * acb.pi())
        scale = scale.exp()  # exp(-2pi Delta) was removed from the full block
        P = tau2_power_table(lam, _onept_exponent(kernel))
        P = [[_flint_scalar(v, "acb") for v in row] for row in P]
        cache = {}

        def half(h):
            key = mp.mpc(h)
            if key in cache:
                return cache[key]
            f_v, C = _onept_chiral_pole_rows(data, h, lam, order, top)
            hhat = _flint_scalar(h - cshift, "acb")
            out = {a: [acb(0)] * (lam + 1) for a in range(-top, top)}
            for p in range(lam + 1):
                for i in range(p + 1):
                    for k in range(i + 1):
                        coefficient = sum((_flint_scalar(C[i][j], "acb") * comb(j, k) * hhat ** (j-k)
                                           for j in range(k, len(C[i]))), acb(0)) / 2 ** k
                        for a, row in out.items():
                            if a-k in f_v:
                                row[p] += coefficient * f_v[a-k][p-i, 0]
            cache[key] = out
            return out

        parts, forced = [], []
        for loc, idxs in _group_roots_by_location(positive, mp.eps * 2 ** 20):
            left, right = half((loc + spin) / 2), half((loc - spin) / 2)
            rows = []
            for m, n in entries:
                row = []
                for a in range(1, len(idxs) + 1):
                    value = acb(0)
                    for i in range(m + 1):
                        for j in range(n + 1):
                            for p in left:
                                q = -a-p
                                if q in right:
                                    term = left[p][i] * right[q][j]
                                    if spin != 0:
                                        term += right[p][i] * left[q][j]
                                    value += P[m-i][n-j] * term
                    row.append(scale * value)
                rows.append(row)
            forced.extend([loc] * len(idxs))
            parts.append((loc, idxs, rows))
        return pool, forced, parts

    old_prec = ctx.prec
    try:
        pool, forced, parts = _certified_ladder(
            compute, lambda result, work: _laurent_deficit(
                [[[acb(1), *(v for row in rows for v in row)]] for _loc, _idx, rows in result[2]], work),
            order, "the torus one-point principal parts",
        )
        return pool, forced, _principal_parts_record(torus_exp_base(), entries, parts)
    finally:
        ctx.prec = old_prec


def torus_onept_delta_table(
    c,
    spin: int,
    lam: int,
    *,
    d_ext,
    q_order: int,
    npoles: int,
    shift=None,
    kernel=None,
    singular: bool | None = None,
    work_dps: int | None = None,
    guard_imag: bool = True,
) -> SerializedDerivTable:
    """Interpolated per-spin reduced one-point table in the standard
    serialized form: base^x * numerator(x) / prod (x - pole), x = Delta - shift.

    The one-point table has genuine Kac poles: at
    Delta = 2 d_{rs} - j (h side) and 2 d_{rs} + j (hbar side), complex for
    1 < c < 25. Every real pole at x > 0 is split off as an exact principal
    part before fitting the remainder and recombined into the stored table.
    `npoles` caps only the remaining poles (nearest-to-half-line first,
    conjugate pairs kept together); all positive poles come on top. Unkept
    remainder poles fold into interpolation error, so check convergence in
    `npoles` exactly as for the sphere tables.
    `q_order` controls the series truncation (~2.7 digits per order at tau=i).
    Entry (m, n)'s remainder is fit at degree (kept remainder poles) + m + n,
    on its own node set. Recombining the positive principal parts gives degree
    (total kept poles) + m + n, so the table keeps the block's large-Delta power
    Delta^(m + n); the node values recorded (`sample_block_values`, at the
    highest-degree entries' nodes) are each entry's own interpolant there,
    the block's for the m + n = lam entries. The default anchor is the
    unitarity bound shift = j, where the hbar = 0 Kac pole sits at x = 0
    (double at j = 0), kept in the prefactor as on the sphere.

    Singular c (c = 1, 25, minimal models, rational-b^2 dual loci) is routed
    through the exact Laurent (t)-series kernel automatically; pass
    `singular=True/False` to force. When positive poles are split at singular
    c0, merged Laurent columns determine each chiral pole's surviving order;
    coincident poles from the two chiral halves add their orders. Without
    positive poles the prefactor conservatively counts colliding Kac labels.
    """
    from .modular import TORUS_COORDINATE, torus_exp_base
    from .interpolation import DampedRational, get_sample_points, lagrange_interpolation_polynomial, \
        lagrange_matrix
    from .polynomial import _arb_to_mpf

    spin = _integer_control("spin", spin)
    lam = _integer_control("lam", lam)
    q_order = _integer_control("q_order", q_order)
    npoles = _integer_control("npoles", npoles)
    c = real_input_label(c, name="c")
    d_ext = _onept_input(d_ext, "d_ext")
    shift = real_input_label(spin if shift is None else shift, name="shift")
    if work_dps is not None:
        work_dps = _integer_control("work_dps", work_dps, mp.mp.dps)
    if kernel is not None:
        _check_kernel_inputs(kernel, c, d_ext, q_order)
    # The hbar = d_{1,1} = 0 Kac pole sits at Delta = |j|: a simple pole at spin j >= 1 (the h side's is at -j) and
    # a double one at spin 0 (both halves).  At the anchor shift = |j| it is the pole at x = 0 the sphere tables keep
    # for the identity and the conserved currents: it stays in the prefactor (so a consumer's x * block, x^2 * block
    # at spin 0, is regular there) and the lowest node is lifted off it (`get_sample_points`).  Below |j| it would
    # sit inside the positivity support.
    if shift < spin:
        raise ValueError(
            f"a spin-{spin} one-point table needs shift >= {spin} (the unitarity bound): the Kac pole at "
            f"Delta = {spin} (hbar = 0{'; double at spin 0' if spin == 0 else ''}) would lie inside the "
            "positivity support x = Delta - shift >= 0"
        )
    if kernel is None:
        kernel = build_torus_onept_state(q_order, d_ext=d_ext, c=c, singular=singular, work_dps=work_dps)

    # Kac poles in Delta from both chiral halves, shifted to x. Both halves
    # always contribute (at j = 0 they coincide: genuine multiplicity-2
    # poles, kept as repeated entries).
    entries = [(m, nn) for m in range(lam + 1) for nn in range(min(m, lam - m) + 1)]
    noise = mp.mpf(10) ** (-(mp.mp.dps - 10))
    pole_candidates = []
    for pole in _kernel_poles(kernel, q_order):
        for sign in (1, -1):
            real, imaginary = literal_components(pole)
            pole_candidates.append(mp.mpc(mp.convert(2*real + sign*spin - shift), mp.convert(2*imaginary)))
    pole_candidates = [mp.mpf(p.real) if abs(p.imag) <= noise * max(1, abs(p)) else p
                       for p in pole_candidates]
    forced_roots, principal_parts = [], None
    if any(mp.im(p) == 0 and mp.re(p) > 0 for p in pole_candidates):
        pool, forced_roots, principal_parts = _onept_positive_principal_parts(
            kernel, spin, lam, q_order, shift, entries,
        )
        # npoles selects only the remainder's poles. Every positive pole of
        # this truncation is split off, even when npoles is zero.
        pole_candidates = [p - shift for p in pool if mp.im(p) != 0 or mp.re(p) <= shift]
    groups = _conjugate_grouped_poles(pole_candidates)
    kept = []
    for _, members in groups:
        if len(kept) + len(members) > npoles:
            break
        kept.extend(members)
    kept_clean = []
    for p in kept:
        if abs(mp.im(p)) <= noise * max(1, abs(p)):
            kept_clean.append(mp.mpf(mp.re(p)))
        else:
            kept_clean.append(p)

    prefactor = DampedRational(
        constant=mp.mpf(1), poles=tuple(kept_clean), base=torus_exp_base()
    )

    # Per-entry numerator degrees, as on the sphere (`polyderivs.interpolate_delta_tables`).  Entry (m, n) is
    # base^x R_mn(x) with R_mn ~ x^(m + n) at large x: [v^p] of the chiral factor is a degree-p polynomial in the
    # weight (the q^hhat part; the f-series coefficients fall off like 1/h) and the tau_2 power table does not
    # depend on Delta.  So R_mn * prod_kept (x - pole) is a polynomial of degree K + m + n (K = the kept poles) up to
    # the unkept poles' 1/(x - pole) tails. Fit each entry at that degree on its
    # own Bohr-Sommerfeld set of K + m + n + 1 nodes to preserve the tail power.
    counts = sorted({len(kept_clean) + m + nn + 1 for m, nn in entries})
    # The node density may place a genuine, tiny negative pole at zero.
    # Keep the actual pole in all scalings and serialized physics data.
    node_prefactor = DampedRational(
        constant=prefactor.constant, base=prefactor.base,
        poles=tuple(mp.mpf(0) if mp.im(p) == 0 and -noise < mp.re(p) < 0 else p for p in prefactor.poles),
    )
    nodes_by_count = {count: _onept_away_from_poles(
        get_sample_points(node_prefactor, count), [p - shift for p in forced_roots],
    ) for count in counts}
    # The numeric table at every node set, and the reduced values (the prefactor divided out) the fits interpolate.
    # (a deeper supplied kernel is truncated to q_order: the recursion is triangular in the q-power, so that is the
    # order-q_order block the table records, not a deeper one)
    sampled_by_count, pref_by_count = {}, {}
    for count, nodes_x in nodes_by_count.items():
        sampled_by_count[count] = [
            torus_onept_table_values(kernel, spin, lam, literal_number(x) + shift,
                                    q_order=q_order, guard_imag=guard_imag)
            for x in nodes_x
        ]
        if principal_parts is not None:
            from .polyderivs import _principal_part_values_many

            parts = _principal_part_values_many(principal_parts, [literal_number(x) + shift for x in nodes_x], entries)
            for values, principal in zip(sampled_by_count[count], parts):
                for m, nn in entries:
                    values[m][nn] -= principal[m, nn]
        pref_by_count[count] = [prefactor.evaluate(x) for x in nodes_x]

    def reduced(count, m, nn):
        return [value[m][nn] / pref for value, pref in zip(sampled_by_count[count], pref_by_count[count])]

    n = lam + 1
    rows = [[None] * (min(m, lam - m) + 1) for m in range(n)]
    for m, nn in entries:
        count = len(kept_clean) + m + nn + 1
        rows[m][nn] = lagrange_interpolation_polynomial(nodes_by_count[count], reduced(count, m, nn), shift=0)

    # The writer nodes are the highest-degree entries' (m + n = lam), where the recorded values are the block's.  A
    # lower entry's value there is ITS interpolant's, formed by Lagrange products from its own node values (never the
    # monomial form): a consumer interpolating the node values then rebuilds that entry's own degree, not a
    # degree-(K + lam) fit through the block's values with the over-fitted tail.
    writer_count = counts[-1]
    writer_x = nodes_by_count[writer_count]
    sampled = [[list(row) for row in table] for table in sampled_by_count[writer_count]]
    for count in counts[:-1]:
        basis = lagrange_matrix(nodes_by_count[count], writer_x)
        basis = basis if isinstance(basis, list) else [[_arb_to_mpf(w) for w in row] for row in basis.tolist()]
        for m, nn in entries:
            if len(kept_clean) + m + nn + 1 != count:
                continue
            values = reduced(count, m, nn)
            for k in range(len(writer_x)):
                sampled[k][m][nn] = pref_by_count[writer_count][k] * mp.fdot(basis[k], values)

    if principal_parts is not None:
        from .polyderivs import InterpolatedDeltaTable, _principal_part_values_many, _recombine_principal_parts

        remainder = InterpolatedDeltaTable(
            prefactor=prefactor, shift=shift, sample_points=tuple(x + shift for x in writer_x), numerators=rows,
        )
        combined = _recombine_principal_parts(remainder, principal_parts, forced_roots)
        rows, kept_clean = combined.numerators, combined.prefactor.poles
        parts = _principal_part_values_many(principal_parts, [literal_number(x) + shift for x in writer_x], entries)
        for values, principal in zip(sampled, parts):
            for m, nn in entries:
                values[m][nn] += principal[m, nn]

    table = SerializedDerivTable(
        shift=shift,
        base=torus_exp_base(),
        pole_list_x=tuple(kept_clean),
        coordinates=(TORUS_COORDINATE,),
        numerators={TORUS_COORDINATE: rows},
        sample_points_x=tuple(writer_x),
        # the node values (the SDP data; never re-evaluate the polynomial there)
        sample_block_values={TORUS_COORDINATE: sampled},
        sample_via="numeric",
        working_dps=mp.mp.dps,
    )
    # the remainder poles a larger npoles could keep: a chosen count stops growing there (at c <= 1 the positive
    # poles, always kept, can leave only a few)
    table.remainder_pool = len(pole_candidates)
    return table


# ---------------------------------------------------------------------------
# Singular-c evaluation: discrete Cauchy (contour) average in the c-plane
# ---------------------------------------------------------------------------
#
# At the recognized singular loci (c = 1, c = 25, minimal models, rational
# b^2 dual-branch points like c = 26) the KERNEL's intermediate data blows up
# (A_rs poles, Kac-weight collisions w_k - d_j -> 0) while the block itself,
# at generic internal weight h, is analytic in c. The finite-order Laurent
# artifacts of the representation are therefore killed exactly by a discrete
# Cauchy average over a small circle around c0:
#
#     (1/K) sum_k f(c0 + r e^{2 pi i k / K})  =  f(c0) + O(r^K)
#
# provided the artifact pole order is < K (the c-plane analogue of the
# sphere pipeline's eps-twin evaluation). Cost: K kernel builds. Roundoff:
# each sample loses ~p * log10(1/r) digits to cancellation (p = artifact
# order), so the default radius balances r^K truncation against amplified
# roundoff at the working dps.


@dataclass(frozen=True)
class TorusOnePtContourKernels:
    """K kernels on a circle around a singular c0, plus combination data."""

    c0: object
    d_ext: object
    q_order: int
    radius: object
    kernels: tuple


def build_torus_onept_contour_kernels(
    q_order: int, *, d_ext, c0, radius=None, n_points: int = 8
) -> TorusOnePtContourKernels:
    """Kernels at c = c0 + radius * e^{2 pi i k / n_points}, k = 0..K-1.

    `radius` defaults to 10^(-dps/12): with K = 8 and artifact pole orders
    well below K this leaves ~2/3 of the working digits in the average.
    """
    q_order = _integer_control("q_order", q_order)
    n_points = _integer_control("n_points", n_points, 4)
    c0 = _onept_input(c0, "c0")
    d_ext = _onept_input(d_ext, "d_ext")
    c0 = mp.convert(c0)
    if radius is None:
        radius = mp.mpf(10) ** (-max(2, mp.mp.dps // 12))
    radius = mp.convert(real_input_label(radius, name="radius"))
    if radius <= 0:
        raise ValueError("radius must be > 0")
    kernels = []
    for k in range(n_points):
        phase = mp.expjpi(2 * mp.mpf(k) / n_points)
        c_k = c0 + radius * phase
        # complex c bypasses the singular-locus guard by construction
        kernels.append(
            build_torus_onept_kernel(q_order, d_ext=d_ext, c=mp.mpc(c_k))
        )
    return TorusOnePtContourKernels(
        c0=c0,
        d_ext=d_ext,
        q_order=q_order,
        radius=radius,
        kernels=tuple(kernels),
    )


def torus_onept_f_coeffs_singular(
    contour: TorusOnePtContourKernels, h
) -> list:
    """[q^n] f(q, h, d, c0) at a singular c0 via the contour average.

    Real (c0, h, d) coerce to mpf with an imaginary-part guard: the average's
    residual imaginary part is a direct estimate of the O(r^K) + roundoff
    error, so a loud failure here means the radius/dps budget is starved.
    """
    K = len(contour.kernels)
    acc = None
    for kernel in contour.kernels:
        coeffs = torus_onept_f_coeffs(kernel, mp.mpc(h), coerce_real=False)
        if acc is None:
            acc = [mp.mpc(coeff) for coeff in coeffs]
        else:
            for i, coeff in enumerate(coeffs):
                acc[i] += mp.mpc(coeff)
    avg = [value / K for value in acc]

    real_inputs = (
        not (isinstance(contour.c0, mp.mpc) and mp.im(contour.c0) != 0)
        and not (isinstance(h, mp.mpc) and mp.im(mp.mpc(h)) != 0)
        and not (
            isinstance(contour.d_ext, mp.mpc)
            and mp.im(mp.mpc(contour.d_ext)) != 0
        )
    )
    if not real_inputs:
        return avg
    # The residual imaginary part measures the contour truncation + the
    # cancellation-amplified roundoff; demand a few digits of headroom.
    tol = mp.mpf(10) ** (-max(1, mp.mp.dps // 4))
    out = []
    for value in avg:
        scale = max(abs(value), mp.mpf(1))
        if abs(mp.im(value)) > tol * scale:
            raise ValueError(
                "contour average did not converge (residual imaginary part "
                f"{mp.nstr(mp.im(value), 5)}); increase dps, shrink the "
                "radius, or raise n_points"
            )
        out.append(mp.re(value))
    return out


# ---------------------------------------------------------------------------
# Singular-c evaluation: exact Laurent series in t, t^2 = c - c0
# ---------------------------------------------------------------------------
#
# In the contour average, a spurious pole of order p at radius r from c0
# costs ~p*log10(1/r) digits through cancellation. The Laurent evaluator
# cancels these pole coefficients algebraically, as in the sphere kernel.
#
# Setting t^2 = c - c0 and carrying every recursion quantity as a Laurent
# series in t: b(c) = b_series(c0) is analytic in t (a sqrt branch point in c
# becomes regular in t), so each residue R_{r,s} and each interaction
# denominator d_{r,s}+rs - d_{r',s'} is a Laurent series whose negative
# powers cancel against the other recursion terms. The physical block is the
# t^0 coefficient; a surviving negative power is a genuine divergence and
# raises an error.
#
# The recursion is the sphere's (one sector, f = 1 + ...): ``f_k(h) = sum_j R_j / (h - d_j) f_j[k - rs_j](d_j + rs_j)``
# with the torus residue ``R = A_rs P[d; d_rs] P[d; d_rs + rs]``, so it runs on the shared level-table engine
# (:mod:`virasoro_blocks.laurent_levels`): ceiling-tracked windows sized from what is read, collisions snapped
# exactly on the locus's rational b^2, exact inputs as balls, and the working precision plus the loss the table's
# own error bound measures.


@dataclass(eq=False)
class TorusOnePtLaurentState:
    """The torus one-point level table at a singular ``c0`` (:func:`laurent_levels.build_levels`).

    ``pairs[k]`` carries ``r, s, rs`` and the LaurentSeries ``d`` (``d_{r,s}(b(t))``), ``R`` (the residue) and ``w``
    (``d + rs``); ``levels`` is the table, built at ``work_dps``.  Each evaluation is verified to ``base_dps`` by its
    own balls, and the table is rebuilt at more digits when one falls short (kept for later evaluations).  Reused
    across many internal weights h (the eval step is cheap).
    """

    c0: object
    d_ext: object
    q_order: int
    work_dps: int
    pairs: tuple
    levels: object
    real: bool
    base_dps: int


def _torus_laurent_poles(d_ext, c0, q_order: int, shared):
    """The torus one-point Kac poles at ``c0`` as Tracked series: ``K = A_rs P[d; d_rs] P[d; d_rs + rs]``, on the
    sphere's Kac data (:class:`recursion._KacLaurentParts`)."""
    from .recursion import _KacLaurentParts, _enumerate_rs_pairs

    kac = _KacLaurentParts(c0, q_order, shared)
    lam_ext = kac.lam_sq(d_ext)
    poles = []
    for r, s, rs in _enumerate_rs_pairs(q_order):
        d = kac.d(r, s)
        lam_lower = kac.Q_sq - 4 * d
        lam_upper = lam_lower - 4 * rs
        K = kac.A_rs(r, s) * kac.fusion(r, s, lam_ext, lam_lower) * kac.fusion(r, s, lam_ext, lam_upper)
        d_const = kac.d_const(r, s)
        poles.append(dict(r=r, s=s, rs=rs, K=K, d=d, v=d + rs, d_const=d_const,
                          v_const=None if d_const is None else d_const + rs, sector=0))
    return poles


def _torus_laurent_levels(d_ext, c0, q_order: int, *, real: bool, half=None):
    from . import tseries as _ts
    from .laurent_levels import build_levels
    from .recursion import _laurent_b_cache, _scalar_cache_key

    cacheable = _scalar_cache_key(c0) is not None
    saved = _ts.set_real_mode(real)
    try:
        return build_levels(
            lambda: _torus_laurent_poles(d_ext, c0, q_order, _laurent_b_cache(c0, mp.mp.dps) if cacheable else {}),
            int(q_order), seeds=lambda _sector, n: [1] + [0] * n, couples=lambda _sector, _pole: True,
            label=lambda p: (p["r"], p["s"]), half=_ts.window_half() if half is None else half,
            window_key=("torus", _scalar_cache_key(c0), int(q_order), mp.mp.prec) if cacheable else None)
    finally:
        _ts.set_real_mode(saved)


def _rebuild(state: TorusOnePtLaurentState, work_dps: int, half=None) -> None:
    with mp.workdps(work_dps):
        state.levels = _torus_laurent_levels(state.d_ext, state.c0, state.q_order, real=state.real, half=half)
    state.work_dps = int(work_dps)


def build_torus_onept_laurent_state(
    q_order: int, *, d_ext, c0, work_dps: int | None = None
) -> TorusOnePtLaurentState:
    """The exact Laurent (t)-series torus one-point kernel at a singular central charge ``c0``.

    Mirrors the sphere's :func:`~virasoro_blocks.recursion.H_q_series_laurent_precompute`: b and every recursion
    quantity are Laurent series in t, so the representation's pole artifacts cancel algebraically.  Reuse the state
    across many internal weights via :func:`torus_onept_f_coeffs_laurent`, which verifies each evaluation by the
    table's own balls (every exact input a ball containing it) and raises the table's precision by the measured
    shortfall where one needs it.  Built at the working precision (or ``work_dps``).
    """
    q_order = _integer_control("q_order", q_order)
    c0, d_ext = _onept_input(c0, "c0"), _onept_input(d_ext, "d_ext")
    if not _is_complex(c0):
        c0 = real_input_label(c0, name="c0")
    base = mp.mp.dps
    if work_dps is None:
        work_dps = base
    work_dps = _integer_control("work_dps", work_dps, base)
    real = not _is_complex(c0) and not _is_complex(d_ext)
    with mp.workdps(work_dps):
        levels = _torus_laurent_levels(d_ext, c0, q_order, real=real)
    pairs = tuple(dict(r=p["r"], s=p["s"], rs=p["rs"], d=p["d"].s, R=p["K"].s, w=p["v"].s) for p in levels.poles)
    return TorusOnePtLaurentState(c0=c0, d_ext=d_ext, q_order=int(q_order), work_dps=int(work_dps), pairs=pairs,
                                  levels=levels, real=real, base_dps=int(base))


def torus_onept_f_coeffs_laurent(state: TorusOnePtLaurentState, h, *, guard_imag: bool = True) -> list:
    """[q^n] f(q, h, d, c0) at the singular c0 via the Laurent kernel: the fixed-h t^0 coefficients (the c -> c0
    limit; on a Kac weight that is the Verma-module trace, the point values take the irreducible one,
    :func:`onept_point_f_coeffs`).  An ``h`` on a Kac weight of c0 has its collision snapped exactly.

    Each coefficient is verified to the state's ``base_dps`` by its ball radius plus the rounding of its
    midpoint (relative, unit floor): a shortfall rebuilds the table at that many more digits, once (the state keeps
    it); a bound that does not shrink raises.  Returns mpf coefficients for real scalar (c0, h, d), else mpc.  A
    negative t-power surviving the cancellation is a genuine divergence and raises.
    """
    from . import tseries as _ts
    from .laurent_levels import WindowDeficit, eval_t0, recognized_rational

    q_order = state.q_order
    real, imaginary = literal_components(_onept_input(h, "h"))
    h_source = real if imaginary == 0 else literal_number(h)
    work = max(mp.mp.dps, state.base_dps)
    if work > state.work_dps:
        _rebuild(state, work)
    with mp.workdps(state.base_dps):
        target = +mp.eps
    previous = None
    while True:
        with mp.workdps(state.work_dps):
            h_w = h_source
            complex_h = isinstance(h_w, mp.mpc) and mp.im(h_w) != 0
            exact = None if complex_h else recognized_rational(h_w)
            radii: list = []
            saved = _ts.set_real_mode(state.real and not complex_h)
            try:
                values, negatives = eval_t0(state.levels, exact if exact is not None else h_w, sector=0,
                                            seed=[1] + [0] * q_order, couples=lambda _sector, _pole: True, radii=radii)
            except WindowDeficit as exc:
                # this weight reads deeper than the build's analysis covers: the table on a window wider by that much
                _rebuild(state, state.work_dps, half=state.levels.half + exc.deficit)
                continue
            finally:
                _ts.set_real_mode(saved)
            if negatives:
                k, valuation = negatives[0]
                raise ValueError(
                    f"torus_onept_f_coeffs_laurent: f_{k} "
                    + (f"keeps a t^{valuation} pole" if valuation is not None else "has a pole at every c, here")
                    + f" at c0 = {state.c0}; the internal "
                    "weight resonates with a degenerate state that does not decouple (a genuine divergence)")
            worst = max(((r + mp.eps / 2 * abs(v)) / max(1, abs(v)) for v, r in zip(values, radii)),
                        default=mp.mpf(0))
            out = [+v for v in values]
        if worst <= target:
            break
        short = state.work_dps if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / target)))
        if previous is not None and mp.isfinite(worst) and worst >= previous:
            raise ArithmeticError(f"the torus one-point kernel at c = {state.c0}: the Laurent table's error bound "
                                  "does not shrink with the precision")
        previous = worst
        _rebuild(state, state.work_dps + max(1, short))
    with mp.workdps(state.base_dps):
        out = [+v for v in out]
        real_inputs = not _is_complex(state.c0) and not _is_complex(h) and not _is_complex(state.d_ext)
        if not real_inputs:
            return out
        return _coerce_real_coeffs(out, "Laurent f coefficient", guard=guard_imag)


# ---------------------------------------------------------------------------
# Irreducible-module one-point traces (degenerate internal weights)
# ---------------------------------------------------------------------------
#
# On a Kac weight h_{r,s}(c0) the physical one-point block is the trace over the IRREDUCIBLE module
# L(h) = M(h) / (null submodule).  Neither limit of the recursion is it: the c -> c0 limit at fixed h is the trace
# over the whole Verma module (the near-null states keep the finite diagonal matrix element <chi|O|chi>/<chi|chi>; at
# d = 0 the recursion gives the Verma character, not the minimal-model one), and so is the limit in h.  The trace
# over L(h) is built level by level in quotient bases (`irreducible_onept_F_series`): L_n is spanned by L_{-1} L_{n-1}
# and L_{-2} L_{n-2} (L_{-1}, L_{-2} generate the negative modes), a maximal independent subset of those candidates
# is chosen exactly (row reduction mod a prime; c0 and h are rational on every recognized locus), and the trace is
# computed mod several 61-bit primes on those bases and reconstructed as the exact rational it is (CRT + rational
# reconstruction, primes added until it is stable).  The external must decouple from the null states
# (<nu|O|h> = 0 for every null nu, checked exactly): otherwise the irreducible block does not exist
# (`TorusOnePtDecouplingError`).  Checks: Tr_L(sigma) epsilon = prod(1 - q^n) in Ising; the minimal-model and c = 1
# irreducible characters at d = 0; Tr_{L(j^2)} J_0 = character at c = 1.


def _partitions(n: int, cap: int | None = None) -> list:
    if n == 0:
        return [()]
    if cap is None:
        cap = n
    out = []
    for first in range(min(n, cap), 0, -1):
        for rest in _partitions(n - first, first):
            out.append((first,) + rest)
    return out


def _apply_L(m: int, part: tuple, h, c) -> dict:
    if part == ():
        if m > 0:
            return {}
        if m == 0:
            return {(): h}
        return {(-m,): mp.mpf(1)}
    a, rest = part[0], part[1:]
    if m < 0 and -m >= a:
        return {(-m,) + part: mp.mpf(1)}
    out: dict = {}
    for p, coef in _apply_L(m, rest, h, c).items():
        for p2, coef2 in _apply_L(-a, p, h, c).items():
            out[p2] = out.get(p2, mp.mpf(0)) + coef * coef2
    for p, coef in _apply_L(m - a, rest, h, c).items():
        out[p] = out.get(p, mp.mpf(0)) + mp.mpf(m + a) * coef
    if m == a:
        out[rest] = out.get(rest, mp.mpf(0)) + mp.mpf(c) / 12 * m * (m * m - 1)
    return out


def _shapovalov(A: tuple, B: tuple, h, c):
    state = {B: mp.mpf(1)}
    for a in A:
        new: dict = {}
        for p, coef in state.items():
            for p2, coef2 in _apply_L(a, p, h, c).items():
                new[p2] = new.get(p2, mp.mpf(0)) + coef * coef2
        state = new
    return state.get((), mp.mpf(0))


_PRIME_LIST: list = []


def _prime(i: int) -> int:
    """The i-th prime below 2^61 (descending): the moduli of the exact multi-modular traces."""
    from flint import fmpz

    p = _PRIME_LIST[-1] - 2 if _PRIME_LIST else 2 ** 61 - 1
    while len(_PRIME_LIST) <= i:
        if fmpz(p).is_prime():
            _PRIME_LIST.append(p)
        p -= 2
    return _PRIME_LIST[i]


def _rational_reconstruction(a: int, m: int):
    """The fraction r/s = a (mod m) with |r|, s <= sqrt(m/2) (Wang), or None."""
    import math
    from fractions import Fraction

    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return Fraction(r1, s1) if s1 > 0 else Fraction(-r1, -s1)


class TorusOnePtDecouplingError(ValueError):
    """The external primary does not decouple from the null submodule of the degenerate internal module, so the
    irreducible one-point block does not exist (a fusion rule forbids it)."""


def _recognized_fraction(x):
    """``x`` as the rational it denotes at the precision it carries (the rule of `singularity.exact_locus_c`), or
    None.  Fractions, ints and decimal / fraction strings are taken exactly."""
    from fractions import Fraction

    if isinstance(x, (Fraction, int)):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x.strip())
    v = mp.convert(x)
    if isinstance(v, mp.mpc):
        if v.imag != 0:
            return None
        v = v.real
    if v == 0:
        return Fraction(0)
    digits = max(15, int(v._mpf_[3] * 0.30103) - 2)
    with mp.workdps(digits):
        frac = Fraction(mp.nstr(v, digits)).limit_denominator(10 ** 6)
        if abs(mp.mpf(frac.numerator) / frac.denominator - v) > abs(v) * mp.mpf(10) ** (-(digits // 2)):
            return None
    return frac


class _ModP:
    def __init__(self, p):
        from flint import nmod

        self.p = p
        self._nmod = nmod

    def scalar(self, x):
        return self._nmod(x.numerator, self.p) / self._nmod(x.denominator, self.p)

    def zeros(self, r, c):
        from flint import nmod_mat

        return nmod_mat(r, c, self.p)

    def fix(self, M):
        return M

    def solve(self, A, B):
        return A.solve(B)


def _eye(ring, n):
    M = ring.zeros(n, n)
    one = ring.scalar(1)
    for i in range(n):
        M[i, i] = one
    return M


def _quotient_levels(ring, c, h, n_max: int, pivots=None):
    """Quotient bases of L(h) at levels 0..n_max: ``S[n]`` (each state ``(k, j)`` = L_{-k} S[n-k][j], k in {1, 2}),
    their Gram matrices ``G[n]`` and the matrices ``up[(m, k)]`` (L_{-k}: level m -> m + k) and ``down[(m, k)]``
    (L_k: m -> m - k) in those bases.  With ``pivots`` None (a mod-p ring) the bases are chosen by row reduction and
    the null directions are returned too (``nulls[n]``: candidate-space kernel of the level-n Gram matrix); otherwise
    ``pivots[n]`` are the candidate indices to keep."""
    from fractions import Fraction

    S = [[(0, 0)]]
    G = [_eye(ring, 1)]
    up, down, cands_at, nulls, piv_at = {}, {}, {0: [(0, 0)]}, {}, {0: [0]}
    cq = ring.scalar(Fraction(c))
    op_cache: dict = {}

    def r(m):
        return len(S[m])

    def op(k, kp, m):
        """L_k L_{-k'} on level m (into level m + k' - k), k, k' in {1, 2}."""
        key = (k, kp, m)
        hit = op_cache.get(key)
        if hit is not None:
            return hit
        tgt = m + kp - k
        M = ring.zeros(r(tgt), r(m))
        if k <= m:
            M = M + up[(m - k, kp)] * down[(m, k)]
        if k > kp:
            M = M + down[(m, k - kp)] * ring.scalar(Fraction(k + kp))
        elif k < kp:
            M = M + up[(m, kp - k)] * ring.scalar(Fraction(k + kp))
        else:
            M = M + _eye(ring, r(m)) * ring.scalar(2 * k * (Fraction(h) + m) + Fraction(c) / 12 * (k ** 3 - k))
        M = ring.fix(M)
        op_cache[key] = M
        return M

    for n in range(1, n_max + 1):
        ks = [k for k in (1, 2) if k <= n]
        offs, cands = {}, []
        for k in ks:
            offs[k] = len(cands)
            cands += [(k, j) for j in range(r(n - k))]
        N = len(cands)
        K = ring.zeros(N, N)
        for k in ks:
            for kp in ks:
                if not (r(n - k) and r(n - kp)):
                    continue
                X = ring.fix(G[n - k] * op(k, kp, n - kp))
                ok, okp = offs[k], offs[kp]
                for a in range(X.nrows()):
                    for b in range(X.ncols()):
                        K[ok + a, okp + b] = X[a, b]
        if pivots is None:
            R, rank = K.rref()
            piv, row = [], 0
            for col in range(N):
                if row < rank and R[row, col] != 0:
                    piv.append(col)
                    row += 1
            X, nullity = K.nullspace()
            nulls[n] = (X, nullity)
        else:
            piv = pivots[n]
        piv_at[n] = piv
        cands_at[n] = cands
        S.append([cands[i] for i in piv])
        Gn = ring.zeros(len(piv), len(piv))
        for a, ia in enumerate(piv):
            for b, ib in enumerate(piv):
                Gn[a, b] = K[ia, ib]
        G.append(Gn)
        for k in (1, 2):
            m = n - k
            if m < 0:
                continue
            rhs = ring.zeros(len(piv), r(m))
            if k in offs:
                for a, ia in enumerate(piv):
                    for b in range(r(m)):
                        rhs[a, b] = K[ia, offs[k] + b]
            up[(m, k)] = ring.solve(Gn, rhs) if piv else rhs
        for k in ks:
            out = ring.zeros(r(n - k), r(n))
            for col, (k1, j1) in enumerate(S[n]):
                v = op(k, k1, n - k1)
                for a in range(r(n - k)):
                    out[a, col] = v[a, j1]
            down[(n, k)] = out
    return {"S": S, "G": G, "up": up, "down": down, "nulls": nulls, "pivots": piv_at, "cands": cands_at}


def _onept_blocks(ring, Q, d, n_max: int):
    """``B[(a, b)]`` = <S_a| O_d(1) |S_b> on the quotient bases (Ward identities from <h|O|h> = 1)."""
    from fractions import Fraction

    S, down = Q["S"], Q["down"]
    B: dict = {}

    def r(m):
        return len(S[m])

    def blk(a, b):
        hit = B.get((a, b))
        if hit is not None:
            return hit
        M = ring.zeros(r(a), r(b))
        if a == 0 and b == 0:
            M = _eye(ring, 1)
        elif b >= a:
            # <u|O|L_{-k} w'> = <L_k u|O|w'> - (a - (b - k) - k d) <u|O|w'>
            per_k: dict = {}
            for col, (k, jp) in enumerate(S[b]):
                if k not in per_k:
                    t = blk(a, b - k) * ring.scalar(-(Fraction(a - b + k) - k * d))
                    if k <= a:
                        t = t + down[(a, k)].transpose() * blk(a - k, b - k)
                    per_k[k] = t
                t = per_k[k] = ring.fix(per_k[k])
                for i in range(r(a)):
                    M[i, col] = t[i, jp]
        else:
            # <L_{-k} u'|O|w> = <u'|O|L_k w> + (a - k - b + k d) <u'|O|w>
            per_k = {}
            for row, (k, ip) in enumerate(S[a]):
                if k not in per_k:
                    t = blk(a - k, b) * ring.scalar(Fraction(a - k - b) + k * d)
                    if k <= b:
                        t = t + blk(a - k, b - k) * down[(b, k)]
                    per_k[k] = t
                t = per_k[k] = ring.fix(per_k[k])
                for jj in range(r(b)):
                    M[row, jj] = t[ip, jj]
        B[(a, b)] = M
        return M

    return blk


def _check_decoupling(ring, Q, d, n_max: int, blk):
    """Raise unless <nu|O|h> = 0 for every null nu at every level (exact, in the mod-p ring).  In quotient
    coordinates every level-n null is annihilated by the lowering modes, so this is the whole decoupling condition."""
    from fractions import Fraction

    for n in range(1, n_max + 1):
        X, nullity = Q["nulls"][n]
        if not nullity:
            continue
        cands = Q["cands"][n]
        vec = ring.zeros(1, len(cands))
        for i, (k, j) in enumerate(cands):
            vec[0, i] = ring.scalar(Fraction(n - k) + k * d) * blk(n - k, 0)[j, 0]
        prod = vec * X
        for col in range(nullity):
            if prod[0, col] != 0:
                raise TorusOnePtDecouplingError(
                    f"the external d = {d} does not decouple from the level-{n} null states of the degenerate "
                    "internal module (<null|O|h> != 0): a fusion rule forbids this one-point block")


_QUOTIENT_PIVOTS: dict = {}
_IRREDUCIBLE_SERIES: dict = {}


def _quotient_pivots(c, h, n_max: int):
    """The exact pivot choice for L(h) at (c, h) and the mod-p data for the decoupling check, from two primes (the
    choice is a rank statement; two primes agreeing rules out an accidental drop mod one of them)."""
    key = (c, h, n_max)
    hit = _QUOTIENT_PIVOTS.get(key)
    if hit is not None:
        return hit
    found = []
    for i in range(2):
        ring = _ModP(_prime(i))
        found.append((ring, _quotient_levels(ring, c, h, n_max)))
    if found[0][1]["pivots"] != found[1][1]["pivots"]:
        raise ArithmeticError("irreducible torus trace: the null-space rank differs between two primes")
    if len(_QUOTIENT_PIVOTS) > 32:
        _QUOTIENT_PIVOTS.clear()
    _QUOTIENT_PIVOTS[key] = found
    return found


def _trace_mod(ring, Q, d, n_max: int) -> list:
    blk = _onept_blocks(ring, Q, d, n_max)
    out = [1]
    for n in range(1, n_max + 1):
        r = len(Q["S"][n])
        if not r:
            out.append(0)
            continue
        X = ring.solve(Q["G"][n], blk(n, n))
        out.append(int(sum((X[i, i] for i in range(r)), ring.scalar(0))))
    return out


def irreducible_onept_F_series(h, c, d_ext, n_max: int, *, exact: bool = False) -> list:
    """[q^n] of the normalized one-point trace over the IRREDUCIBLE module L(h) at central charge c, n = 0..n_max.

    ``c`` and ``h`` must be rational (a recognized singular c and a Kac weight of it: Fractions, or values that
    carry them), ``d_ext`` the external chiral weight, rational as well (a non-rational external cannot decouple
    from the null states).  The coefficients are exact rationals (``exact=True`` returns them as Fractions; else as
    mpf at the working precision).  For a non-degenerate (h, c) every state is kept and this is the Verma trace
    F(q, h, d, c); on a Kac weight it is the physical irreducible-module one-point block.  Raises
    :class:`TorusOnePtDecouplingError` when the external does not decouple from the null submodule.
    """
    n_max = _integer_control('n_max', n_max)
    from .laurent_levels import recognized_rational

    cq, hq, dq = (recognized_rational(v) for v in (c, h, d_ext))
    if cq is None or hq is None:
        raise ValueError("irreducible_onept_F_series: c and h must be exact rationals (a recognized singular c and "
                         "one of its Kac weights)")
    if dq is None:
        raise TorusOnePtDecouplingError(
            f"the external d = {mp.nstr(mp.convert(d_ext), 15)} is not rational, so it cannot decouple from the null "
            "states of the degenerate internal module: the irreducible one-point block does not exist")
    key = (cq, hq, dq, n_max)
    series = _IRREDUCIBLE_SERIES.get(key)
    if series is None:
        found = _quotient_pivots(cq, hq, n_max)
        for ring, Qp in found:
            _check_decoupling(ring, Qp, dq, n_max, _onept_blocks(ring, Qp, dq, n_max))
        pivots = found[0][1]["pivots"]
        residues, modulus, previous, i = None, 1, None, 0
        while True:
            if i < 2:
                ring, Q = found[i]
            else:
                ring = _ModP(_prime(i))
                try:
                    Q = _quotient_levels(ring, cq, hq, n_max, pivots=pivots)
                except ZeroDivisionError:          # a Gram determinant divisible by this prime: skip it
                    i += 1
                    continue
            values = _trace_mod(ring, Q, dq, n_max)
            p = ring.p
            if residues is None:
                residues = values
            else:
                inv = pow(modulus, -1, p)
                residues = [x + modulus * (((v - x) * inv) % p) for x, v in zip(residues, values)]
            modulus *= p
            current = [_rational_reconstruction(x, modulus) for x in residues]
            if None not in current and current == previous:
                series = current
                break
            previous = current
            i += 1
        if len(_IRREDUCIBLE_SERIES) > 64:
            _IRREDUCIBLE_SERIES.clear()
        _IRREDUCIBLE_SERIES[key] = series
    if exact:
        return list(series)
    return [mp.mpf(x.numerator) / x.denominator for x in series]


def _kac_labels(c, h, max_level: int):
    """``(r, s)`` with ``h = d_{r,s}(c)``, ``rs <= max_level``: exactly at a recognized singular c (rational b^2 and
    h), at the working precision's pole tolerance elsewhere."""
    from .params import b_from_c, d_rs
    from .laurent_levels import recognized_rational
    from .singularity import exact_b_squared, exact_locus_c

    x = exact_b_squared(c, q_order=max_level) if exact_locus_c(c, q_order=max_level) is not None else None
    labels = []
    if x is not None:
        hq = recognized_rational(h)
        if hq is None:
            return labels
        for r in range(1, max_level + 1):
            for s_ in range(1, max_level // r + 1):
                if ((1 - r * r) * x + (1 - s_ * s_) / x + 2 - 2 * r * s_) / 4 == hq:
                    labels.append((r, s_))
        return labels
    b = b_from_c(mp.convert(c))
    tol = 8 * mp.eps * max(1, abs(mp.mpc(h)))
    for r in range(1, max_level + 1):
        for s_ in range(1, max_level // r + 1):
            if abs(mp.mpc(d_rs(r, s_, b)) - mp.mpc(h)) <= tol:
                labels.append((r, s_))
    return labels


def irreducible_character_factor(c, h, q):
    """``g(q)`` with ``q^{h - (c-1)/24} g(q)`` the reduced (eta-stripped) character of the irreducible module L(h),
    and whether ``h`` is a Kac weight (``(g, degenerate)``): ``g = 1`` for a Verma module; at a Kac weight of a
    generic c the one null submodule gives ``1 - q^{rs}``; at a recognized singular c the module is resolved
    exactly (`irreducible_onept_F_series` at d = 0, the dimensions of L(h)): ``g = prod(1 - q^n) sum_n dim L_n q^n``,
    summed to the order whose tail sum_{n > N} p(n) |q|^n (dim L_n <= p(n)) is below the working precision."""
    from .singularity import exact_locus_c

    aq = abs(q)
    if not mp.isfinite(aq) or aq >= 1:
        raise ValueError("irreducible_character_factor requires a finite nome with |q| < 1")
    if aq == 0:
        return mp.mpf(1), False
    eta_product = mp.qp(q)
    # Bound the positive partition tail without subtracting nearly equal sums:
    # for r = exp(-t) > aq, tail_N <= (aq/r)^(N+1) / prod(1-r^k).
    # log(prod(1-r^k)^-1) = sum_k 1/[k(exp(kt)-1)] <= pi^2/(6t).
    # Minimizing in t gives log(tail_N) <= pi sqrt(2(N+1)/3) + (N+1) log(aq).
    # Solve this quadratic in sqrt(N+1), including the final eta-product factor.
    with mp.workdps(mp.mp.dps + 10):
        a = -mp.log(aq)
        b = mp.pi * mp.sqrt(mp.mpf(2) / 3)
        digits = (mp.mp.dps - 10) * mp.log(10) + mp.log(max(1, abs(eta_product)))
        root = (b + mp.sqrt(b*b + 4*a*digits)) / (2*a)
        N = max(1, int(mp.ceil(root*root)))
    labels = _kac_labels(c, h, N)
    if not labels:
        return mp.mpf(1), False
    exact_c = exact_locus_c(c, q_order=N)
    if exact_c is None:
        (r, s_), = labels
        return 1 - q ** (r * s_), True
    dims = irreducible_onept_F_series(h, _recognized_fraction(exact_c), 0, N, exact=True)
    series = mp.mpc(0)
    for dim in reversed(dims):
        series = series * q + dim.numerator
    return eta_product * series, True


def irreducible_character_levels(c, h, max_level: int):
    """The reduced irreducible character of L(h) as a sparse q-series: ``[(level, coefficient), ...]`` (levels
    ``<= max_level``, ascending, starting with ``(0, 1)``) with ``q^{h - (c-1)/24} sum coefficient q^level`` the
    eta-stripped character, or None when ``h`` is not a Kac weight (L(h) is the Verma module).

    Every submodule of a Virasoro Verma module is a sum of Verma submodules, and V(h') sits in V(h) exactly when a
    chain of singular vectors (h = d_{r,s}, a singular vector at level rs, then the same from h + rs) leads from h to
    h' (Feigin-Fuchs).  The character is then ``sum_{h'} mu(h, h') ch V(h')`` with mu the Moebius function of that
    embedding poset: ``1 - q^{rs}`` at a generic c (one null submodule), the chain and braid (Rocha-Caridi) series at
    a recognized singular c, where the poset is built exactly from the rational ``b^2``.  Unlike
    :func:`irreducible_character_factor` (the level dimensions from Gram ranks) this costs nothing per level, so the
    deep truncations of high Taylor orders are cheap.  At a generic c the Kac weight is matched at the working
    precision's tolerance, as :func:`irreducible_character_factor` matches it."""
    from .singularity import exact_b_squared, exact_locus_c
    from .laurent_levels import recognized_rational

    max_level = _integer_control("max_level", max_level)
    x = exact_b_squared(c, q_order=max(1, max_level)) if exact_locus_c(c, q_order=max(1, max_level)) is not None \
        else None
    if x is None:
        labels = _kac_labels(c, h, max_level)
        if not labels:
            return None
        levels = sorted({r * s_ for r, s_ in labels})
        if len(levels) != 1:
            raise ArithmeticError(f"h = {mp.nstr(mp.mpc(h), 15)} matches Kac weights at levels {levels} of a "
                                  "c that is not a recognized singular locus")
        return [(0, 1)] + ([(levels[0], -1)] if levels[0] <= max_level else [])
    root = recognized_rational(h)
    if root is None:
        return None

    def kac_lines(weight):
        # d_{r,s} = weight  <=>  (r x + s)^2 = (1 + x)^2 - 4 weight x: the values t = r x + s, or () off the Kac table
        square = (1 + x) ** 2 - 4 * weight * x
        root_num, root_den = _isqrt_exact(square.numerator), _isqrt_exact(square.denominator)
        if root_num is None or root_den is None:
            return ()
        sigma = Fraction(root_num, root_den)
        return (sigma, -sigma) if sigma else (sigma,)

    def singular_levels(weight, budget):
        # the levels rs <= budget of the singular vectors of V(weight)
        found = set()
        for t in kac_lines(weight):
            for r in range(1, budget + 1):
                s_ = t - r * x
                if s_.denominator == 1 and s_ > 0 and r * s_ <= budget:
                    found.add(r * int(s_))
        return found

    def is_kac_weight(weight):
        # a positive integer point (r, s) on a line r x + s = t: for x < 0 one with integral s at r <= den(x) moves to
        # s > 0 in steps of den(x); for x > 0 s falls with r, so r < t / x
        for t in kac_lines(weight):
            stop = x.denominator if x < 0 else int(t / x) if t > 0 else 0
            if any((t - r * x).denominator == 1 and (x < 0 or t - r * x > 0) for r in range(1, stop + 1)):
                return True
        return False

    if not is_kac_weight(root):
        return None
    # the embedding poset through max_level: parents of each submodule weight (by level above h)
    parents: dict = {0: set()}
    frontier = [0]
    while frontier:
        level = min(frontier)
        frontier.remove(level)
        for step in singular_levels(root + level, max_level - level):
            child = level + step
            if child not in parents:
                parents[child] = set()
                frontier.append(child)
            parents[child].add(level)
    ancestors: dict = {}
    mobius: dict = {}
    for level in sorted(parents):
        above = set()
        for parent in parents[level]:
            above |= {parent} | ancestors[parent]
        ancestors[level] = above
        mobius[level] = 1 if level == 0 else -sum(mobius[z] for z in above)
    return [(level, mobius[level]) for level in sorted(mobius) if mobius[level] != 0]


def _isqrt_exact(n: int):
    """``sqrt(n)`` when ``n >= 0`` is a perfect square, else None."""
    import math

    if n < 0:
        return None
    root = math.isqrt(n)
    return root if root * root == n else None


def _f_irr_from_F_irr(F_irr: list) -> list:
    """f_irr = prod_{n>=1}(1 - q^n) * F_irr, the reduced-convention recursion
    object (the eta factor; reduces to f for a Verma module)."""
    n_max = len(F_irr) - 1
    prod = [mp.mpf(1)] + [mp.mpf(0)] * n_max
    for k in range(1, n_max + 1):
        new = list(prod)
        for i in range(n_max + 1 - k):
            new[i + k] -= prod[i]
        prod = new
    out = [mp.mpf(0)] * (n_max + 1)
    for i in range(n_max + 1):
        for j in range(n_max + 1 - i):
            out[i + j] += mp.convert(F_irr[i]) * prod[j]
    return out


def kac_weight(r: int, s: int, p: int, q: int):
    """Minimal-model M(p, q) Kac weight h_{r,s} = ((q r - p s)^2 - (q-p)^2)/(4 p q)."""
    return (mp.mpf((q * r - p * s) ** 2 - (q - p) ** 2)) / (4 * p * q)


def _chiral_irr_v_coeffs(f_irr: list, h, c, lam: int) -> list:
    """[v^p] of q(v)^{h-(c-1)/24} * (sum_n f_irr[n] q(v)^n): the reduced
    chiral factor of an irreducible-module one-point contribution."""
    from .modular import _chiral_coeff_polys

    hhat = h - (c - 1) / 24
    C = _chiral_coeff_polys(lam + 1)
    scale = mp.exp(-2 * mp.pi * hhat)
    pref = []
    for pp in range(lam + 1):
        acc = mp.mpc(0)
        for coef in reversed(C[pp]):
            acc = acc * hhat + coef
        pref.append(scale * acc)
    q_order = len(f_irr) - 1
    Q = _qpow_matrix(lam, q_order)
    f_v = []
    for pp in range(lam + 1):
        row = Q[pp]
        acc = mp.mpc(0)
        for n, fn in enumerate(f_irr):
            if fn != 0:
                acc += row[n] * fn
        f_v.append(acc)
    out = []
    for pp in range(lam + 1):
        acc = pref[0] * f_v[pp]
        for i in range(1, pp + 1):
            acc += pref[i] * f_v[pp - i]
        out.append(acc)
    return out


def torus_onept_degenerate_point_table(
    r: int,
    s: int,
    p: int,
    q: int,
    lam: int,
    *,
    d_ext,
    n_max: int,
    spin: int = 0,
    symmetrize: bool = True,
) -> list:
    """Reduced one-point (m, n) POINT-block at the degenerate internal primary
    h_{r,s} of minimal model M(p, q) with a scalar external of chiral weight
    d_ext (Delta_Phi = 2 d_ext).

    A single crossing vector (no Delta-interpolation): the internal weight is
    the fixed Kac weight, so this is the torus analogue of a point-constraint
    block. Chiral halves carry h = h_{r,s}, hbar = h_{r,s} - spin; the
    default `symmetrize` adds the (j, -j) pair. Built from the irreducible
    quotient trace at central charge c(p, q).
    """
    from .modular import tau2_power_table
    from .params import c_from_b, b_from_c

    if not (1 <= r < p and 1 <= s < q):
        raise ValueError(
            f"(r, s) = ({r}, {s}) outside the M({p},{q}) Kac table "
            f"1 <= r < p, 1 <= s < q"
        )
    c = 1 - 6 * mp.mpf((p - q) ** 2) / (p * q)
    spin = int(spin)
    h_deg = kac_weight(r, s, p, q)
    # (Delta, j) at this degenerate primary: Delta = 2 h_deg is the scalar
    # value; spinning degenerate insertions use h = h_deg, hbar = h_deg - j
    # (both chiral halves degenerate is the physical scalar case j = 0).
    h_chiral = h_deg
    hbar_chiral = h_deg - spin if spin else h_deg

    F_h = irreducible_onept_F_series(h_chiral, c, d_ext, n_max)
    f_h = _f_irr_from_F_irr(F_h)
    u = _chiral_irr_v_coeffs(f_h, h_chiral, c, lam)
    if spin == 0:
        w = u
    else:
        F_hb = irreducible_onept_F_series(hbar_chiral, c, d_ext, n_max)
        w = _chiral_irr_v_coeffs(_f_irr_from_F_irr(F_hb), hbar_chiral, c, lam)

    P2D = tau2_power_table(lam, d_ext + mp.mpf(1) / 2)
    nn = lam + 1
    table = [[mp.mpc(0)] * nn for _ in range(nn)]
    for m in range(nn):
        for n2 in range(nn):
            acc = mp.mpc(0)
            for a in range(m + 1):
                row = P2D[a]
                for b in range(n2 + 1):
                    s_ab = row[b]
                    if s_ab == 0:
                        continue
                    acc += s_ab * u[m - a] * w[n2 - b]
            table[m][n2] = acc
    if symmetrize and spin != 0:
        table = [
            [table[m][n2] + table[n2][m] for n2 in range(nn)] for m in range(nn)
        ]
    return table
