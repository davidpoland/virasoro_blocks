"""Lazy compatibility exports for the historical package initializer.

The supported block-engine surface lives in :mod:`virasoro_blocks.api`.
Research code written before that facade was introduced can still import the
older names from ``virasoro_blocks``; this module resolves those names only
when they are requested.  Keeping the mapping here makes the compatibility
boundary explicit without making the package initializer import every
recursion, interpolation, and table-building module at startup.
"""

from importlib import import_module


_MODULE_EXPORTS = {
    ".api": ("build_table_request",),
    ".chiral": (
        "FChiralH",
        "F_block_chiral_h",
    ),
    ".composer": (
        "B_block",
        "G_block",
        "parity_projected_B_block",
        "parity_reversed_B_block",
    ),
    ".coordinates": (
        "q_of_z",
        "q_series_around_half",
        "q_symmetric",
        "y_of_z",
        "y_series_around_half",
        "y_symmetric",
        "z_of_q",
        "z_of_y",
        "z_series_around_q_symmetric",
        "z_series_around_y_symmetric",
        "z_symmetric",
    ),
    ".derivs": (
        "B_qqb_table",
        "B_yyb_table",
        "B_zzb_table",
        "G_qqb_table",
        "G_qqb_ws_radial_table",
        "G_qqb_ws_table",
        "G_ws_radial_table",
        "G_ws_table",
        "G_xt_radial_table",
        "G_xt_table",
        "G_yyb_table",
        "G_zzb_table",
        "global_G_qqb_table",
        "global_G_yyb_table",
        "global_G_zzb_table",
        "chiral_q_taylor",
        "chiral_y_taylor",
        "chiral_z_taylor",
        "evaluate_block",
        "global_chiral_q_taylor",
        "global_chiral_y_taylor",
        "global_chiral_z_taylor",
    ),
    ".evaluation": (
        "BlockPoleError",
        "evaluate_block_request",
    ),
    ".interpolation": (
        "DampedRational",
        "DeltaInterpolant",
        "build_delta_interpolant",
        "build_reduced_delta_polynomial",
        "build_interpolants_for_npoles",
        "compute_interpolation_error_curve",
        "get_sample_points",
        "get_sample_points_jacobi",
        "integrated_density",
        "interpolation_error_curves_for_npoles",
        "lagrange_interpolation_polynomial",
        "max_abs_errors_by_npoles",
        "normalize_delta_evaluator",
        "normalize_delta_prefactor",
        "ordered_block_delta_prefactor",
        "ordered_block_delta_value",
        "plot_max_abs_error_by_npoles",
        "plot_interpolation_error_curves",
        "scalar_block_delta_prefactor",
        "scalar_block_delta_value",
        "solve_support_endpoint",
        "support_endpoint_equation",
    ),
    ".hseries": (
        "HSeries",
        "exp_h_times",
    ),
    ".params": (
        "A_rs",
        "A_rs_imag_b",
        "arithmetic_mode",
        "b_from_c",
        "c_from_b",
        "d_rs",
        "fusion_polynomial",
        "fusion_polynomial_via_squares",
        "lambda_from_h",
    ),
    ".parity": (
        "normalize_external_parity",
        "normalize_parity_phase",
        "parity_phase_value",
        "parity_reversal_phase",
    ),
    ".polyblock": (
        "HRational",
        "H_q_series_rational",
    ),
    ".polyderivs": (
        "B_qqb_poly_table",
        "B_yyb_poly_table",
        "B_zzb_poly_table",
        "ChiralSymbolic",
        "DeltaDerivTable",
        "F_q_poly_table",
        "F_y_poly_table",
        "F_z_poly_table",
        "G_qqb_poly_table",
        "G_qqb_ws_poly_table",
        "G_qqb_ws_radial_poly_table",
        "G_ws_poly_table",
        "G_ws_radial_poly_table",
        "G_xt_poly_table",
        "G_xt_radial_poly_table",
        "G_yyb_poly_table",
        "G_zzb_poly_table",
        "InterpolatedDeltaTable",
        "build_chiral_cache",
        "build_table_from_cache",
        "chiral_symbolic_useries",
        "interpolate_delta_table",
        "parity_projected_B_qqb_ws_poly_table",
        "parity_projected_B_qqb_ws_radial_poly_table",
        "parity_projected_B_qqb_poly_table",
        "parity_projected_B_ws_poly_table",
        "parity_projected_B_ws_radial_poly_table",
        "parity_projected_B_xt_poly_table",
        "parity_projected_B_xt_radial_poly_table",
        "parity_projected_B_yyb_poly_table",
        "parity_projected_B_zzb_poly_table",
    ),
    ".polynomial": ("Polynomial",),
    ".precision": (
        "iteration_cap",
        "set_precision",
        "tolerance",
    ),
    ".recursion": (
        "H_q_series",
        "F_block",
        "q_from_z",
        "block_prefactor",
    ),
    ".requests": (
        "BlockCapabilities",
        "BlockRequest",
        "BlockValue",
        "NumericalPolicy",
        "RequestValidationError",
        "capabilities_for",
        "validate_request",
    ),
    ".regularization": (
        "B_block_regularized",
        "F_block_regularized",
        "G_block_regularized",
        "H_q_series_regularized",
        "merged_delta_table",
    ),
    ".serialization": (
        "SerializedDerivTable",
        "serialize_deriv_tables",
        "serialized_table_from_json",
    ),
    ".writers": (
        "mathematica_filename",
        "write_json",
        "write_mathematica",
    ),
}

_LEGACY_IMPORTS = {
    name: (module_name, name)
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}


def resolve(name):
    """Resolve one historical root-level export, or raise ``KeyError``."""

    module_name, attribute_name = _LEGACY_IMPORTS[name]
    module = import_module(module_name, __package__)
    return getattr(module, attribute_name)


__all__ = ["resolve"]
