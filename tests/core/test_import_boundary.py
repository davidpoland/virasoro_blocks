def test_block_engine_imports_no_other_virasoro_package():
    import subprocess
    import sys

    code = (
        "import sys; import virasoro_blocks; "
        "import virasoro_blocks.superconformal_blocks; "
        "other = sorted({m.split('.')[0] for m in sys.modules} - {'virasoro_blocks'}); "
        "assert not [m for m in other if m.startswith('virasoro_')], other"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_generic_super_block_api_excludes_model_specific_components():
    import subprocess
    import sys

    code = """
from virasoro_blocks import superconformal_blocks
assert not any("top_component" in name for name in superconformal_blocks.__all__)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_block_facades_defer_legacy_numerical_modules():
    import subprocess
    import sys

    code = """
import sys
import virasoro_blocks
import virasoro_blocks.api as api

assert set(sys.modules) >= {"virasoro_blocks.api", "virasoro_blocks._legacy"}
assert "virasoro_blocks.derivs" not in sys.modules
assert "virasoro_blocks.interpolation" not in sys.modules
assert "virasoro_blocks.polyderivs" not in sys.modules
assert "virasoro_blocks.regularization" not in sys.modules

from virasoro_blocks.api import BlockRequest
assert BlockRequest.__name__ == "BlockRequest"
assert "virasoro_blocks.requests" in sys.modules
assert "virasoro_blocks.interpolation" not in sys.modules

from virasoro_blocks import G_zzb_poly_table

assert callable(G_zzb_poly_table)
assert "virasoro_blocks.polyderivs" in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_historical_root_exports_still_resolve():
    import subprocess
    import sys

    code = """
from virasoro_blocks import F_block, build_table_request, regularization

assert callable(F_block)
assert callable(build_table_request)
assert regularization.__name__ == "virasoro_blocks.regularization"
    """
    subprocess.run([sys.executable, "-c", code], check=True)


def test_block_source_keeps_the_dependency_boundary_static():
    import ast
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[2] / "src" / "virasoro_blocks"

    def application(name):
        # any other virasoro_* package (the bootstrap layer, or an application built on the engine)
        return name.startswith("virasoro_") and name != "virasoro_blocks"

    violations = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
                if any(application(name) for name in names):
                    violations.append((path, node.lineno, "application import"))
            elif isinstance(node, ast.ImportFrom):
                if node.module == "virasoro_blocks":
                    violations.append((path, node.lineno, "root package import"))
                elif node.module:
                    top_level = node.module.split(".", 1)[0]
                    if application(top_level):
                        violations.append((path, node.lineno, "application import"))
    assert not violations, violations


def test_supported_api_exports_match_its_lazy_registry():
    from virasoro_blocks import api

    assert set(api.__all__) == set(api._EXPORTS) | {"build_table_request"}
    assert all(not name.startswith("_") for name in api.__all__)
