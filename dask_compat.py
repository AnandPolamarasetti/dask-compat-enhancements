from __future__ import annotations

import importlib
import sys
import types
import warnings
from typing import Literal, Optional

if sys.version_info >= (3, 12):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata
from packaging.version import Version, InvalidVersion

PY_VERSION = Version(".".join(map(str, sys.version_info[:3])))

EMSCRIPTEN = sys.platform == "emscripten"
LINUX = sys.platform == "linux"
MACOS = sys.platform == "darwin"
WINDOWS = sys.platform == "win32"

def entry_points(group: Optional[str] = None):
    warnings.warn(
        "`dask._compatibility.entry_points` has been replaced by `importlib_metadata.entry_points` "
        "and will be removed in a future version. Please use `importlib_metadata.entry_points` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return importlib_metadata.entry_points(group=group)

VERSIONS = {
    "numpy": "1.21.0",
    "pandas": "2.0.0",
    "bokeh": "2.4.2",
    "jinja2": "2.10.3",
    "pyarrow": "7.0.0",
    "lz4": "4.3.2",
}

# A mapping from import name to package name (on PyPI) for packages where these two names are different.
INSTALL_MAPPING = {
    "sqlalchemy": "SQLAlchemy",
    "tables": "pytables",
}

def get_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if not version:
        raise ImportError(f"Cannot determine the version for the module '{module.__name__}'.")
    if module.__name__ == "psycopg2":
        # psycopg2 appends extra strings to its version; split to get the actual version.
        version = version.split()[0]
    return version

def import_optional_dependency(
    name: str,
    extra: str = "",
    min_version: Optional[str] = None,
    *,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> Optional[types.ModuleType]:
    """
    Import an optional dependency.

    If a dependency is missing, raises an ImportError with a descriptive message.
    If the dependency is present but too old, either raises, warns, or ignores the issue based on `errors`.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.
    min_version : str, default None
        Minimum required version. Defaults to the global minimum if not provided.

    Returns
    -------
    Optional[ModuleType]
        The imported module, if found and version requirements are met. Otherwise, None.
    """
    assert errors in {"warn", "raise", "ignore"}, "Invalid error handling strategy specified."

    package_name = INSTALL_MAPPING.get(name, name)
    msg = (
        f"Missing optional dependency '{package_name}'. {extra} "
        f"Use pip or conda to install {package_name}."
    )

    try:
        module = importlib.import_module(name)
    except ImportError as err:
        if errors == "raise":
            raise ImportError(msg) from err
        return None

    # Retrieve the module's parent if it's a submodule
    parent = name.split(".")[0]
    module_to_check = sys.modules.get(parent, module)

    # Check version compatibility
    minimum_version = min_version or VERSIONS.get(parent)
    if minimum_version:
        try:
            installed_version = Version(get_version(module_to_check))
        except InvalidVersion:
            raise ImportError(f"Invalid version for the module '{parent}'.")
        required_version = Version(minimum_version)
        
        if installed_version < required_version:
            version_msg = (
                f"'{parent}' version '{installed_version}' is installed; "
                f"Dask requires version '{required_version}' or newer."
            )
            if errors == "warn":
                warnings.warn(version_msg, UserWarning)
                return None
            elif errors == "raise":
                raise ImportError(version_msg)
            else:
                return None

    return module
