"""
NuRec gRPC package - dynamically compiles proto files if needed.

This package provides the generated gRPC client and server code. When imported,
it automatically checks if the generated Python files exist and compiles the
proto files if necessary.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Determine package directories
_PACKAGE_DIR = Path(__file__).parent.resolve()
_PROTO_DIR = _PACKAGE_DIR / "proto"


def _find_proto_files() -> List[Path]:
    """Find all .proto files in the proto directory."""
    if not _PROTO_DIR.exists():
        return []
    return list(_PROTO_DIR.glob("*.proto"))


def _get_generated_files(proto_file: Path) -> Tuple[Path, Path]:
    """Get the expected generated file paths for a proto file."""
    base_name = proto_file.stem
    return (_PACKAGE_DIR / f"{base_name}_pb2.py", _PACKAGE_DIR / f"{base_name}_pb2_grpc.py")


def _check_compilation_needed(proto_file: Path) -> bool:
    """Check if proto compilation is needed."""
    pb2_file, pb2_grpc_file = _get_generated_files(proto_file)

    if not pb2_file.exists() or not pb2_grpc_file.exists():
        return True

    try:
        proto_mtime = proto_file.stat().st_mtime
        pb2_mtime = pb2_file.stat().st_mtime
        pb2_grpc_mtime = pb2_grpc_file.stat().st_mtime
        return proto_mtime > pb2_mtime or proto_mtime > pb2_grpc_mtime
    except OSError:
        return True


def _compile_proto(proto_file: Path) -> bool:
    """Compile a proto file using grpc_tools.protoc."""
    try:
        from grpc_tools import protoc
    except ImportError:
        print("Error: grpcio-tools not installed. Please install with: pip install grpcio-tools", file=sys.stderr)
        return False

    pb2_file, pb2_grpc_file = _get_generated_files(proto_file)

    args = [
        "-I",
        str(_PROTO_DIR),
        "--python_out",
        str(_PACKAGE_DIR),
        "--grpc_python_out",
        str(_PACKAGE_DIR),
        str(proto_file.relative_to(_PROTO_DIR)),
    ]

    try:
        ret = protoc.main(["protoc"] + args)
        return ret == 0
    except Exception as e:
        print(f"Error compiling {proto_file}: {e}", file=sys.stderr)
        return False


def _load_generated_module(module_name: str, file_path: Path) -> object:
    """Load a generated Python module from file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Failed to load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_proto_compiled() -> None:
    """Ensure all proto files are compiled."""
    proto_files = _find_proto_files()

    if not proto_files:
        print(f"Warning: No .proto files found in {_PROTO_DIR}", file=sys.stderr)
        return

    for proto_file in proto_files:
        if _check_compilation_needed(proto_file):
            print(f"Compiling {proto_file.name}...", file=sys.stderr)
            if not _compile_proto(proto_file):
                raise ImportError(f"Failed to compile {proto_file}")


# Compile proto files if needed
_ensure_proto_compiled()

# Load all generated modules
_proto_files = _find_proto_files()
__all__ = []

for proto_file in _proto_files:
    base_name = proto_file.stem
    pb2_file, pb2_grpc_file = _get_generated_files(proto_file)

    if not pb2_file.exists() or not pb2_grpc_file.exists():
        continue

    # Module names
    pb2_module_name = f"{base_name}_pb2"
    pb2_grpc_module_name = f"{base_name}_pb2_grpc"

    # Load pb2 module first and register as top-level for pb2_grpc to import
    pb2_module = _load_generated_module(pb2_module_name, pb2_file)
    sys.modules[pb2_module_name] = pb2_module
    sys.modules[f"{__name__}.{pb2_module_name}"] = pb2_module

    # Load pb2_grpc module
    pb2_grpc_module = _load_generated_module(pb2_grpc_module_name, pb2_grpc_file)
    sys.modules[f"{__name__}.{pb2_grpc_module_name}"] = pb2_grpc_module

    # Expose at package level
    globals()[pb2_module_name] = pb2_module
    globals()[pb2_grpc_module_name] = pb2_grpc_module
    __all__.extend([pb2_module_name, pb2_grpc_module_name])
