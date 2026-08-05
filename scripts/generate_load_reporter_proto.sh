#!/usr/bin/env bash
# scripts/generate_load_reporter_proto.sh
#
# Regenerate Python gRPC bindings from the canonical LoadMonitorService proto.
#
# Usage:
#   bash scripts/generate_load_reporter_proto.sh          # regenerate in-tree
#   bash scripts/generate_load_reporter_proto.sh --check  # verify reproducibility, exit 1 on diff
#
# The canonical IDL lives at:
#   proto/sglang/router/loadmonitor/v1/load_monitor.proto
#
# Output Python files (kept at their existing import path):
#   python/sglang/srt/load_reporter/proto/load_monitor_pb2.py
#   python/sglang/srt/load_reporter/proto/load_monitor_pb2_grpc.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_SRC="sglang/router/loadmonitor/v1/load_monitor.proto"
DEST_DIR="${REPO_ROOT}/python/sglang/srt/load_reporter/proto"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GRPCIO_TOOLS_VERSION="1.78.0"
PROTOBUF_GENCODE_VERSION="6.31.1"

installed_grpcio_tools_version="$("${PYTHON_BIN}" -c \
    'from importlib.metadata import version; print(version("grpcio-tools"))' \
    2>/dev/null || true)"
if [[ "${installed_grpcio_tools_version}" != "${GRPCIO_TOOLS_VERSION}" ]]; then
    echo "ERROR: load reporter codegen requires grpcio-tools==${GRPCIO_TOOLS_VERSION}; found ${installed_grpcio_tools_version:-not installed}." >&2
    echo "Set PYTHON_BIN to a Python environment containing the pinned generator." >&2
    exit 1
fi

# Absolute import alias that grpc_tools emits for this proto path.
# We replace it with a relative import so the existing import path is stable.
ABS_ALIAS="sglang_dot_router_dot_loadmonitor_dot_v1_dot_load__monitor__pb2"
REL_ALIAS="load__monitor__pb2"
ABS_IMPORT_LINE="from sglang.router.loadmonitor.v1 import load_monitor_pb2 as ${ABS_ALIAS}"
REL_IMPORT_LINE="from . import load_monitor_pb2 as ${REL_ALIAS}"

CHECK_MODE=false
if [[ "${1-}" == "--check" ]]; then
    CHECK_MODE=true
fi

# ---------------------------------------------------------------------------
# Generate into a temp directory
# ---------------------------------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

(
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m grpc_tools.protoc \
        -I proto \
        --python_out="${TMP_DIR}" \
        --grpc_python_out="${TMP_DIR}" \
        "${PROTO_SRC}"
)

GENERATED_PB2="${TMP_DIR}/sglang/router/loadmonitor/v1/load_monitor_pb2.py"
GENERATED_GRPC="${TMP_DIR}/sglang/router/loadmonitor/v1/load_monitor_pb2_grpc.py"

if [[ ! -f "${GENERATED_PB2}" || ! -f "${GENERATED_GRPC}" ]]; then
    echo "ERROR: protoc did not produce expected output files in ${TMP_DIR}" >&2
    exit 1
fi

grep -q "^# Protobuf Python Version: ${PROTOBUF_GENCODE_VERSION}$" "${GENERATED_PB2}" \
    || { echo "ERROR: generated protobuf code does not target ${PROTOBUF_GENCODE_VERSION}" >&2; exit 1; }
grep -q "^GRPC_GENERATED_VERSION = '${GRPCIO_TOOLS_VERSION}'$" "${GENERATED_GRPC}" \
    || { echo "ERROR: generated gRPC code does not target ${GRPCIO_TOOLS_VERSION}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Post-process: rewrite absolute import → relative import in the grpc file
# ---------------------------------------------------------------------------
sed -i \
    -e "s|${ABS_IMPORT_LINE}|${REL_IMPORT_LINE}|g" \
    -e "s|${ABS_ALIAS}|${REL_ALIAS}|g" \
    "${GENERATED_GRPC}"

grep -q "^from \. import load_monitor_pb2" "${GENERATED_GRPC}" \
    || { echo "ERROR: import rewrite produced no match — grpc_tools alias may have changed" >&2; exit 1; }

# ---------------------------------------------------------------------------
# --check mode: diff generated files against committed files, no writes
# ---------------------------------------------------------------------------
if [[ "${CHECK_MODE}" == true ]]; then
    echo "==> --check mode: comparing generated files against committed files"
    DIFF_FOUND=false

    for basename in load_monitor_pb2.py load_monitor_pb2_grpc.py; do
        generated="${TMP_DIR}/sglang/router/loadmonitor/v1/${basename}"
        committed="${DEST_DIR}/${basename}"
        if ! diff -u "${committed}" "${generated}"; then
            echo "ERROR: ${basename} differs from committed version" >&2
            DIFF_FOUND=true
        fi
    done

    if [[ "${DIFF_FOUND}" == true ]]; then
        echo ""
        echo "Run 'bash scripts/generate_load_reporter_proto.sh' to regenerate." >&2
        exit 1
    else
        echo "OK: generated files match committed files."
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal mode: copy generated files into the destination package
# ---------------------------------------------------------------------------
cp "${GENERATED_PB2}"  "${DEST_DIR}/load_monitor_pb2.py"
cp "${GENERATED_GRPC}" "${DEST_DIR}/load_monitor_pb2_grpc.py"

echo "Regenerated:"
echo "  ${DEST_DIR}/load_monitor_pb2.py"
echo "  ${DEST_DIR}/load_monitor_pb2_grpc.py"
