#!/usr/bin/env bash
# User-local build and install; never installs system packages or grants capabilities.
set -euo pipefail
if [[ $# -gt 2 ]]; then
  echo 'usage: bash scripts/install.sh [PREFIX] [BUILD_DIRECTORY]' >&2
  exit 2
fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
prefix=${1:-"$source_dir/install"}
build_dir=${2:-"$source_dir/build-install"}
cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel "${REQSCOPE_BUILD_JOBS:-4}"
# BPF compilation is optional for users without a BPF-capable clang.
if [[ ${REQSCOPE_BUILD_BPF:-0} == 1 ]]; then
  cmake --build "$build_dir" --target reqscope_bpf_program
fi
ctest --test-dir "$build_dir" --output-on-failure
cmake --install "$build_dir" --prefix "$prefix"
"$prefix/bin/reqscope" --version
printf 'Installed to %s\n' "$prefix"
printf 'The vLLM plugin is installed separately into your selected Python environment.\n'
