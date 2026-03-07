#!/usr/bin/env bash
set -euo pipefail

repo_dir="/tmp/instant-meshes"
build_dir="/tmp/instant-meshes-build"
binary="${build_dir}/Instant Meshes.app/Contents/MacOS/Instant Meshes"

jobs() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
    return
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  printf '4\n'
}

ensure_reference() {
  if [[ -x "${binary}" ]]; then
    return
  fi

  if [[ ! -d "${repo_dir}/.git" ]]; then
    git clone --recursive https://github.com/wjakob/instant-meshes "${repo_dir}"
  fi

  cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${build_dir}" --config Release -j "$(jobs)"
}

ensure_reference
exec "${binary}" "$@"
