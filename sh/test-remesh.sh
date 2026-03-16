#!/usr/bin/env bash
set -euo pipefail

mesh_dir="meshes"

mesh_name="${1:-}"
if [[ -z "${mesh_name}" ]]; then
  printf 'usage: %s <mesh-name> [remesh args...]\n' "${0}" >&2
  exit 1
fi
shift

mesh_url() {
  case "$1" in
    teapot) printf '%s\n' "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj" ;;
    *) return 1 ;;
  esac
}

download() {
  local url="$1"
  local target="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${url}" -o "${target}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "${target}" "${url}"
    return
  fi
  printf 'missing downloader: install curl or wget\n' >&2
  exit 1
}

input="${mesh_dir}/${mesh_name}.obj"
output="${OUTPUT:-${mesh_dir}/${mesh_name}-remeshed.obj}"

mkdir -p "${mesh_dir}"

if [[ ! -f "${input}" ]]; then
  if ! url="$(mesh_url "${mesh_name}")"; then
    printf 'missing mesh: %s\n' "${input}" >&2
    printf 'unknown downloadable mesh: %s\n' "${mesh_name}" >&2
    exit 1
  fi
  download "${url}" "${input}"
fi

cargo build --release --bin remesh

args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
  args=(--target-faces 3000)
fi

target/release/remesh "${input}" -o "${output}" "${args[@]}"
test -s "${output}"
printf 'wrote %s\n' "${output}"
