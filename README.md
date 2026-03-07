# quadrs

Rust quad remeshing prototype based on Instant Meshes.

Created by Codex 5.4.

## Run

Build:

```bash
cargo build --release --bins
```

Remesh an OBJ file:

```bash
cargo run --release --bin remesh -- input.obj -o output.obj --target-faces 3000
```

Useful tools:

```bash
cargo run --release --bin mesh-stats -- output.obj
cargo run --release --bin preprocess -- input.obj -o preprocessed.obj --target-faces 3000
```

## Reference

Run the C++ reference implementation directly:

```bash
scripts/run_reference.sh -d -i -b -f 3000 -o output.obj input.obj
```

`scripts/run_reference.sh` will:

1. Check for an existing Instant Meshes binary.
2. If missing, clone the reference repository into `/tmp/instant-meshes`.
3. Build it into `/tmp/instant-meshes-build`.
4. Run it with the arguments you provide.

## Sources

Original paper:

https://igl.ethz.ch/projects/instant-meshes/instant-meshes-SA-2015-jakob-et-al-compressed.pdf

Reference implementation:

https://github.com/wjakob/instant-meshes
