Write succint, to the point code
Avoid unnecessary early outs and other complications
After a refactor, recursively follow up with cleanup to avoid code structure we would not create from scratch.
After changing code in hot loops, run a performance comparision.

Committing:
Each commit should pass: `cargo test`

