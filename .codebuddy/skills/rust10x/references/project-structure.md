# Project Structure & Organization Best Practices

## Struct Best Practices

Here are some best practices for `struct` types:

- If a struct property has a comment or attribute and is not the first property in the struct, add an empty line before it to improve clarity.

- When a file contains multiple types, place the main type at the top, with supporting types or functions below.

## Support / Utils Module Best Practices

In Rust10x, the best practice for crate or sub-module utilities is the following:

Typically, when generic utilities don't fit into a clean sub-module name, we use the `support` module namespace to compartmentalize them for the appropriate module tree.

We use `support.rs` (or `support/mod.rs`), depending on how large they are.

For example:

```rust
// region: --- Modules

mod support;

//.. more modules

// endregion: --- Modules
```

Then, the sub-modules will use that support.

Note that they are designed to NOT BE public and only at the level of the modules for which they are intended.

So, the types and functions in support will be `pub ...`, but since it is defined as `mod support;`, it will only be accessible to this module and its sub-modules by design.

When support is needed across all of the sub-modules of a crate, it might be at the root.

Otherwise, some sub-modules might need `support` as well, and they will repeat this pattern.

## Example Files Best Practices

- When creating example files, a good naming convention is to use chapters, guiding the user from simple to more complex examples.

- The convention is to have `examples/c01-simple.rs` for the simplest case.

- Then, for each topic, use `examples/c02-some-functionality.rs` for a given functionality.

- The goal is for each of these examples to focus on one aspect of functionality from the main crate or for learning about another crate (in the case of an `xp-project`).

The main signature will be:

- `fn main() -> Result<(), Box<dyn std::error::Error>>` (for example files, do not `use std::error::Error`; just use it this way)

- In most cases, we don't need a type alias for these, since it's only one file, and that will allow exporting the crate's `Result/Error` if needed.

## xp project Best Practices

- Sometimes the user may create an `xp-...`-like project, where `xp` stands for experiment or exploration.

- For example, if the goal of the `xp-...` project is to learn about `blake3`, the name is `xp-blake3`.

- So, this will be a lib project with a `src/lib.rs` file (empty to start with),

- Start with an empty `src/lib.rs` unless the user asks specifically.

- with `examples/c01-simple.rs` 

- if asked to be async, add the tokio cargo group, and make the example tokio async. 

- When doing async do not do `use tokio::main` but `#[tokio::main] async fn main...`

- The content of the `main` example function should be

```rust
println!("Hello World");

Ok(())
```
