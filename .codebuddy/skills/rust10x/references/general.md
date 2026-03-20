# General Rules & Best Practices

- When users start a new project without specifying "xp" or "library," assume it is a binary project.

- Never use `.unwrap()` and `.expect("...")` even in test or example codes. 
  - For test and example, use the `.ok_or("should have ...")?` scheme which works well and production safer with the ?.

- However, using the `.unwrap_or_..(..)` are completely ok and good practices when it fit the logic.

- In enum variants and struct fields, if there is a comment or attribute before the variant or field, add an empty line before it for readability.

- If no edition is specified, assume Edition 2024, and use if-let chains when possible.

- When using proc or declarative macros, make sure to import them with `use ...` rather than using the qualified name like `lib_name::macro_name!(...)` (this is a bad pattern).
    - So the good pattern for macros is:
    - First, import them like `use lib_name::macro_name;`
    - Then use `macro_name!(...)`
    - A more complete example:
        - Do not write:
            - `use lopdf::Document;`
            - `let dict = lopdf::dictionary! { "Title" => "My PDF", "Author" => "User" };`
        - Instead, write:
            - `use lopdf::{Document, dictionary};`
            - `let dict = dictionary! { "Title" => "My PDF", "Author" => "User" };`

## Single-File Code Structure

When writing or adding code to a file, follow this structure.

- Public types in that file, if any, should be at the top, from the "container" type(s) to leaf ones.
- If there are many types, put them in a code comment region called "Types" (see comments-best-practices.md for code comment regions).
- Then add the public function or struct implementations for this module.
- Then, if there are any private functions, implementations, or types for this module, put them in the "Support" code comment regions.
- Then, at the end, if appropriate, add the unit tests under the "Tests" code region.

## Iterator Implementation

When a user asks you to implement iterators for a type, implement:

impl IntoIterator for T  
impl IntoIterator for &T

Put them in a code comment region named `// region:    --- Iterator Implementations`, following the comment convention.

Before the `impl IntoIterator`, also add an `impl T { pub iter(&self) ... }` implementation block.

This way, all iterator-related implementations are inside the `Iterator Implementations` region (this section should be only for iterator implementations).

# Rust Edition 2024 & Modern Practices

If no Cargo.toml or no edition 2024 available, assume that it is edition 2024. 

Here are some important new guideline to follow when writing Rust code with modern best practices. 

Some are new features of Rust Edition 2024 and modern Rust that needs to be follows.

Make sure to use them when appropriate.

## Ref change

- In Rust 2024, explicit ref, ref mut, or mut annotations on a binding are only allowed if the pattern leading up to that binding is fully explicit (i.e. you did not rely on the so-called “match ergonomics”).

- So, In many cases where you previously might have written ref x, you no longer need it. 

- So avoid the `ref ..` all together. 

## if-let-chain is now supported, use it

IMPORTANT: Make sure to use this if-let-chain when using edition 2024 and above.

The Rust compiler now supports let-chain.

So, use this style when possible:

```rust
// NEW WAY with if-let-chain
if let Some(person) = maybe_person
    && let Some(name) = person.name()
    && name.contains("John")
    && name.len() > 4 {
    // do stuff with person and name
}
```

Rather than the old way of writing nested if statements.

```rust
// OLD WAY
if let Some(person) = maybe_person {
    if let Some(name) = person.name() {
        if name.contains("John") && name.len() > 4 {
            // do stuff with person and name
        }
    }
}
```

## Inline macro values

For `println!` `assert...!` and all of those macro that take string literal, now when simple variables they should be inline. 

So, do this `println!("Hello {name}")` rather than `println!("Hello {}", name)`

When composed variable name, then, keep it separate (for example `println!("Hello {}", person.name)` is still ok

## Future and IntoFuture

- The `Future` and `IntoFuture` traits are now part of the prelude.
- `IntoIterator` for `Box<[T]>`
    - Boxed slices implement `IntoIterator` in all editions.
    - Calls to `IntoIterator::into_iter` are hidden in editions prior to 2024 when using method-call syntax (i.e., `boxed_slice.into_iter()`). So, `boxed_slice.into_iter()` still resolves to `(&(*boxed_slice)).into_iter()` as it has before.
    - `boxed_slice.into_iter()` now calls `IntoIterator::into_iter` in Rust 2024.
- Cargo: Rust-version aware resolver
    - `edition = "2024"` implies `resolver = "3"` in Cargo.toml which enables a Rust-version aware dependency resolver.
- Cargo: Table/key consistency - now all with a `-` rather than `_` (e.g., `default-features`)
- Reject unused inherited default-features
    - default-features = false is no longer allowed in an inherited workspace dependency if the workspace dependency specifies default-features = true (or does not specify default-features).
- Async closures (see below)

## `async` closures

Rust now supports asynchronous closures like `async || {}`.  
New traits: `AsyncFn`, `AsyncFnMut`, `AsyncFnOnce`.

```rust
let mut vec: Vec<String> = vec![];

let closure = async || {
    vec.push(ready(String::from("")).await);
};
```

More info: [RFC 3668](https://rust-lang.github.io/rfcs/3668-async-closures.html), [Stabilization PR](https://github.com/rust-lang/rust/pull/132706)

## Hiding trait implementations from diagnostics

You can now use `#[diagnostic::do_not_recommend]` to suppress confusing trait suggestion messages.

More info: [RFC 2397](https://rust-lang.github.io/rfcs/2397-do-not-recommend.html), [Reference](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-diagnosticdo_not_recommend-attribute)

## `FromIterator` and `Extend` for tuples

Now supported for tuples of length 1 through 12. You can collect into multiple containers at once:

```rust
use std::collections::{LinkedList, VecDeque};

fn main() {
    let (squares, cubes, tesseracts): (Vec<_>, VecDeque<_>, LinkedList<_>) =
        (0i32..10).map(|i| (i * i, i.pow(3), i.pow(4))).collect();
}
```
