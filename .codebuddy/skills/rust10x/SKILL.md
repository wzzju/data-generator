---
name: rust10x
description: Use when the user asks for Rust编程最佳实践, Rust最佳编程实践, Rust最佳实践, Rust best practices, project structure guidance, error handling patterns, testing conventions, CLI layout, Cargo.toml conventions, comment region patterns, or derive alias usage in Rust projects.
---

# Rust10x Best Practices

This skill defines the mandatory coding standards and audit workflow for Rust projects.

## When to Use

- When starting a new Rust project or file
- When implementing features in Rust or modifying rust code
- When reviewing or refactoring existing Rust code

## Mandatory Audit Workflow

When writing or reviewing code, you MUST execute the audit steps in order and validate each rule in the references files.

### 1. General Rules & Safety
**Read [references/general.md](references/general.md)**
- **No `unwrap()`/`expect()`**: Use `?` or `unwrap_or`.
- **Edition 2024**: Use `if let` chains, inline format args (`println!("{var}")`), async closures.
- **Macro Imports**: Use `use crate::macro_name;` before usage.
- **File Structure**: Order types -> public impl -> private/support -> tests.

### 2. Error Handling
**Read [references/error-handling.md](references/error-handling.md)**
- **No `thiserror`/`anyhow`**: Use custom `Error` enum with `derive_more`.
- **Result Alias**: `pub type Result<T> = core::result::Result<T, Error>;`.
- **Structure**: `mod error` in `lib.rs`/`main.rs`, re-exported as `pub use error::{Error, Result};`.

### 3. Project Structure
**Read [references/project-structure.md](references/project-structure.md)**
- **Support Modules**: Use `mod support` for internal utilities.
- **Examples**: `examples/c01-simple.rs`, `c02-...`.
- **XP Projects**: `xp-name` for experiments.

### 4. Testing
**Read [references/testing.md](references/testing.md)**
- **Unit Tests**: `// region: Tests`, `#[cfg(test)] mod tests`.
- **Test Result**: `type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;`.
- **Test Support**: `src/_test_support` or `tests/test_support`.

### 5. CLI Development
**Read [references/cli.md](references/cli.md)**
- **Structure**: `src/cli/cmd.rs` (clap), `src/cli/executor.rs`, `src/handlers/` (logic).
- **Separation**: Handlers must not depend on `clap`.

### 6. Comments & Cargo
**Read [references/comments.md](references/comments.md) & [references/cargo.md](references/cargo.md)**
- **Regions**: `// region:    --- Name` ... `// endregion: --- Name`.
- **Sections**: `// -- Section Name`.
- **Dependencies**: Grouped by comments `# -- Group`.

### 7. Derive Aliases & Macros
**Read [references/macros.md](references/macros.md)**
- **derive_alias!**: Use `macro_rules_attribute` and keep aliases in `derive_aliases.rs`.
- **No overlap**: Do not combine aliases that expand to overlapping traits.

## TDD Compliance Workflow

Apply the writing-skills RED-GREEN-REFACTOR cycle to validate compliance and ensure every change has executable audit steps.

### RED
- Use a minimal code snippet or file to reproduce potential violations
- Record the specific violated rules and their reference files

### GREEN
- Fix violations by following the Mandatory Audit Workflow
- List the rule items that were corrected

### REFACTOR
- Check for any rule categories that remain uncovered
- If new violation types appear, add them back into the audit steps

## Quick Reference: Common Violations to Fix

- ❌ `println!("Val: {}", val)` -> ✅ `println!("Val: {val}")`
- ❌ `.unwrap()` -> ✅ `?` or `.ok_or(...)?`
- ❌ `anyhow::Result` -> ✅ `crate::Error` (or `Box<dyn Error>` in tests)
- ❌ `use my_macro!` -> ✅ `use my_crate::my_macro; my_macro!(...)`
- ❌ `#[derive(Cmp!, Hash!)]` -> ✅ use the broader alias, add missing derives explicitly

## Enforcement Rules

- Any output that skips the Mandatory Audit Workflow is non-compliant
- Changes that do not complete the TDD Compliance Workflow must not be submitted or merged
