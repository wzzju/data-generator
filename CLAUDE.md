# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this codebase.

## Project Overview

A Rust CLI tool (`data-generator`) that randomly extracts text from corpus files to generate datasets in AIAK (JSON multi-turn conversations) or Bench (JSONL prompts) format, with precise token count control via HuggingFace Tokenizers.

## Build & Test Commands

```bash
# Build (debug)
cargo build

# Build (release, optimized)
cargo build --release

# Run all tests
cargo test

# Run a specific test
cargo test test_parse_max_only

# Lint check
cargo clippy -- -W clippy::all

# Format check
cargo fmt --check
```

**Rust Edition**: 2024 (requires rustc ≥ 1.85.0)

## Project Architecture

```
src/
├── main.rs         # Entry point: CLI parsing, orchestration
├── cmd.rs          # CLI argument definitions (clap derive)
├── error.rs        # Custom Error enum + Result alias (derive_more)
├── generator.rs    # Core logic: corpus loading, tokenization, text extraction, dataset generation
└── token_range.rs  # Token range string parsing ([min-]max[:avg])
```

### Module Responsibilities

- **`cmd.rs`**: Defines `CliCmd` (clap `Parser`) and `OutputFormat` enum (`Aiak`/`Bench`). All CLI args: `-i`, `-f`, `-o`, `-t`, `-r`, `-c`, `-j`.
- **`error.rs`**: Single `Error` enum with `Custom`, `InvalidTokenRange`, `Io`, `SerdeJson`, `TokenizerError` variants. Uses `derive_more` (`Display`, `Error`, `From`) instead of `thiserror`/`anyhow`.
- **`token_range.rs`**: Parses `[min-]max[:avg]` format strings into `TokenRange { min, max, avg }`.
- **`generator.rs`**: Contains all generation logic — corpus loading (async via Tokio), tokenizer loading, sentence boundary snapping, binary-search token extraction, normal distribution target generation, AIAK/Bench output writing (async via Tokio). Entry generation is parallelized using Rayon.

### Concurrency Model

- **Tokio async runtime** (`#[tokio::main]`): Handles all file IO (reading corpus, writing output).
- **Rayon parallel iterators**: Parallelize CPU-intensive entry generation (`tokenizer.encode()`, binary-search extraction). Each entry is generated independently; entries within an AIAK dataset are unordered, but multi-turn conversations within a single entry are generated sequentially to preserve turn ordering.
- **`-j` / `--jobs`**: Controls Rayon thread pool size. Defaults to number of CPU cores via `num_cpus`.
- **`Arc`-wrapped shared state**: `corpus_files` and `tokenizer` are wrapped in `Arc` for safe sharing across Rayon threads.

## Code Style & Conventions

** Please be sure to strictly follow Rust's best programming practices.**

### Region Comments

Use `// region:` and `// endregion:` comments to organize code sections:
```rust
// region:    --- Types
...
// endregion: --- Types
```

### Error Handling

- Use `derive_more` (`Display`, `Error`, `From`) for the `Error` enum — **not** `thiserror` or `anyhow`.
- Propagate errors via `?` operator with the crate-level `Result<T>` alias.
- The `Custom` variant accepts `String`, `&String`, and `&str` via `#[from(...)]`.
- External errors (`std::io::Error`, `serde_json::Error`, `tokenizers::Error`) have dedicated variants with `#[from]`.
- `Error::custom()` and `Error::custom_from_err()` for ad-hoc errors.

### Naming & Style

- Module-level `pub` exports go through `main.rs` (`pub use error::{Error, Result}`).
- Trait imports used only for method dispatch get `as _` suffix: `use clap::Parser as _;`.
- `unsafe` code is **forbidden**: `[lints.rust] unsafe_code = "forbid"`.

### Testing

- Tests live in `#[cfg(test)] mod tests` at the bottom of each file.
- Test modules use their own `Result<T>` type alias: `type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;`.

### Dependencies (Cargo.toml)

- Dependencies are grouped with `# -- Category` comments: `# -- Cli`, `# -- Json`, `# -- Tokenizer`, `# -- Random`, `# -- Others`.
- When upgrading deps, keep the old version as a comment above the new line for reference.

### Key Algorithms

- **Sentence boundary snapping**: `snap_to_sentence_start()` looks backward (up to 200 chars) for sentence terminators (`。！？!?.\\n`), then forward if none found, ensuring extracted text begins at a complete sentence.
- **Token-precise extraction**: Binary search over character count to find text snippets matching target token count within `[min, max]` range.
- **Normal distribution targets**: Box-Muller transform generates normally-distributed token targets centered on `avg` with `stddev = (max - min) / 6`.

## Git Conventions

### Commit Message Format

Use prefix conventions:
- `+` for new features/initial: `+ feature - Description`
- `^` for improvements/upgrades: `^ area - Description`
- `-` for removals/fixes: `- area - Description`

### .gitignore

The `demo/` and `.codebuddy/` directories are excluded from git tracking.
