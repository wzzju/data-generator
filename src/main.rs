// region:    --- Modules

mod cmd;
mod error;
mod generator;
mod token_range;

pub use error::{Error, Result};

// endregion: --- Modules

// NOTE: `Parser` trait must be in scope for `CliCmd::parse()` to be callable,
// because `parse()` is a trait method, not an inherent method.
use clap::Parser as _;

fn main() -> Result<()> {
	let cli = cmd::CliCmd::parse();

	// -- Parse token range
	let token_range = token_range::TokenRange::parse(&cli.token_range)?;

	// -- Validate inputs
	if cli.count == 0 {
		return Err(Error::custom("Count must be greater than 0"));
	}

	// -- Load tokenizer
	println!("Loading tokenizer from: {}", cli.tokenizer.display());
	let tokenizer = generator::load_tokenizer(&cli.tokenizer)?;

	// -- Load corpus
	println!(
		"Loading corpus from {} file(s)...",
		cli.input.len()
	);
	let corpus = generator::load_corpus(&cli.input)?;
	println!("Corpus loaded: {} characters", corpus.len());

	// -- Print generation info
	println!(
		"Generating {} entries with token range: min={}, max={}, avg={}",
		cli.count, token_range.min, token_range.max, token_range.avg
	);

	// -- Build generator config
	let config = generator::GeneratorConfig {
		corpus: &corpus,
		tokenizer: &tokenizer,
		token_range: &token_range,
		count: cli.count,
	};

	// -- Generate dataset
	match cli.format {
		cmd::OutputFormat::Aiak => {
			generator::generate_aiak(&config, &cli.output)?;
			println!("AIAK dataset written to: {}", cli.output.display());
		}
		cmd::OutputFormat::Bench => {
			generator::generate_bench(&config, &cli.output)?;
			println!("Bench dataset written to: {}", cli.output.display());
		}
	}

	Ok(())
}
