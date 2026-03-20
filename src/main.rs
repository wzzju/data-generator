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
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
	let cli = cmd::CliCmd::parse();

	// -- Parse token range
	let token_range = token_range::TokenRange::parse(&cli.token_range)?;

	// -- Validate inputs
	if cli.count == 0 {
		return Err(Error::custom("Count must be greater than 0"));
	}

	// -- Initialize Rayon thread pool
	let num_jobs = cli.jobs.unwrap_or_else(num_cpus::get);
	generator::init_thread_pool(cli.jobs)?;
	println!("Using {num_jobs} parallel worker thread(s)");

	// -- Load tokenizer
	println!("Loading tokenizer from: {}", cli.tokenizer.display());
	let tokenizer = generator::load_tokenizer(&cli.tokenizer)?;

	// -- Load corpus
	println!("Loading corpus from {} file(s)...", cli.input.len());
	let corpus_files = generator::load_corpus(&cli.input).await?;
	let total_chars: usize = corpus_files.iter().map(|c| c.len()).sum();
	println!(
		"Corpus loaded: {} file(s), {total_chars} characters total",
		corpus_files.len(),
	);

	// -- Print generation info
	println!(
		"Generating {} entries with token range: min={}, max={}, avg={}",
		cli.count, token_range.min, token_range.max, token_range.avg
	);

	// -- Build generator config (using Arc for shared ownership across threads)
	let config = generator::GeneratorConfig {
		corpus_files: Arc::new(corpus_files),
		tokenizer: Arc::new(tokenizer),
		token_range,
		count: cli.count,
	};

	// -- Generate dataset
	match cli.format {
		cmd::OutputFormat::Aiak => {
			generator::generate_aiak(&config, &cli.output).await?;
			println!("AIAK dataset written to: {}", cli.output.display());
		}
		cmd::OutputFormat::Bench => {
			generator::generate_bench(&config, &cli.output).await?;
			println!("Bench dataset written to: {}", cli.output.display());
		}
	}

	Ok(())
}
