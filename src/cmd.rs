use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// A CLI tool to generate datasets from text corpus files.
///
/// Reads txt corpus files and randomly extracts text to produce
/// aiak (JSON) or bench (JSONL) format datasets with precise
/// token count control via HuggingFace tokenizers.
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct CliCmd {
	/// Input text corpus files (at least one required)
	#[arg(short, long, required = true, num_args = 1..)]
	pub input: Vec<PathBuf>,

	/// Output format: aiak (json) or bench (jsonl)
	#[arg(short, long)]
	pub format: OutputFormat,

	/// Output file path
	#[arg(short, long)]
	pub output: PathBuf,

	/// Path to the HuggingFace tokenizer.json file
	#[arg(short, long)]
	pub tokenizer: PathBuf,

	/// Token range specification.
	///
	/// Syntax: [min-]max[:avg]
	///   - "300"         => max=300
	///   - "100-300"     => min=100, max=300
	///   - "100-300:200" => min=100, max=300, avg=200
	///   - "300:200"     => max=300, avg=200
	#[arg(short = 'r', long = "token-range")]
	pub token_range: String,

	/// Number of dataset entries to generate (default: 10)
	#[arg(short, long, default_value_t = 10)]
	pub count: usize,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
	/// AIAK format: JSON array with multi-turn conversations
	Aiak,
	/// Bench format: JSONL with one prompt per line
	Bench,
}
