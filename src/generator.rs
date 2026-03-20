use crate::error::{Error, Result};
use crate::token_range::TokenRange;

use rand::RngExt;
use rayon::prelude::*;
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;

// region:    --- Types

/// A single conversation turn in the aiak format.
#[derive(Debug, Serialize)]
pub struct Conversation {
	pub from: String,
	pub value: String,
}

impl Conversation {
	/// Create a new conversation turn.
	fn new(from: &str, value: String) -> Self {
		Self {
			from: from.to_string(),
			value,
		}
	}
}

/// A single entry in the aiak dataset.
#[derive(Debug, Serialize)]
pub struct AiakEntry {
	pub id: String,
	pub conversations: Vec<Conversation>,
}

/// A single entry in the bench dataset.
#[derive(Debug, Serialize)]
pub struct BenchEntry {
	pub prompt: String,
}

/// Configuration for the dataset generator.
pub struct GeneratorConfig {
	/// Each element is the full text content of one input file.
	pub corpus_files: Arc<Vec<String>>,
	pub tokenizer: Arc<Tokenizer>,
	pub token_range: TokenRange,
	pub count: usize,
}

// endregion: --- Types

// region:    --- Corpus Loading

/// Load all input text files into a vector of strings (one per file).
///
/// Each file is kept as a separate corpus entry so that during generation
/// we can randomly pick one file per data entry.
///
/// Uses Tokio async IO for file reading.
pub async fn load_corpus(paths: &[impl AsRef<Path>]) -> Result<Vec<String>> {
	let mut corpus_files: Vec<String> = Vec::with_capacity(paths.len());

	for path in paths {
		let path = path.as_ref();
		if !path.exists() {
			return Err(Error::custom(format!(
				"Input file not found: {}",
				path.display()
			)));
		}
		let content = tokio::fs::read_to_string(path).await?;
		if content.trim().is_empty() {
			eprintln!("Warning: input file is empty, skipping: {}", path.display());
			continue;
		}
		corpus_files.push(content);
	}

	if corpus_files.is_empty() {
		return Err(Error::custom("All input files are empty"));
	}

	Ok(corpus_files)
}

/// Load a HuggingFace tokenizer from a JSON file.
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
	let path = path.as_ref();
	if !path.exists() {
		return Err(Error::custom(format!(
			"Tokenizer file not found: {}",
			path.display()
		)));
	}
	Ok(Tokenizer::from_file(path)?)
}

// endregion: --- Corpus Loading

// region:    --- Token Counting

/// Count the number of tokens in a text using the given tokenizer.
fn count_tokens(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
	Ok(tokenizer.encode(text, false)?.get_ids().len())
}

// endregion: --- Token Counting

// region:    --- Sentence Boundary

/// Characters that mark the end of a sentence.
const SENTENCE_TERMINATORS: &[char] = &['。', '！', '？', '；', '!', '?', '.', '\n'];

/// Characters that mark a clause boundary (weaker than sentence terminators,
/// used as a fallback when no sentence terminator is found nearby).
const CLAUSE_BOUNDARIES: &[char] = &['，', '、', '：', '“', '"', ',', ':'];

/// Snap a start index forward (or backward) to the nearest sentence boundary
/// so that the extracted text begins at the start of a complete sentence.
///
/// Looks backward from `start` for the closest sentence-ending character,
/// then returns the position right after it. If no terminator is found
/// within `max_lookback` characters, falls back to `start`.
fn snap_to_sentence_start(corpus_chars: &[char], start: usize) -> usize {
	if start == 0 {
		return 0;
	}

	// -- If start already sits right after a terminator, keep it
	if SENTENCE_TERMINATORS.contains(&corpus_chars[start - 1]) {
		return start;
	}

	// -- Find the first non-whitespace, non-trailing-quote position after a terminator
	let skip_whitespace_after = |terminator_pos: usize| -> usize {
		const TRAILING_QUOTES: &[char] =
			&['\u{201d}', '\u{300d}', '\u{300f}', ')', '\u{ff09}'];
		let mut pos = terminator_pos + 1;
		while pos < corpus_chars.len()
			&& (corpus_chars[pos].is_whitespace()
				|| TRAILING_QUOTES.contains(&corpus_chars[pos]))
		{
			pos += 1;
		}
		pos.min(corpus_chars.len())
	};

	const MAX_LOOK: usize = 500;

	// -- Scan backward for the nearest sentence terminator
	let search_start = start.saturating_sub(MAX_LOOK);
	if let Some(pos) = corpus_chars[search_start..start]
		.iter()
		.rposition(|c| SENTENCE_TERMINATORS.contains(c))
	{
		return skip_whitespace_after(search_start + pos);
	}

	// -- No terminator found backward, try scanning forward
	let search_end = (start + MAX_LOOK).min(corpus_chars.len());
	if let Some(pos) = corpus_chars[start..search_end]
		.iter()
		.position(|c| SENTENCE_TERMINATORS.contains(c))
	{
		return skip_whitespace_after(start + pos);
	}

	// -- Still nothing: fall back to nearest clause boundary (comma, colon, etc.)
	let clause_start = start.saturating_sub(MAX_LOOK);
	if let Some(pos) = corpus_chars[clause_start..start]
		.iter()
		.rposition(|c| CLAUSE_BOUNDARIES.contains(c))
	{
		return skip_whitespace_after(clause_start + pos);
	}

	// -- Absolute fallback: use original start
	start
}

// endregion: --- Sentence Boundary

// region:    --- Text Extraction

/// Extract a text snippet from the corpus that has approximately
/// the target number of tokens (within min..=max).
///
/// Delegates to `extract_exact_tokens` with random start positions,
/// retrying up to 100 times before falling back to a raw chunk.
fn extract_text_with_tokens(
	corpus: &str,
	tokenizer: &Tokenizer,
	target_tokens: usize,
	min_tokens: usize,
	max_tokens: usize,
) -> Result<String> {
	let corpus_chars: Vec<char> = corpus.chars().collect();
	let corpus_len = corpus_chars.len();

	if corpus_len == 0 {
		return Err(Error::custom("Corpus is empty"));
	}

	let mut rng = rand::rng();
	let estimated_chars = target_tokens * 2;

	for _ in 0..100 {
		let max_start = corpus_len.saturating_sub(estimated_chars);
		let raw_start = if max_start > 0 {
			rng.random_range(0..max_start)
		} else {
			0
		};

		let start = snap_to_sentence_start(&corpus_chars, raw_start);
		if start >= corpus_len {
			continue;
		}

		if let Some((text, _)) = extract_exact_tokens(
			&corpus_chars,
			start,
			tokenizer,
			target_tokens,
			min_tokens,
			max_tokens,
		)? {
			return Ok(text);
		}
	}

	// -- Fallback: just take a raw chunk
	let raw_start = rng.random_range(0..corpus_len.max(1));
	let start = snap_to_sentence_start(&corpus_chars, raw_start);
	let end = (start + estimated_chars).min(corpus_len);
	Ok(corpus_chars[start..end].iter().collect())
}

/// Extract a contiguous text pair (human_text, gpt_text) from the corpus.
///
/// The human text starts at a random position, and the gpt text is
/// taken from immediately after the human text in the corpus.
fn extract_text_pair(
	corpus: &str,
	tokenizer: &Tokenizer,
	human_tokens: usize,
	gpt_tokens: usize,
	min_tokens: usize,
	max_tokens: usize,
) -> Result<(String, String)> {
	let corpus_chars: Vec<char> = corpus.chars().collect();
	let corpus_len = corpus_chars.len();

	let mut rng = rand::rng();

	// -- Estimate total chars needed
	let estimated_human_chars = human_tokens * 2;
	let estimated_gpt_chars = gpt_tokens * 2;
	let total_estimated = estimated_human_chars + estimated_gpt_chars;

	for _ in 0..100 {
		let max_start = corpus_len.saturating_sub(total_estimated);
		let raw_start = if max_start > 0 {
			rng.random_range(0..max_start)
		} else {
			0
		};

		// -- Align to the nearest sentence boundary
		let start = snap_to_sentence_start(&corpus_chars, raw_start);
		if start >= corpus_len {
			continue;
		}

		// -- Extract human text
		let human_text = extract_exact_tokens(
			&corpus_chars,
			start,
			tokenizer,
			human_tokens,
			min_tokens,
			max_tokens,
		)?;

		let Some((h_text, h_char_len)) = human_text else {
			continue;
		};

		// -- Extract gpt text from right after human text
		// NOTE: gpt value starts directly after human text, no sentence snapping needed.
		let gpt_start = start + h_char_len;
		if gpt_start >= corpus_len {
			continue;
		}

		let gpt_text = extract_exact_tokens(
			&corpus_chars,
			gpt_start,
			tokenizer,
			gpt_tokens,
			1, // gpt has no strict min requirement per turn
			max_tokens,
		)?;

		if let Some((g_text, _)) = gpt_text {
			return Ok((h_text, g_text));
		}
	}

	// -- Fallback: extract independently
	let h = extract_text_with_tokens(
		corpus,
		tokenizer,
		human_tokens,
		min_tokens,
		max_tokens,
	)?;
	let g = extract_text_with_tokens(corpus, tokenizer, gpt_tokens, 1, max_tokens)?;
	Ok((h, g))
}

/// Try to extract text from corpus_chars starting at `start` with
/// approximately `target_tokens`. Returns (text, char_length) if successful.
fn extract_exact_tokens(
	corpus_chars: &[char],
	start: usize,
	tokenizer: &Tokenizer,
	target_tokens: usize,
	min_tokens: usize,
	max_tokens: usize,
) -> Result<Option<(String, usize)>> {
	let corpus_len = corpus_chars.len();
	if start >= corpus_len {
		return Ok(None);
	}

	let available = corpus_len - start;
	let mut low = 0usize;
	let mut high = available;

	let mut best: Option<(String, usize, usize)> = None; // (text, char_len, diff)

	for _ in 0..30 {
		if low > high {
			break;
		}
		let mid = (low + high) / 2;
		if mid == 0 {
			low = 1;
			continue;
		}
		let end = start + mid;
		let text: String = corpus_chars[start..end].iter().collect();
		let token_count = count_tokens(tokenizer, &text)?;

		if token_count >= min_tokens && token_count <= max_tokens {
			let diff = token_count.abs_diff(target_tokens);
			let is_better = best.as_ref().is_none_or(|(_, _, bd)| diff < *bd);
			if is_better {
				best = Some((text, mid, diff));
			}
			if diff <= target_tokens / 10 + 1 {
				break;
			}
		}

		if token_count < target_tokens {
			low = mid + 1;
		} else {
			high = mid.saturating_sub(1);
		}
	}

	Ok(best.map(|(text, char_len, _)| (text, char_len)))
}

// endregion: --- Text Extraction

// region:    --- Bench Generation

/// Generate a bench-format dataset (JSONL) and write to the output path.
///
/// Entry generation is parallelized with Rayon; file IO uses Tokio async.
pub async fn generate_bench(
	config: &GeneratorConfig,
	output: impl AsRef<Path>,
) -> Result<()> {
	let corpus = Arc::clone(&config.corpus_files);
	let tok = Arc::clone(&config.tokenizer);
	let range = config.token_range.clone();
	let count = config.count;

	// -- Generate all entries in parallel using Rayon
	let entries: Vec<Result<BenchEntry>> = (0..count)
		.into_par_iter()
		.map(|_| {
			let mut rng = rand::rng();
			let file = &corpus[rng.random_range(0..corpus.len())];

			let target = generate_target_tokens(&range);
			let text =
				extract_text_with_tokens(file, &tok, target, range.min, range.max)?;

			Ok(BenchEntry { prompt: text })
		})
		.collect();

	// -- Write all entries asynchronously using Tokio
	let file = tokio::fs::File::create(output.as_ref()).await?;
	let mut writer = tokio::io::BufWriter::new(file);

	for entry_result in entries {
		let entry = entry_result?;
		let json = serde_json::to_string(&entry)?;
		writer.write_all(json.as_bytes()).await?;
		writer.write_all(b"\n").await?;
	}

	writer.flush().await?;
	Ok(())
}

// endregion: --- Bench Generation

// region:    --- Aiak Generation

/// Generate an aiak-format dataset (JSON) and write to the output path.
///
/// Each entry has an id and a conversations array with at least one
/// human-gpt pair. The total tokens across all turns in one entry
/// must fall within the specified token range.
///
/// Entry generation is parallelized with Rayon; file IO uses Tokio async.
/// Multi-turn conversations within a single entry are generated sequentially
/// to preserve ordering.
pub async fn generate_aiak(
	config: &GeneratorConfig,
	output: impl AsRef<Path>,
) -> Result<()> {
	let corpus = Arc::clone(&config.corpus_files);
	let tok = Arc::clone(&config.tokenizer);
	let range = config.token_range.clone();
	let count = config.count;

	// -- Generate all entries in parallel using Rayon
	let results: Vec<Result<AiakEntry>> = (0..count)
		.into_par_iter()
		.map(|_| generate_single_aiak_entry(&corpus, &tok, &range))
		.collect();

	// -- Collect results, propagating any errors
	let entries: Vec<AiakEntry> = results.into_iter().collect::<Result<Vec<_>>>()?;

	// -- Write as pretty-printed JSON array asynchronously using Tokio
	let json_bytes = serde_json::to_vec_pretty(&entries)?;
	tokio::fs::write(output.as_ref(), &json_bytes).await?;

	Ok(())
}

/// Generate a single AIAK entry (called from Rayon parallel iterator).
///
/// Multi-turn conversations are generated sequentially within each entry
/// to preserve turn ordering.
fn generate_single_aiak_entry(
	corpus_files: &[String],
	tokenizer: &Tokenizer,
	token_range: &TokenRange,
) -> Result<AiakEntry> {
	let mut rng = rand::rng();

	// -- Randomly pick one file for this entry
	let corpus = &corpus_files[rng.random_range(0..corpus_files.len())];

	let total_target = generate_target_tokens(token_range);

	// -- Decide number of conversation rounds (1-3 pairs)
	let max_rounds = if total_target >= 100 {
		3
	} else if total_target >= 40 {
		2
	} else {
		1
	};
	let num_rounds: usize = rng.random_range(1..=max_rounds);

	// -- Distribute tokens across rounds
	let tokens_per_round = distribute_tokens(total_target, num_rounds);

	let mut conversations: Vec<Conversation> = Vec::new();
	let mut actual_total_tokens = 0usize;

	for round_tokens in &tokens_per_round {
		// -- Split round tokens between human (~60%) and gpt (~40%)
		let human_tokens = (*round_tokens * 6) / 10;
		let gpt_tokens = round_tokens - human_tokens;

		let human_min = 1;
		let human_max = token_range.max;

		let (human_text, gpt_text) = extract_text_pair(
			corpus,
			tokenizer,
			human_tokens.max(1),
			gpt_tokens.max(1),
			human_min,
			human_max,
		)?;

		let h_tokens = count_tokens(tokenizer, &human_text)?;
		let g_tokens = count_tokens(tokenizer, &gpt_text)?;
		actual_total_tokens += h_tokens + g_tokens;

		conversations.push(Conversation::new("human", human_text));
		conversations.push(Conversation::new("gpt", gpt_text));
	}

	// -- Verify total tokens are within range, retry if needed
	if actual_total_tokens >= token_range.min
		&& actual_total_tokens <= token_range.max
	{
		let id = generate_id();
		Ok(AiakEntry { id, conversations })
	} else {
		// -- Retry with single round for simpler control
		let (human_text, gpt_text) = extract_text_pair(
			corpus,
			tokenizer,
			(total_target * 6 / 10).max(1),
			(total_target * 4 / 10).max(1),
			1,
			token_range.max,
		)?;
		let id = generate_id();
		Ok(AiakEntry {
			id,
			conversations: vec![
				Conversation::new("human", human_text),
				Conversation::new("gpt", gpt_text),
			],
		})
	}
}

// endregion: --- Aiak Generation

// region:    --- Support

/// Initialize the Rayon global thread pool with the specified number of threads.
///
/// If `jobs` is `None`, defaults to the number of available CPU cores.
pub fn init_thread_pool(jobs: Option<usize>) -> Result<()> {
	let num_threads = jobs.unwrap_or_else(num_cpus::get);
	rayon::ThreadPoolBuilder::new()
		.num_threads(num_threads)
		.build_global()
		.map_err(|e| {
			Error::custom(format!("Failed to build Rayon thread pool: {e}"))
		})?;
	Ok(())
}

/// Generate a target token count based on the token range,
/// using a normal-like distribution centered around avg.
fn generate_target_tokens(range: &TokenRange) -> usize {
	let mut rng = rand::rng();

	// -- Box-Muller transform for approximate normal distribution
	let u1: f64 = rng.random_range(0.0001f64..1.0);
	let u2: f64 = rng.random_range(0.0001f64..1.0);
	let normal =
		(-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

	// -- Scale: stddev ≈ (max - min) / 6 keeps most values in range
	let stddev = (range.max as f64 - range.min as f64) / 6.0;
	let value = (range.avg as f64 + normal * stddev)
		.round()
		.clamp(range.min as f64, range.max as f64);

	value as usize
}

/// Distribute total tokens approximately equally across rounds,
/// with some random variation.
fn distribute_tokens(total: usize, rounds: usize) -> Vec<usize> {
	if rounds == 1 {
		return vec![total];
	}

	let mut rng = rand::rng();
	let base = total / rounds;
	let mut result: Vec<usize> = (0..rounds)
		.map(|_| {
			let variation = rng.random_range(0..=(base / 4).max(1));
			if rng.random_bool(0.5) {
				base + variation
			} else {
				base.saturating_sub(variation)
			}
		})
		.collect();

	// -- Adjust the last element to ensure the sum equals total
	let current_sum: usize = result.iter().sum();
	if let Some(last) = result.last_mut() {
		if current_sum > total {
			*last = last.saturating_sub(current_sum - total);
		} else {
			*last += total - current_sum;
		}
	}

	result
}

/// Generate a short unique id for aiak entries (first 7 chars of UUIDv4).
fn generate_id() -> String {
	let id = uuid::Uuid::new_v4().simple().to_string();
	format!("{}_0", &id[..7])
}

// endregion: --- Support
