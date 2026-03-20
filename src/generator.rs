use crate::error::{Error, Result};
use crate::token_range::TokenRange;

// use rand::Rng;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use tokenizers::Tokenizer;

// region:    --- Types

/// A single conversation turn in the aiak format.
#[derive(Debug, Serialize, Deserialize)]
pub struct Conversation {
	pub from: String,
	pub value: String,
}

/// A single entry in the aiak dataset.
#[derive(Debug, Serialize, Deserialize)]
pub struct AiakEntry {
	pub id: String,
	pub conversations: Vec<Conversation>,
}

/// A single entry in the bench dataset.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchEntry {
	pub prompt: String,
}

/// Configuration for the dataset generator.
pub struct GeneratorConfig<'a> {
	pub corpus: &'a str,
	pub tokenizer: &'a Tokenizer,
	pub token_range: &'a TokenRange,
	pub count: usize,
}

// endregion: --- Types

// region:    --- Corpus Loading

/// Load and concatenate all input text files into a single corpus string.
pub fn load_corpus(paths: &[impl AsRef<Path>]) -> Result<String> {
	let mut corpus = String::new();

	for path in paths {
		let path = path.as_ref();
		if !path.exists() {
			return Err(Error::custom(format!(
				"Input file not found: {}",
				path.display()
			)));
		}
		let content = fs::read_to_string(path)?;
		if !corpus.is_empty() {
			corpus.push('\n');
		}
		corpus.push_str(&content);
	}

	if corpus.trim().is_empty() {
		return Err(Error::custom("All input files are empty"));
	}

	Ok(corpus)
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
	let tokenizer = Tokenizer::from_file(path)?;
	Ok(tokenizer)
}

// endregion: --- Corpus Loading

// region:    --- Token Counting

/// Count the number of tokens in a text using the given tokenizer.
fn count_tokens(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
	let encoding = tokenizer
		.encode(text, false)
		.map_err(|e| Error::custom(e.to_string()))?;
	Ok(encoding.get_ids().len())
}

// endregion: --- Token Counting

// region:    --- Sentence Boundary

/// Characters that mark the end of a sentence.
const SENTENCE_TERMINATORS: &[char] = &['。', '！', '？', '!', '?', '.', '\n'];

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

	// -- If start already sits right after a terminator (or at line beginning), keep it
	if start > 0 && SENTENCE_TERMINATORS.contains(&corpus_chars[start - 1]) {
		return start;
	}

	let max_lookback = 200;
	let search_start = start.saturating_sub(max_lookback);

	// -- Scan backward for the nearest sentence terminator
	for i in (search_start..start).rev() {
		if SENTENCE_TERMINATORS.contains(&corpus_chars[i]) {
			let candidate = i + 1;
			// -- Skip any trailing whitespace after the terminator
			let mut pos = candidate;
			while pos < corpus_chars.len() && corpus_chars[pos].is_whitespace() {
				pos += 1;
			}
			return pos.min(corpus_chars.len());
		}
	}

	// -- No terminator found, try scanning forward instead
	let max_lookforward = 200;
	let search_end = (start + max_lookforward).min(corpus_chars.len());
	for i in start..search_end {
		if SENTENCE_TERMINATORS.contains(&corpus_chars[i]) {
			let candidate = i + 1;
			let mut pos = candidate;
			while pos < corpus_chars.len() && corpus_chars[pos].is_whitespace() {
				pos += 1;
			}
			return pos.min(corpus_chars.len());
		}
	}

	// -- Absolute fallback: use original start
	start
}

// endregion: --- Sentence Boundary

// region:    --- Text Extraction

/// Extract a text snippet from the corpus that has approximately
/// the target number of tokens (within min..=max).
///
/// Strategy:
/// 1. Pick a random start position (char-aligned).
/// 2. Estimate initial char count from target token count.
/// 3. Binary-search/adjust to land within [min, max] tokens.
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

	// -- Estimate chars per token (rough: ~1.5 chars per token for Chinese, ~4 for English)
	let estimated_chars = target_tokens * 2;

	// -- Try multiple times to find a valid snippet
	for _ in 0..100 {
		let max_start = corpus_len.saturating_sub(estimated_chars);
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

		// -- Binary search for the right char count
		let mut low = target_tokens.saturating_sub(target_tokens / 2);
		let mut high = (estimated_chars * 2).min(corpus_len - start);

		let mut best_text = None;
		let mut best_diff = usize::MAX;

		for _ in 0..30 {
			if low > high {
				break;
			}
			let mid = (low + high) / 2;
			let end = (start + mid).min(corpus_len);
			let text: String = corpus_chars[start..end].iter().collect();
			let token_count = count_tokens(tokenizer, &text)?;

			if token_count >= min_tokens && token_count <= max_tokens {
				let diff = token_count.abs_diff(target_tokens);
				if diff < best_diff {
					best_diff = diff;
					best_text = Some(text);
				}
				// -- Good enough if within 10% of target
				if diff <= target_tokens / 10 {
					break;
				}
			}

			if token_count < target_tokens {
				low = mid + 1;
			} else {
				high = mid.saturating_sub(1);
			}
		}

		if let Some(text) = best_text {
			return Ok(text);
		}
	}

	// -- Fallback: just take a chunk and truncate/extend until valid
	let raw_start = rng.random_range(0..corpus_len.max(1));
	let start = snap_to_sentence_start(&corpus_chars, raw_start);
	let end = (start + estimated_chars).min(corpus_len);
	let text: String = corpus_chars[start..end].iter().collect();
	Ok(text)
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

		if let Some((h_text, h_char_len)) = human_text {
			// -- Extract gpt text from right after human text
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
	}

	// -- Fallback: extract independently
	let h = extract_text_with_tokens(corpus, tokenizer, human_tokens, min_tokens, max_tokens)?;
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
pub fn generate_bench(config: &GeneratorConfig, output: impl AsRef<Path>) -> Result<()> {
	let file = fs::File::create(output.as_ref())?;
	let mut writer = BufWriter::new(file);

	for i in 0..config.count {
		let target = generate_target_tokens(config.token_range);
		let text = extract_text_with_tokens(
			config.corpus,
			config.tokenizer,
			target,
			config.token_range.min,
			config.token_range.max,
		)?;

		let entry = BenchEntry { prompt: text };
		let json = serde_json::to_string(&entry)?;
		writer.write_all(json.as_bytes())?;
		if i < config.count - 1 {
			writer.write_all(b"\n")?;
		}
	}

	writer.flush()?;
	Ok(())
}

// endregion: --- Bench Generation

// region:    --- Aiak Generation

/// Generate an aiak-format dataset (JSON) and write to the output path.
///
/// Each entry has an id and a conversations array with at least one
/// human-gpt pair. The total tokens across all turns in one entry
/// must fall within the specified token range.
pub fn generate_aiak(config: &GeneratorConfig, output: impl AsRef<Path>) -> Result<()> {
	let mut entries: Vec<AiakEntry> = Vec::with_capacity(config.count);
	let mut rng = rand::rng();

	for _ in 0..config.count {
		let total_target = generate_target_tokens(config.token_range);

		// -- Decide number of conversation rounds (1-3 pairs)
		let max_rounds = if total_target >= 100 { 3 } else if total_target >= 40 { 2 } else { 1 };
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
			let human_max = config.token_range.max;

			let (human_text, gpt_text) = extract_text_pair(
				config.corpus,
				config.tokenizer,
				human_tokens.max(1),
				gpt_tokens.max(1),
				human_min,
				human_max,
			)?;

			let h_tokens = count_tokens(config.tokenizer, &human_text)?;
			let g_tokens = count_tokens(config.tokenizer, &gpt_text)?;
			actual_total_tokens += h_tokens + g_tokens;

			conversations.push(Conversation {
				from: "human".to_string(),
				value: human_text,
			});
			conversations.push(Conversation {
				from: "gpt".to_string(),
				value: gpt_text,
			});
		}

		// -- Verify total tokens are within range, retry if needed
		let entry = if actual_total_tokens >= config.token_range.min
			&& actual_total_tokens <= config.token_range.max
		{
			let id = generate_id();
			AiakEntry {
				id,
				conversations,
			}
		} else {
			// -- Retry with single round for simpler control
			let (human_text, gpt_text) = extract_text_pair(
				config.corpus,
				config.tokenizer,
				(total_target * 6 / 10).max(1),
				(total_target * 4 / 10).max(1),
				1,
				config.token_range.max,
			)?;
			let id = generate_id();
			AiakEntry {
				id,
				conversations: vec![
					Conversation {
						from: "human".to_string(),
						value: human_text,
					},
					Conversation {
						from: "gpt".to_string(),
						value: gpt_text,
					},
				],
			}
		};

		entries.push(entry);
	}

	// -- Write as pretty-printed JSON array
	let file = fs::File::create(output.as_ref())?;
	let writer = BufWriter::new(file);
	serde_json::to_writer_pretty(writer, &entries)?;

	Ok(())
}

// endregion: --- Aiak Generation

// region:    --- Support

/// Generate a target token count based on the token range,
/// using a normal-like distribution centered around avg.
fn generate_target_tokens(range: &TokenRange) -> usize {
	let mut rng = rand::rng();

	// -- Use Box-Muller transform for approximate normal distribution
	let u1: f64 = rng.random_range(0.0001f64..1.0);
	let u2: f64 = rng.random_range(0.0001f64..1.0);
let normal = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

	// -- Scale: stddev is roughly (max - min) / 6 to keep most values in range
	let stddev = (range.max as f64 - range.min as f64) / 6.0;
	let value = range.avg as f64 + normal * stddev;

	// -- Clamp to [min, max]
	let clamped = value.round() as isize;
	let clamped = clamped.max(range.min as isize).min(range.max as isize);
	clamped as usize
}

/// Distribute total tokens approximately equally across rounds,
/// with some random variation.
fn distribute_tokens(total: usize, rounds: usize) -> Vec<usize> {
	if rounds == 1 {
		return vec![total];
	}

	let mut rng = rand::rng();
	let base = total / rounds;
	let mut result: Vec<usize> = (0..rounds).map(|_| {
		let variation = rng.random_range(0..=(base / 4).max(1));
		if rng.random_bool(0.5) {
			base + variation
		} else {
			base.saturating_sub(variation)
		}
	}).collect();

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

/// Generate a short unique id for aiak entries.
fn generate_id() -> String {
	let uuid = uuid::Uuid::new_v4();
	let encoded = uuid.simple().to_string();
	// -- Take first 7 chars for a compact id
	format!("{}_{}", &encoded[..7], 0)
}

// endregion: --- Support
