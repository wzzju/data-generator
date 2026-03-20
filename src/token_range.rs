use crate::error::{Error, Result};

/// Parsed token range specification: [min-]max[:avg]
///
/// Examples:
///   - "300"         => min=1,   max=300, avg=150 (auto-calculated)
///   - "100-300"     => min=100, max=300, avg=200 (auto-calculated)
///   - "100-300:200" => min=100, max=300, avg=200
///   - "300:200"     => min=1,   max=300, avg=200
#[derive(Debug, Clone)]
pub struct TokenRange {
	pub min: usize,
	pub max: usize,
	pub avg: usize,
}

impl TokenRange {
	/// Parse token range from string with format: [min-]max[:avg]
	pub fn parse(input: &str) -> Result<Self> {
		let input = input.trim();
		if input.is_empty() {
			return Err(Error::InvalidTokenRange(
				"Token range string is empty".to_string(),
			));
		}

		// -- Split by ':' to separate the range part from the avg part
		let (range_part, avg_part) = match input.split_once(':') {
			Some((r, a)) => (r.trim(), Some(a.trim())),
			None => (input, None),
		};

		// -- Parse the range part: either "max" or "min-max"
		let (min, max) = if let Some((min_str, max_str)) = range_part.split_once('-')
		{
			let min = parse_usize(min_str.trim(), "min")?;
			let max = parse_usize(max_str.trim(), "max")?;
			(min, max)
		} else {
			let max = parse_usize(range_part, "max")?;
			(1, max)
		};

		// -- Parse the avg part
		let avg = match avg_part {
			Some(a) => parse_usize(a, "avg")?,
			None => (min + max) / 2,
		};

		// -- Validate constraints
		if min > max {
			return Err(Error::InvalidTokenRange(format!(
				"min ({min}) must be <= max ({max})"
			)));
		}
		if avg < min || avg > max {
			return Err(Error::InvalidTokenRange(format!(
				"avg ({avg}) must be between min ({min}) and max ({max})"
			)));
		}
		if max == 0 {
			return Err(Error::InvalidTokenRange(
				"max must be greater than 0".to_string(),
			));
		}

		Ok(Self { min, max, avg })
	}
}

// region:    --- Support

fn parse_usize(s: &str, field_name: &str) -> Result<usize> {
	s.parse::<usize>().map_err(|_| {
		Error::InvalidTokenRange(format!("Invalid {field_name} value: '{s}'"))
	})
}

// endregion: --- Support

// region:    --- Tests

#[cfg(test)]
mod tests {
	type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;
	use super::*;

	#[test]
	fn test_parse_max_only() -> Result<()> {
		let tr = TokenRange::parse("300")?;
		assert_eq!(tr.min, 1);
		assert_eq!(tr.max, 300);
		assert_eq!(tr.avg, 150);
		Ok(())
	}

	#[test]
	fn test_parse_min_max() -> Result<()> {
		let tr = TokenRange::parse("100-300")?;
		assert_eq!(tr.min, 100);
		assert_eq!(tr.max, 300);
		assert_eq!(tr.avg, 200);
		Ok(())
	}

	#[test]
	fn test_parse_full() -> Result<()> {
		let tr = TokenRange::parse("100-300:200")?;
		assert_eq!(tr.min, 100);
		assert_eq!(tr.max, 300);
		assert_eq!(tr.avg, 200);
		Ok(())
	}

	#[test]
	fn test_parse_max_with_avg() -> Result<()> {
		let tr = TokenRange::parse("300:200")?;
		assert_eq!(tr.min, 1);
		assert_eq!(tr.max, 300);
		assert_eq!(tr.avg, 200);
		Ok(())
	}

	#[test]
	fn test_parse_invalid_min_gt_max() -> Result<()> {
		let result = TokenRange::parse("500-300");
		assert!(result.is_err());
		Ok(())
	}

	#[test]
	fn test_parse_invalid_avg_out_of_range() -> Result<()> {
		let result = TokenRange::parse("100-300:50");
		assert!(result.is_err());
		Ok(())
	}

	#[test]
	fn test_parse_empty() -> Result<()> {
		let result = TokenRange::parse("");
		assert!(result.is_err());
		Ok(())
	}
}

// endregion: --- Tests
