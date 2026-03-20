//! # Error Module
//!
//! Defines the custom `Error` type and `Result` alias for the data-generator crate.
//! This follows the "no thiserror/anyhow" rule, using `derive_more` for boilerplate.

use derive_more::{Display, Error as DeriveError, From};

// region:    --- Result

/// Custom Result type for the data-generator crate.
pub type Result<T> = core::result::Result<T, Error>;

// endregion: --- Result

// region:    --- Error

/// Core error type for the data-generator crate.
///
/// Note on `From` derivation:
/// Using `#[from(...)]` on the `Custom` variant disables automatic `From` derivation
/// for other variants. Therefore, each variant that wraps an external error also
/// requires an explicit `#[from]`.
#[derive(Debug, Display, DeriveError, From)]
pub enum Error {
	/// Wraps a custom error message.
	///
	/// This variant is used for ad-hoc errors that don't fit into other variants.
	#[display("{_0}")]
	#[error(ignore)]
	#[from(String, &String, &str)]
	Custom(String),

	/// Returned when the token range specification is invalid.
	#[display("invalid token range: {_0}")]
	#[error(ignore)]
	InvalidTokenRange(String),

	// -- Externals
	/// Wraps I/O errors from `std::io`.
	#[display("io error: {_0}")]
	#[from]
	Io(std::io::Error),

	/// Wraps JSON serialization/deserialization errors.
	#[display("json error: {_0}")]
	#[from]
	SerdeJson(serde_json::Error),

	/// Wraps tokenizer errors from the `tokenizers` crate.
	#[display("tokenizer error: {_0}")]
	#[from]
	TokenizerError(tokenizers::Error),
}

// endregion: --- Error

// region:    --- Custom

impl Error {
	/// Creates a `Custom` error from a string-like value.
	///
	/// # Examples
	///
	/// ```
	/// use data_generator::Error;
	///
	/// let err = Error::custom("something went wrong");
	/// assert!(matches!(err, Error::Custom(_)));
	/// ```
	pub fn custom(val: impl Into<String>) -> Self {
		Self::Custom(val.into())
	}

	/// Creates a `Custom` error from any type that implements `std::error::Error`.
	///
	/// This is useful for converting external errors that are not explicitly
	/// handled by other variants into a `Custom` error.
	pub fn custom_from_err(err: impl std::error::Error) -> Self {
		Self::Custom(err.to_string())
	}
}

// endregion: --- Custom

// region:    --- Tests

#[cfg(test)]
mod tests {
	type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

	use super::*;

	#[test]
	fn test_error_custom() -> Result<()> {
		let err = Error::custom("something went wrong");
		assert!(matches!(err, Error::Custom(_)));
		assert_eq!(err.to_string(), "something went wrong");
		Ok(())
	}

	#[test]
	fn test_error_custom_from_err() -> Result<()> {
		let io_err = std::io::Error::other("io error");
		let err = Error::custom_from_err(io_err);
		assert!(matches!(err, Error::Custom(_)));
		assert_eq!(err.to_string(), "io error");
		Ok(())
	}

	#[test]
	fn test_error_from_string() -> Result<()> {
		let err: Error = "string error".to_string().into();
		assert!(matches!(err, Error::Custom(_)));
		assert_eq!(err.to_string(), "string error");
		Ok(())
	}

	#[test]
	fn test_error_from_str() -> Result<()> {
		let err: Error = "str error".into();
		assert!(matches!(err, Error::Custom(_)));
		assert_eq!(err.to_string(), "str error");
		Ok(())
	}

	#[test]
	fn test_error_invalid_token_range_display() -> Result<()> {
		let err = Error::InvalidTokenRange("bad range".to_string());
		assert_eq!(err.to_string(), "invalid token range: bad range");
		Ok(())
	}

	#[test]
	fn test_error_io_from() -> Result<()> {
		let io_err = std::io::Error::other("disk full");
		let err: Error = io_err.into();
		assert!(matches!(err, Error::Io(_)));
		assert!(err.to_string().contains("disk full"));
		Ok(())
	}
}

// endregion: --- Tests
