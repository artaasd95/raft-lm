# Data

Data storage for raw and processed datasets.

## Structure

- **raw/** - Original, unmodified data files
- **processed/** - Preprocessed, cleaned, ready-to-use data

## Data Types

- Synthetic data (heavy-tailed distributions for initial testing)
- Real financial data (for validation)
- Risk scenario datasets
- Trading/position datasets

## Guidelines

- Never modify raw data
- Document preprocessing steps
- Version processed datasets
- Include data generation scripts in `scripts/`
- Track data provenance in metadata files

