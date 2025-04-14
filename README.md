# Labor Market Agent-Based Model

## CPS Data Pipeline

This project includes tools for processing Current Population Survey (CPS) data from IPUMS to analyze labor market dynamics.

### Prerequisites

1. R with required packages (will be installed automatically when needed): 
   - ipumsr
   - tidyverse
   - lubridate
   - plm
   - rvest
   - janitor
   - dotenv

2. IPUMS API key
   - Register at https://www.ipums.org/
   - Get your API key from your IPUMS account
   - Create a `.env` file in the project root with: `IPUMS_API_KEY=your_key_here`

### Running the Pipeline

#### Option 1: Using RStudio

```r
# Set project root
project_root <- "/Users/brendanchapuis/Projects/research/labor-abm"  # Adjust as needed

# Source the pipeline script
source(file.path(project_root, "code", "build", "data", "get_data.R"))

# Run the pipeline with custom parameters
run_cps_pipeline(
  project_root = project_root,
  output_file = "data/processed/cps_transitions_2019_2020.csv",
  refresh_extract = FALSE,  # Set to TRUE to force new API extract
  force_scrape = FALSE,     # Set to TRUE to force re-scraping sample IDs
  start_date = "2019-01",
  end_date = "2020-12",
  include_asec = FALSE,
  debug = TRUE
)
```

#### Option 2: From Command Line

```bash
Rscript code/build/data/get_data.R start_date=2019-01 end_date=2020-12 debug=TRUE
```

### Pipeline Steps

1. **Identify CPS Samples**: Scrapes or loads sample IDs from IPUMS for the specified time range
2. **Fetch CPS Data**: Uses IPUMS API to download raw data or loads existing files
3. **Process into Panel Format**: Creates consistent panel structure for analysis
4. **Add Lead Variables**: Adds forward-looking variables for employment transitions
5. **Calculate Transitions**: Identifies employment state changes between periods
6. **Save Processed Data**: Exports clean dataset in CSV format

### Data Files

- **Raw data**: Stored in `data/raw/`
- **Metadata**: Stored in `data/metadata/`
- **Processed output**: Stored in `data/processed/`

### Module Structure

- `code/build/data/scrape_cps_samples.R` - Scrapes CPS sample IDs from IPUMS
- `code/build/data/clean_cps.R` - Core data processing functions
- `code/build/data/get_data.R` - Orchestrates the entire workflow

Each module can be run independently or as part of the pipeline.