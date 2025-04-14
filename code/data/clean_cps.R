# Module for CPS data cleaning and processing
# =========================================

# Load required packages directly for renv detection
library(ipumsr)
library(janitor)
library(lubridate)
library(plm)
# library(tidyverse) # Removed - Load specific packages if needed, many are dependencies anyway
library(dotenv)
library(data.table)

# Set options after loading packages
options(scipen = 999, digits = 4)
set.seed(8675309)

#' Initialize IPUMS API key from .env file
#'
#' @param project_root Root directory of the project
#' @return TRUE if successful, stops with error otherwise
init_ipums_api <- function(project_root) {
  env_file <- file.path(project_root, ".env")

  if (file.exists(env_file)) {
    load_dot_env(file = env_file)
    IPUMS_API_KEY <- Sys.getenv("IPUMS_API_KEY")

    if (IPUMS_API_KEY == "" || IPUMS_API_KEY == "your_api_key_here") {
      stop("Please set your IPUMS API key in the .env file at ", env_file)
    }
  } else {
    cat("Creating .env file at ", env_file, "\n")
    writeLines("IPUMS_API_KEY=your_api_key_here", env_file)
    stop("Please edit the .env file at ", env_file, " with your actual IPUMS API key.")
  }

  set_ipums_api_key(IPUMS_API_KEY)
  return(TRUE)
}

#' Get CPS sample IDs for a date range
#'
#' @param project_root Project root directory
#' @param start_date Start date for data in YYYY-MM format
#' @param end_date End date for data in YYYY-MM format
#' @param include_asec Whether to include ASEC samples
#' @param force_scrape Whether to force scraping even if file exists
#' @return Data frame of filtered sample IDs
get_cps_samples <- function(project_root, start_date = "2019-01", end_date = "2020-12",
                            include_asec = FALSE, force_scrape = FALSE) {
  metadata_dir <- file.path(project_root, "data", "metadata")
  samples_file <- file.path(metadata_dir, "cps_sample_ids.csv")
  scraper_path <- file.path(project_root, "code", "data", "scrape_cps_samples.R") # Define scraper path

  # Scrape if needed
  if (!file.exists(samples_file) || force_scrape) {
    message("Scraping sample IDs from IPUMS (force_scrape=", force_scrape, ")...")
    # Ensure the scraper script exists
    if (!file.exists(scraper_path)) {
      stop("Cannot find scrape_cps_samples.R script at: ", scraper_path)
    }
    # Source the script which should define the scrape_cps_samples function
    source(scraper_path)
    # Call the scraping function (assuming it's named scrape_cps_samples)
    # This function should handle the actual scraping and saving/overwriting the file
    # and return the path to the file.
    samples_file_scraped <- scrape_cps_samples(output_dir = metadata_dir, force = force_scrape)
    # Verify the file was created/updated by the scraper
    if (!file.exists(samples_file_scraped)) {
        stop("Scraping function ran but failed to produce the expected file: ", samples_file)
    }
    # Use the path returned by the scraper, which should match samples_file
    samples_file <- samples_file_scraped
    message("Scraping complete. Using file: ", samples_file)
  } else {
      message("Using existing sample ID file: ", samples_file)
  }

  # Parse dates and read samples
  start_date <- as.Date(paste0(start_date, "-01"))
  end_date <- as.Date(paste0(end_date, "-01"))

  message("Loading CPS sample IDs from ", samples_file)
  # Using data.table::fread for potentially faster reading
  samples_dt <- data.table::fread(samples_file)

  samples_dt[, `:=`(
    sample_year = as.numeric(sample_year),
    sample_month = as.numeric(sample_month),
    date = as.Date(paste0(sample_year, "-", sample_month, "-01"))
  )]

  samples_dt <- samples_dt[date >= start_date & date <= end_date]


  # Apply ASEC filter if requested
  if (!include_asec) {
    samples_dt <- samples_dt[asec == FALSE | is.na(asec)] # Keep if asec is FALSE or NA
  }

  # Convert back to data frame if downstream code expects it, or keep as DT
  # Returning data.frame as per original function signature
  return(as.data.frame(samples_dt))
}

#' Get CPS data from IPUMS API
#'
#' @param project_root Root directory of the project
#' @param refresh_extract Whether to force retrieving a new extract
#' @param start_date Start date for data in YYYY-MM format
#' @param end_date End date for data in YYYY-MM format
#' @param subset_start_date Start date for filtering the data subset (YYYY-MM)
#' @param subset_end_date End date for filtering the data subset (YYYY-MM)
#' @param include_asec Whether to include ASEC samples
#' @param force_scrape Whether to force scraping of sample IDs
#' @param debug Print debug information
#' @return CPS data.table
get_cps_data <- function(project_root, refresh_extract = FALSE,
                         start_date = "2019-01", end_date = "2020-12",
                         subset_start_date = NULL, subset_end_date = NULL,
                         include_asec = FALSE, force_scrape = FALSE,
                         debug = FALSE) {
  # Setup directories
  download_dir <- file.path(project_root, "data", "raw")
  dir.create(download_dir, recursive = TRUE, showWarnings = FALSE)

  # Ensure refresh_extract is logical
  refresh_extract <- as.logical(refresh_extract)
  if (debug) message("refresh_extract parameter is: ", refresh_extract)

  # Get sample IDs
  message("Getting sample IDs for date range: ", start_date, " to ", end_date)
  samples_df <- get_cps_samples(
    project_root = project_root,
    start_date = start_date,
    end_date = end_date,
    include_asec = include_asec,
    force_scrape = force_scrape
  )

  samples <- samples_df$sample_id
  if (length(samples) == 0) {
    stop("No CPS samples found in the specified date range")
  }

  message("Using ", length(samples), " CPS samples")

  # Check for existing extract files
  data_files <- list.files(download_dir, pattern = "\\.dat(\\.gz)?$|\\.csv(\\.gz)?$", full.names = TRUE)
  ddi_files <- list.files(download_dir, pattern = "\\.xml$", full.names = TRUE)

  # Force refresh by setting file lists to empty when refresh_extract is TRUE
  if (refresh_extract == TRUE) {
    message("Forcing refresh extract, ignoring existing files.")
    # Optionally remove existing files
    # file.remove(c(data_files, ddi_files))
    data_files <- character(0)
    ddi_files <- character(0)
  }

  cps_data <- NULL # Initialize cps_data

  if (length(data_files) > 0 && length(ddi_files) > 0) {
    # Try loading existing files first
    message("Attempting to load existing extract files using DDI: ", ddi_files[[1]], "...")
    tryCatch({
      # Use read_ipums_micro with the DDI file path.
      # It should automatically find the corresponding data file if named conventionally.
      cps_data <- read_ipums_micro(ddi_files[[1]], verbose = FALSE) # Pass only DDI path

      # Basic validation: Check if the loaded data seems reasonable
      if (is.null(cps_data) || nrow(cps_data) == 0) {
          stop("Loaded data from existing files is empty or null.")
      }
      # Optional: Add a check for date range consistency if needed, comparing
      # min/max dates in cps_data with start_date/end_date.

      if (debug) message("Raw data dimensions (from existing files): ", paste(dim(cps_data), collapse = " x "))
      message("Successfully loaded data from existing files.")

    }, error = function(e) {
      message("Error loading existing files: ", e$message)
      message("Will proceed to download a new extract.")
      cps_data <<- NULL # Ensure cps_data is NULL if loading failed
      # Optionally delete potentially corrupt files
      # file.remove(c(data_files, ddi_files))
      data_files <<- character(0) # Reset file lists
      ddi_files <<- character(0)
    })
  }

  # Download new extract if no valid existing files or if refresh forced
  if (is.null(cps_data)) {
      if (!refresh_extract && length(data_files) > 0) {
          message("Existing files found but failed to load. Proceeding to download new extract.")
      } else if (refresh_extract) {
          message("refresh_extract is TRUE. Proceeding to download new extract.")
      } else {
          message("No suitable existing files found. Proceeding to download new extract.")
      }

    # Define variables for extract - includes CBSASZ
    variables <- c("CPSIDP", "AGE", "SEX", "RACE", "HISPAN",
                  "LABFORCE", "EMPSTAT", "EDUC", "DURUNEMP",
                  "IND1990", "OCC1990", "STATEFIP", "MONTH", "YEAR",
                  "MISH", "WTFINL", "RELATE", "METRO", "CBSASZ", # Includes CBSASZ
                  "MARST", "CITIZEN", "NATIVITY", "VETSTAT", "FAMSIZE",
                  "NCHILD", "DIFFMOB", "DIFFANY", "CLASSWKR", "PROFCERT")

    message("Submitting new extract request...")
    extract <- define_extract_micro(
      collection = "cps",
      description = "CPS extract for unemployment analysis",
      samples = samples,
      variables = variables,
      data_format = "csv", # Requesting CSV
      data_structure = "rectangular",
      rectangular_on = "P"
    )

    submitted_extract <- submit_extract(extract)
    extract_number_str <- sprintf("%05d", submitted_extract$number) # Format number e.g., 00023
    message("Extract number: ", extract_number_str)

    wait_for_extract(submitted_extract)
    message("Downloading extract ", extract_number_str, " to ", download_dir)
    # Call download_extract, but don't rely solely on its return value
    download_result <- tryCatch(
        download_extract(submitted_extract, download_dir = download_dir),
        error = function(e) {
            message("Error during download_extract: ", e$message)
            return(NULL) # Return NULL on error
        }
    )
    # Even if download_result seems incomplete, check the directory
    message("Download attempt finished. Verifying files in directory...")

    # Search the download directory for files matching the extract number
    expected_ddi_pattern <- paste0("cps_", extract_number_str, "\\.xml$")
    expected_data_pattern <- paste0("cps_", extract_number_str, "\\.csv(\\.gz)?$")

    all_files_in_dir <- list.files(download_dir, full.names = TRUE)
    ddi_file_path <- all_files_in_dir[grepl(expected_ddi_pattern, basename(all_files_in_dir))]
    data_file_path <- all_files_in_dir[grepl(expected_data_pattern, basename(all_files_in_dir))]

    # Validate that we found exactly one of each
    if (length(ddi_file_path) != 1 || length(data_file_path) != 1) {
        message("DEBUG: Files found in directory matching patterns:")
        message("  DDI files: ", paste(ddi_file_path, collapse=", "))
        message("  Data files: ", paste(data_file_path, collapse=", "))
        stop("Could not find exactly one DDI and one data file for extract ", extract_number_str, " in ", download_dir)
    }
    # Use the first (and only) match found
    ddi_file_path <- ddi_file_path[[1]]
    data_file_path <- data_file_path[[1]]

    message("Found DDI: ", ddi_file_path)
    message("Found Data: ", data_file_path)

    # Read directly using read_ipums_micro with the path to the found DDI file
    message("Reading downloaded data using DDI: ", ddi_file_path)
    cps_data <- read_ipums_micro(ddi_file_path, verbose = FALSE)

    if (is.null(cps_data) || nrow(cps_data) == 0) {
        stop("Failed to read data from the newly downloaded files using DDI: ", ddi_file_path)
    }
    if (debug) message("Raw data dimensions (from new extract): ", paste(dim(cps_data), collapse = " x "))
  }

  # Initial cleaning and date column creation
  message("Creating date column and basic cleaning...")
  cps_data <- cps_data %>%
    janitor::clean_names() %>% # Clean names early
    # Ensure it's a data.frame before data.table conversion if needed, but read_ipums_micro often returns tibble/df
    as.data.frame()

  # Convert to data.table for efficiency
  setDT(cps_data)
  if (debug) message("Converted raw data to data.table")

  # Create date column using data.table
  cps_data[, date := ymd(paste0(year, "-", month, "-01"))]

  # Early subsetting by date if requested - before any heavy processing
  if (!is.null(subset_start_date) && !is.null(subset_end_date)) {
    message("*** SUBSETTING DATA EARLY - Before processing ***")
    subset_start_date_parsed <- as.Date(paste0(subset_start_date, "-01"))
    subset_end_date_parsed <- as.Date(paste0(subset_end_date, "-01"))

    message("Filtering data to subset date range: ", subset_start_date, " to ", subset_end_date)
    rows_before <- nrow(cps_data)

    # Use data.table filtering
    cps_data <- cps_data[date >= subset_start_date_parsed & date <= subset_end_date_parsed]

    rows_after <- nrow(cps_data)
    message("Rows removed by date filtering: ", rows_before - rows_after,
            " (", round((rows_before - rows_after) / rows_before * 100, 1), "% reduction)")

    if (nrow(cps_data) == 0) {
      stop("No data remaining after filtering for subset dates")
    }

    if (debug) {
      message("Data dimensions after date subsetting: ", paste(dim(cps_data), collapse = " x "))
      message("Unique individuals after date subsetting: ", uniqueN(cps_data$cpsidp))
    }
  }

  # Continue with remaining processing - more efficient now with smaller dataset
  message("Processing employment status and filtering for longitudinal analysis using data.table...")
  rows_before <- nrow(cps_data)

  # Ensure sorting for lead calculation
  setkey(cps_data, cpsidp, date)

  # Calculate month difference using data.table's shift more robustly
  # We need the *next* month's data for the same person
  cps_data[, lead_date := shift(date, type = "lead"), by = cpsidp]
  # Calculate difference in months
  cps_data[, month_diff := as.numeric(lead_date - date) / 30.44] # Approximate days per month

  # Apply filters using data.table syntax
  # Filter for working age (>=16), those *potentially* in labor force (labforce != 0 removes NIU),
  # and those observed in the *immediately following* month (month_diff approx 1).
  # Allow a small tolerance for month_diff calculation (e.g., 0.9 to 1.1)
  cps_data_filtered <- cps_data[age >= 16 & labforce != 0 & month_diff > 0.9 & month_diff < 1.1]

  # Remove temporary columns
  cps_data_filtered[, `:=`(lead_date = NULL, month_diff = NULL)]

  rows_after <- nrow(cps_data_filtered)
  message("Rows removed by age/labor force/follow-up filtering: ", rows_before - rows_after,
          " (", round((rows_before - rows_after) / rows_before * 100, 1), "% reduction)")

  # Remove unnecessary columns - ensure wtfinl is NOT removed
  # Add any other IPUMS system variables not needed (check DDI if unsure)
  cols_to_remove <- c("serial", "asecflag", "asecwth", "pernum", "cpsid", "ident", "hwtfinl", "asecwt", # Common system/weight vars
                      "lfproxy", "d_year", "cpsidv", "asecwt", "earnwke", "hrswork", "paidhour") # Examples of other potentially unneeded vars
  # Find which of these columns actually exist in the data to avoid errors
  existing_cols_to_remove <- intersect(names(cps_data_filtered), cols_to_remove)

  if (length(existing_cols_to_remove) > 0) {
    message("Removing unnecessary IPUMS system/intermediate columns: ", paste(existing_cols_to_remove, collapse=", "))
    # Use data.table's way to remove columns by reference
    cps_data_filtered[, (existing_cols_to_remove) := NULL]
  }

  if (debug) {
    message("Final processed data dimensions before deriving variables: ", paste(dim(cps_data_filtered), collapse = " x "))
    message("Unique individuals in final data: ", uniqueN(cps_data_filtered$cpsidp))
  }

  # Return as data.table
  return(cps_data_filtered)
}


#' Process CPS data into panel format with derived variables
#'
#' @param cps CPS data.table (expects output from get_cps_data)
#' @param national_rates_output_path Path to save the national rates CSV.
#' @return Processed CPS data as data.table
process_cps_data <- function(cps, national_rates_output_path = NULL) {
  # Ensure input is a data.table
  if (!is.data.table(cps)) setDT(cps)

  message("Deriving core employment status variables...")
  # Create core employment status variables first using fifelse
  # EMPSTAT Codes: 10, 12 = Employed; 20, 21, 22 = Unemployed
  # LABFORCE Codes: 1 = Not in Labor Force, 2 = In Labor Force (implies emp or unemp)
  cps[, `:=`(
    emp = fifelse(empstat %in% c(10L, 12L), 1L, 0L),
    unemp = fifelse(empstat %in% c(20L, 21L, 22L), 1L, 0L),
    nilf = fifelse(labforce == 1L, 1L, 0L)
  )]

  message("Deriving other categorical and numerical variables...")
  # Create other derived variables using data.table syntax and fcase
  cps[, `:=`(
    # Cyclical encoding for month
    mth_dim1 = round(sin(month / 12 * 2 * pi), 4) + 1,
    mth_dim2 = round(cos(month / 12 * 2 * pi), 4) + 1,
    # HISPAN: 0=No; 100-400=Yes (specific origin); >=900=NIU/Unknown
    hispan_cat = fcase(
      hispan == 0L, "Not hispanic",
      hispan > 0L & hispan < 900L, "Hispanic",
      default = NA_character_ # Handle 900+ codes as NA
    ),
    # EDUC: Grouping educational attainment
    educ_cat = fcase(
      educ > 1L & educ <= 73L, "High school or less", # Up to HS diploma (code 73)
      educ > 73L & educ <= 110L, "Some college, no bachelor's degree", # Includes associate degrees (codes 81, 91, 92) but not bachelor's (111)
      educ == 111L, "Bachelor's degree",
      educ %in% c(123L, 124L, 125L), "Advanced degree", # Master's (123), Prof (124), PhD (125)
      default = NA_character_ # Codes 0, 1, 112+ (except 123-125) become NA
    ),
    # DURUNEMP: Duration of unemployment. 999=NIU. Set to 0 if not unemployed.
    # Ensure integer type consistency
    durunemp = fifelse(unemp == 0L, 0L, fifelse(durunemp == 999L, NA_integer_, as.integer(durunemp))),
    # IND1990: Convert to numeric for grouping
    ind1990 = as.numeric(ind1990), # Keep as numeric for range checks
    # RELATE: Relationship to head. Simplify. 101=Head, 201=Spouse, 301=Child.
    relate_cat = fcase(
        relate == 101L, "Head/Householder",
        relate == 201L, "Spouse",
        relate == 301L, "Child",
        # Group other relatives and non-relatives
        relate > 301L & relate < 1200L, "Other Relative/Non-relative", # Broad grouping
        default = "Unknown/NIU" # Handle 0 or other codes
    ),
    # METRO: Metropolitan status. 0=NIU, 1=Not ID, 2=Central City, 3=Suburb, 4=Non-metro
    metro_cat = fcase(
        metro == 0L, "NIU",
        metro == 1L, "Not identifiable",
        metro == 2L, "Central city",
        metro == 3L, "Outside central city",
        metro == 4L, "Not in metro area",
        default = "Unknown" # Should generally not happen if codes are 0-4
    ),
    # CBSASZ: CBSA size. 0=NIU, 1-7=Size categories.
    cbsasz_cat = fcase(
        cbsasz == 0L, "NIU/Non-metro", # Often 0 means non-metro or not ID'd
        cbsasz == 1L, "<100k",
        cbsasz == 2L, "100k-250k",
        cbsasz == 3L, "250k-500k",
        cbsasz == 4L, "500k-1M",
        cbsasz == 5L, "1M-2.5M",
        cbsasz == 6L, "2.5M-5M",
        cbsasz == 7L, "5M+",
        default = "Unknown"
    ),
    # MARST: Marital status. 1-6 = Categories. 9=NIU/Unknown.
    marst_cat = fcase(
        marst == 1L, "Married, spouse present",
        marst == 2L, "Married, spouse absent",
        marst == 3L, "Separated",
        marst == 4L, "Divorced",
        marst == 5L, "Widowed",
        marst == 6L, "Never married/single",
        default = NA_character_ # Handle 9 (NIU) as NA
    ),
    # CITIZEN: Citizenship status. 1-5 = Categories. 9=NIU/Unknown.
    citizen_cat = fcase(
        citizen == 1L, "Born in US",
        citizen == 2L, "Born in PR/Territory",
        citizen == 3L, "Born abroad, US parents",
        citizen == 4L, "Naturalized citizen",
        citizen == 5L, "Not a citizen",
        default = NA_character_ # Handle 9 (NIU) as NA
    ),
    # NATIVITY: Nativity. 1=Native, 5=Foreign born. 9=NIU/Unknown.
    nativity_cat = fcase(
        nativity == 1L, "Native born",
        nativity > 1L & nativity < 5L, "Native born, foreign parents",
        nativity == 5L, "Foreign born",
        default = NA_character_ # Handle 9 (NIU) as NA
    ),
    # VETSTAT: Veteran status. 0=NIU, 1=No service, 2=Yes service. 9=Unknown.
    vetstat_cat = fcase(
        vetstat == 0L, "NIU",
        vetstat == 1L, "No service",
        vetstat == 2L, "Yes service",
        default = "Unknown" # Handle 9 or other codes
    ),
    # FAMSIZE: Family size. Keep numeric. Handle potential high codes (e.g., 99) if they mean NIU.
    famsize = fifelse(famsize >= 99L, NA_integer_, as.integer(famsize)),
    # NCHILD: Number of own children. Keep numeric. Handle potential high codes.
    nchild = fifelse(nchild >= 99L, NA_integer_, as.integer(nchild)),
    # DIFFANY: Any difficulty. 0=NIU, 1=No, 2=Yes.
    diff_cat = fcase(
        diffany == 0L, "NIU",
        diffany == 1L, "No difficulty",
        diffany == 2L, "Has difficulty",
        default = "Unknown"
    ),
    # PROFCERT: Professional certification. 0=NIU, 1=No, 2=Yes.
    profcert_cat = fcase(
        profcert == 0L, "NIU",
        profcert == 1L, "No certification",
        profcert == 2L, "Has certification",
        default = "Unknown"
    ),
    # CLASSWKR: Class of worker. Simplify. 10/13/14=Self/Unpaid; 21/22=Private; 25-28=Govt.
    classwkr_cat = fcase(
        classwkr %in% c(10L, 13L, 14L), "Self-employed/Unpaid", # Inc/Not Inc/Unpaid Family
        classwkr %in% c(21L, 22L), "Private", # Profit/Non-Profit
        classwkr %in% c(25L, 27L, 28L), "Government", # Fed/State/Local
        classwkr == 29, "Government", # Include Govt unspecified if present
        default = "Other/NIU" # Handle 0 (NIU) or unemployed (not applicable)
    ),
    # OCC1990: Convert to numeric for grouping
    occ1990 = as.numeric(occ1990),
    # WTFINL: Ensure weight is numeric
    wtfinl = as.numeric(wtfinl)
  )]

  # Create race_cat from race (treating numeric codes as factors)
  cps[, race_cat := as.factor(race)]

  # Derive ind_group_cat after ind1990 is numeric
  # Based on IND1990 codes
  cps[, ind_group_cat := fcase(
    ind1990 >= 10 & ind1990 <= 32, "Agr, Forest, Fish",
    ind1990 >= 40 & ind1990 <= 50, "Mining",
    ind1990 == 60, "Construction",
    ind1990 >= 100 & ind1990 <= 392, "Manufacturing",
    ind1990 >= 400 & ind1990 <= 472, "Transport, Comm, Util",
    ind1990 >= 500 & ind1990 <= 571, "Wholesale Trade",
    ind1990 >= 580 & ind1990 <= 691, "Retail Trade",
    ind1990 >= 700 & ind1990 <= 712, "Finance, Ins, Real Estate",
    ind1990 >= 721 & ind1990 <= 760, "Business & Repair Svcs",
    ind1990 >= 761 & ind1990 <= 791, "Personal Svcs",
    ind1990 >= 800 & ind1990 <= 810, "Entertainment & Rec Svcs",
    ind1990 >= 812 & ind1990 <= 893, "Professional & Related Svcs",
    ind1990 >= 900 & ind1990 <= 932, "Public Administration",
    ind1990 >= 940 & ind1990 <= 960, "Active Duty Military", # Sometimes excluded
    ind1990 == 991 | ind1990 == 992, "Unemployed/NIU", # Handle unemployed code if present
    default = NA_character_ # Other codes (e.g., 0) become NA
  )]
  # Remove original ind_group if it existed from a previous run or intermediate step
  if ("ind_group" %in% names(cps)) cps[, ind_group := NULL]


  # Derive occ_group_cat and ensure wtfinl is numeric
  # Based on OCC1990 codes
  cps[, occ_group_cat := fcase(
      occ1990 >= 3 & occ1990 <= 194, "Managerial/Professional", # Range adjusted based on OCC1990 detail
      occ1990 >= 203 & occ1990 <= 389, "Technical/Sales/Admin Support",
      occ1990 >= 403 & occ1990 <= 469, "Service Occupations",
      occ1990 >= 473 & occ1990 <= 498, "Farming/Forestry/Fishing",
      occ1990 >= 503 & occ1990 <= 699, "Precision Prod/Craft/Repair",
      occ1990 >= 703 & occ1990 <= 889, "Operators/Fabricators/Laborers",
      occ1990 >= 900, "Military Occupations", # If applicable
      default = NA_character_
  )]
  # Remove original occ_group if it existed
  if ("occ_group" %in% names(cps)) cps[, occ_group := NULL]


  # --- Aggregate Rate Calculations ---
  message("Calculating weighted labor force aggregates using data.table...")

  # Ensure weight is not NA for calculations, replace with 0 if necessary (or handle appropriately)
  cps[is.na(wtfinl), wtfinl := 0]

  # Define labor force: emp + unemp
  cps[, lf := emp + unemp]

  # Calculate state and industry rates efficiently
  # Filter out NA ind_group_cat before grouping
  rates_ind <- cps[!is.na(ind_group_cat) & lf == 1L, .(
    state_ind_emp_w = sum(wtfinl * emp, na.rm = TRUE),
    state_ind_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
    state_ind_lf_w = sum(wtfinl, na.rm = TRUE) # Sum weights of those in LF
  ), by = .(date, statefip, ind_group_cat)] # Use ind_group_cat

  # Aggregate to state level (using only people in the labor force)
  state_rates <- cps[lf == 1L, .(
    state_emp_w = sum(wtfinl * emp, na.rm = TRUE),
    state_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
    state_lf_w = sum(wtfinl, na.rm = TRUE)
  ), by = .(date, statefip)]

  state_rates[, `:=`(
    state_unemp_rate = fifelse(state_lf_w > 0, state_unemp_w / state_lf_w, NA_real_),
    state_emp_rate = fifelse(state_lf_w > 0, state_emp_w / state_lf_w, NA_real_) # Emp rate = Emp / LF
  )]
  # Select columns to merge
  state_rates <- state_rates[, .(date, statefip, state_unemp_rate, state_emp_rate)]

  # Aggregate to industry level (using only people in the labor force and valid industry)
  industry_rates <- rates_ind[, .(
    ind_group_emp_w = sum(state_ind_emp_w, na.rm = TRUE),
    ind_group_unemp_w = sum(state_ind_unemp_w, na.rm = TRUE),
    ind_group_lf_w = sum(state_ind_lf_w, na.rm = TRUE)
  ), by = .(date, ind_group_cat)] # Use ind_group_cat

  industry_rates[, `:=`(
    ind_group_unemp_rate = fifelse(ind_group_lf_w > 0, ind_group_unemp_w / ind_group_lf_w, NA_real_),
    ind_group_emp_rate = fifelse(ind_group_lf_w > 0, ind_group_emp_w / ind_group_lf_w, NA_real_)
  )]
  # Select columns to merge
  industry_rates <- industry_rates[, .(date, ind_group_cat, ind_group_unemp_rate, ind_group_emp_rate)] # Use ind_group_cat

  # Aggregate to national level (using only people in the labor force)
  national_rates <- cps[lf == 1L, .(
      national_emp_w = sum(wtfinl * emp, na.rm = TRUE),
      national_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
      national_lf_w = sum(wtfinl, na.rm = TRUE)
  ), by = date]

  national_rates[, `:=`(
      national_unemp_rate = fifelse(national_lf_w > 0, national_unemp_w / national_lf_w, NA_real_),
      national_emp_rate = fifelse(national_lf_w > 0, national_emp_w / national_lf_w, NA_real_)
  )]
  # Select columns and ensure unique by date
  national_rates <- unique(national_rates[, .(date, national_unemp_rate, national_emp_rate)])
  setorder(national_rates, date) # Ensure chronological order

  # --- Save National Rates ---
  if (!is.null(national_rates_output_path)) {
    message("Saving national aggregate rates to: ", national_rates_output_path)
    tryCatch({
      dir.create(dirname(national_rates_output_path), recursive = TRUE, showWarnings = FALSE)
      fwrite(national_rates, file = national_rates_output_path)
    }, error = function(e) {
      warning("Failed to save national rates file: ", e$message)
    })
  } else {
    message("National rates output path not provided, skipping save.")
  }

  # --- Merge Rates Back (Optional for individual data, keep for consistency) ---
  message("Merging aggregate rates onto individual data using data.table joins...")
  # Use on= to specify join columns, use i. prefix for incoming columns
  cps[national_rates, on = "date", `:=`(national_unemp_rate = i.national_unemp_rate, national_emp_rate = i.national_emp_rate)]
  cps[state_rates, on = c("date", "statefip"), `:=`(state_unemp_rate = i.state_unemp_rate, state_emp_rate = i.state_emp_rate)]
  cps[industry_rates, on = c("date", "ind_group_cat"), `:=`(ind_group_unemp_rate = i.ind_group_unemp_rate, ind_group_emp_rate = i.ind_group_emp_rate)] # Use ind_group_cat

  # Remove the intermediate labor force flag
  cps[, lf := NULL]

  # --- Validation and Cleanup ---
  message("Performing final validation and cleanup...")
  # Create validation column using data.table
  # This checks if exactly one status (emp, unemp, nilf) is 1
  cps[, status_check := emp + unemp + nilf]
  cps[, status_error := fifelse(status_check != 1L, 1L, 0L)]

  # Report any errors in classification
  error_count <- sum(cps$status_error, na.rm = TRUE)
  if(error_count > 0) {
    warning(paste0("Found ", error_count, " records with inconsistent employment classifications (emp+unemp+nilf != 1). Check raw EMPSTAT/LABFORCE values for these cases."))
    # Optional: View inconsistent records
    # print(cps[status_error == 1L, .(cpsidp, date, empstat, labforce, emp, unemp, nilf)])
  }

  # Remove validation columns and original source variables that have been categorized/processed
  cols_to_remove <- c("status_check", "status_error", "year", "month", "empstat", "labforce",
                      "race", "occ1990", "ind1990", "hispan", "educ", "relate", "metro", "cbsasz", "marst", # Add race
                      "citizen", "nativity", "vetstat", "diffmob", "diffany", "profcert", "classwkr")
  existing_cols_to_remove <- intersect(cols_to_remove, names(cps))
  if (length(existing_cols_to_remove) > 0) {
      message("Removing intermediate/redundant columns: ", paste(existing_cols_to_remove, collapse=", "))
      cps[, (existing_cols_to_remove) := NULL]
  }

  # Check and remove duplicates using data.table::unique (should be minimal after get_cps_data filtering)
  message("De-duplicating data (if any duplicates remain)...")
  n_before <- nrow(cps)
  # Ensure keys are set for unique, or specify by argument
  setkey(cps, cpsidp, date) # Ensure keys are set
  cps <- unique(cps, by = key(cps))
  n_after <- nrow(cps)
  if (n_before > n_after) {
      message("Removed ", n_before - n_after, " duplicate observations (based on cpsidp, date)")
  }

  # Return data.table
  message("Finished processing variables. Returning data.table.")
  return(cps)
}


#' Add lead variables and filter to matched individuals using data.table
#'
#' @param cps Processed CPS data.table (output from process_cps_data)
#' @return Filtered CPS data.table with lead variables and employment states
add_lead_variables <- function(cps) {
  # Ensure input is a data.table and sorted
  if (!is.data.table(cps)) setDT(cps)
  setkey(cps, cpsidp, date)

  message("Adding lead variables using data.table...")

  # Define columns for which to create leads (key demographics + status indicators)
  # Use the newly created categorical variables where appropriate (e.g., hispan_cat, race_cat)
  lead_cols_demog <- c("age", "sex", "race_cat", "hispan_cat") # Use race_cat
  lead_cols_status <- c("emp", "unemp", "nilf")
  lead_cols <- c(lead_cols_demog, lead_cols_status)

  # Generate names for the new lead columns
  lead_cols_new_names <- paste0(lead_cols, "_f1")

  # Create lead variables using data.table::shift within each individual's timeline
  cps[, (lead_cols_new_names) := shift(.SD, n = 1L, type = "lead"), .SDcols = lead_cols, by = cpsidp]

  message("Creating current and future employment state categories...")
  # Create categorical employment state variables using fcase
  cps[, `:=`(
    emp_state = fcase(
      emp == 1L, "Employed",
      unemp == 1L, "Unemployed",
      nilf == 1L, "Not in Labor Force",
      default = NA_character_ # Should not happen if status_error check passed
    ),
    emp_state_f1 = fcase(
      emp_f1 == 1L, "Employed",
      unemp_f1 == 1L, "Unemployed",
      nilf_f1 == 1L, "Not in Labor Force",
      default = NA_character_ # Will be NA if person not observed next month
    )
  )]

  message("Filtering to matched individuals with valid future state and consistent demographics...")
  rows_before <- nrow(cps)

  # Filter using data.table syntax:
  # 1. Keep only rows where the next month's state is known (!is.na(emp_state_f1))
  # 2. Keep only rows where key demographics are consistent between months
  #    (allowing for age to increase by 1 is complex, easier to check equality for short panels)
  #    Need to handle NA comparisons correctly
  cps_final <- cps[
    !is.na(emp_state_f1) & # Future state must be known
    age == age_f1 &        # Age should be identical (short panel assumption)
    sex == sex_f1 &        # Sex should be identical
    # Compare categorical race status, handle NAs
    ((race_cat == race_cat_f1) | (is.na(race_cat) & is.na(race_cat_f1))) &
    # Compare categorical hispanic status, handle NAs
    ((hispan_cat == hispan_cat_f1) | (is.na(hispan_cat) & is.na(hispan_cat_f1)))
  ]

  rows_after <- nrow(cps_final)
  reduction_pct <- if (rows_before > 0) round((rows_before - rows_after) / rows_before * 100, 1) else 0
  message("Rows removed by matching/filtering: ", rows_before - rows_after,
          " (", reduction_pct, "% reduction)")

  # Remove temporary lead columns and intermediate binary status indicators
  cols_to_remove <- c(lead_cols_new_names, "emp", "unemp", "nilf", "mish") # Remove originals too
  existing_cols_to_remove <- intersect(names(cps_final), cols_to_remove)
  if (length(existing_cols_to_remove) > 0) {
    message("Removing temporary lead/intermediate columns: ", paste(existing_cols_to_remove, collapse=", "))
    cps_final[, (existing_cols_to_remove) := NULL]
  }

  message("Finished creating lead variables and filtering.")
  return(cps_final)
}


#' Label CPS variables for clarity
#'
#' @param data CPS data.table (output from add_lead_variables)
#' @return Data.table with labeled variables using attributes
label_variables <- function(data) {
    if (!is.data.table(data)) setDT(data) # Ensure it's a data.table

    message("Adding variable labels...")

    # Define labels in a list
    var_labels <- list(
        cpsidp = "IPUMS CPS Person Identifier (Longitudinal)",
        date = "Reference Date (YYYY-MM-01)",
        wtfinl = "Final Person Weight",
        age = "Age",
        sex = "Sex (1=Male, 2=Female)",
        race_cat = "Race (Categorical, based on IPUMS codes)", # Add race_cat
        statefip = "State FIPS Code",
        mth_dim1 = "Month Sine Component (Cyclical Encoding)",
        mth_dim2 = "Month Cosine Component (Cyclical Encoding)",
        hispan_cat = "Hispanic Origin (Categorical)",
        educ_cat = "Educational Attainment (Categorical)",
        durunemp = "Duration of Unemployment in Weeks (0 if Employed/NILF)",
        ind_group_cat = "Industry Group (Categorical, based on IND1990)", # Update name
        occ_group_cat = "Occupation Group (Categorical, based on OCC1990)", # Update name
        relate_cat = "Relationship to Household Head (Categorical)",
        metro_cat = "Metropolitan Status (Categorical)",
        cbsasz_cat = "CBSA Size Category (Categorical)",
        marst_cat = "Marital Status (Categorical)",
        citizen_cat = "Citizenship Status (Categorical)",
        nativity_cat = "Nativity (Native/Foreign Born)",
        vetstat_cat = "Veteran Status (Categorical)",
        famsize = "Family Size (Numeric)",
        nchild = "Number of Own Children in Household (Numeric)",
        diff_cat = "Any Difficulty Reported (Yes/No/NIU)",
        profcert_cat = "Professional Certification Reported (Yes/No/NIU)",
        classwkr_cat = "Class of Worker (Categorical)",
        national_unemp_rate = "Monthly National Unemployment Rate (Weighted)",
        national_emp_rate = "Monthly National Employment Rate (Weighted, Emp/LF)",
        state_unemp_rate = "Monthly State Unemployment Rate (Weighted)",
        state_emp_rate = "Monthly State Employment Rate (Weighted, Emp/LF)",
        ind_group_unemp_rate = "Monthly Industry Group Unemployment Rate (Weighted)",
        ind_group_emp_rate = "Monthly Industry Group Employment Rate (Weighted, Emp/LF)",
        emp_state = "Current Month Employment Status (Categorical)",
        emp_state_f1 = "Next Month Employment Status (Categorical)"
        # Remove label for 'race' if it existed
    )

    # Apply labels using setattr
    for (var_name in names(var_labels)) {
        if (var_name %in% names(data)) {
            setattr(data[[var_name]], "label", var_labels[[var_name]])
        }
    }

    # Could also add labels for original coded variables if they were kept

    message("Finished labeling variables.")
    return(data)
}
