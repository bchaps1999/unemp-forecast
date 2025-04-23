# Module for CPS data cleaning and processing
# =========================================

# Load required packages directly for renv detection
library(ipumsr)
library(janitor)
library(lubridate)
library(plm) # Note: plm is loaded but not used in *this* specific script. Keep if used elsewhere.
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

  # Filter by date range
  samples_dt <- samples_dt[date >= start_date & date <= end_date]

  # Apply ASEC filter if requested
  if (!include_asec) {
    samples_dt <- samples_dt[asec == FALSE | is.na(asec)] # Keep if asec is FALSE or NA
  }

  # Convert back to data frame if downstream code expects it, or keep as DT
  # Returning data.frame as per original function signature
  # Note: Could return data.table if calling function adapted
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
      cps_data <- read_ipums_micro(ddi_files[[1]], verbose = FALSE) # Pass only DDI path

      # Basic validation
      if (is.null(cps_data) || nrow(cps_data) == 0) {
          stop("Loaded data from existing files is empty or null.")
      }

      if (debug) message("Raw data dimensions (from existing files): ", paste(dim(cps_data), collapse = " x "))
      message("Successfully loaded data from existing files.")

    }, error = function(e) {
      message("Error loading existing files: ", e$message)
      message("Will proceed to download a new extract.")
      cps_data <<- NULL # Ensure cps_data is NULL if loading failed
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

    # Define variables for extract
    variables <- c("CPSIDP", "AGE", "SEX", "RACE", "HISPAN",
                  "LABFORCE", "EMPSTAT", "EDUC", "DURUNEMP",
                  "IND1990", "OCC1990", "STATEFIP", "MONTH", "YEAR",
                  "MISH", "WTFINL", "RELATE", "METRO", "CBSASZ",
                  "MARST", "CITIZEN", "FAMSIZE", "NCHILD", "DIFFANY", "CLASSWKR")

    message("Submitting new extract request...")
    extract <- define_extract_micro(
      collection = "cps",
      description = "CPS extract for analysis",
      samples = samples,
      variables = variables,
      data_format = "csv", # Requesting CSV
      data_structure = "rectangular",
      rectangular_on = "P"
    )

    submitted_extract <- submit_extract(extract)
    extract_number_str <- sprintf("%05d", submitted_extract$number)
    message("Extract number: ", extract_number_str)

    wait_for_extract(submitted_extract)
    message("Downloading extract ", extract_number_str, " to ", download_dir)
    # Call download_extract, check directory afterwards
    download_result <- tryCatch(
        download_extract(submitted_extract, download_dir = download_dir),
        error = function(e) {
            message("Error during download_extract: ", e$message)
            return(NULL) # Return NULL on error
        }
    )
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

  # Initial cleaning and convert to data.table
  message("Cleaning names and converting to data.table...")
  cps_data <- janitor::clean_names(cps_data)
  setDT(cps_data)
  if (debug) message("Converted raw data to data.table")

  # Create date column
  cps_data[, date := ymd(paste0(year, "-", month, "-01"))]

  # Early subsetting by date if requested
  if (!is.null(subset_start_date) && !is.null(subset_end_date)) {
    message("*** SUBSETTING DATA EARLY - By Date ***")
    subset_start_date_parsed <- as.Date(paste0(subset_start_date, "-01"))
    subset_end_date_parsed <- as.Date(paste0(subset_end_date, "-01"))

    message("Filtering data to subset date range: ", subset_start_date, " to ", subset_end_date)
    rows_before_date_subset <- nrow(cps_data)
    cps_data <- cps_data[date >= subset_start_date_parsed & date <= subset_end_date_parsed]
    rows_after_date_subset <- nrow(cps_data)
    reduction_pct <- if(rows_before_date_subset > 0) round((rows_before_date_subset - rows_after_date_subset) / rows_before_date_subset * 100, 1) else 0
    message("Rows removed by date filtering: ", rows_before_date_subset - rows_after_date_subset,
            " (", reduction_pct, "% reduction)")
    if (nrow(cps_data) == 0) stop("No data remaining after filtering for subset dates")
    if (debug) message("Data dimensions after date subsetting: ", paste(dim(cps_data), collapse = " x "))
  }

  # Filter for relevant population first (Age, Potential Labor Force)
  message("Applying initial population filters (Age >= 16, LABFORCE != 0)...")
  rows_before_pop_filter <- nrow(cps_data)
  cps_data <- cps_data[age >= 16 & labforce != 0]
  rows_after_pop_filter <- nrow(cps_data)
  reduction_pct_pop <- if(rows_before_pop_filter > 0) round((rows_before_pop_filter - rows_after_pop_filter) / rows_before_pop_filter * 100, 1) else 0
  message("Rows removed by initial population filter: ", rows_before_pop_filter - rows_after_pop_filter,
          " (", reduction_pct_pop, "% reduction)")
  if (nrow(cps_data) == 0) stop("No data remaining after initial population filtering")

  # Calculate leads and filter for month-to-month matches
  message("Calculating leads and filtering for month-to-month observations...")
  rows_before_month_filter <- nrow(cps_data)
  setkey(cps_data, cpsidp, date)
  cps_data[, lead_date := shift(date, type = "lead"), by = cpsidp]
  cps_data[, month_diff := as.numeric(lead_date - date) / 30.44]
  cps_data_filtered <- cps_data[month_diff > 0.9 & month_diff < 1.1]
  cps_data_filtered[, `:=`(lead_date = NULL, month_diff = NULL)] # Remove temporary columns
  rows_after_month_filter <- nrow(cps_data_filtered)
  reduction_pct_month <- if(rows_before_month_filter > 0) round((rows_before_month_filter - rows_after_month_filter) / rows_before_month_filter * 100, 1) else 0
  message("Rows removed by requiring consecutive month observation: ", rows_before_month_filter - rows_after_month_filter,
          " (", reduction_pct_month, "% reduction)")

  # Remove unnecessary IPUMS columns
  cols_to_remove <- c("serial", "asecflag", "asecwth", "pernum", "cpsid", "ident", "hwtfinl", "asecwt",
                      "lfproxy", "d_year", "cpsidv", "asecwt", "earnwke", "hrswork", "paidhour")
  existing_cols_to_remove <- intersect(names(cps_data_filtered), cols_to_remove)
  if (length(existing_cols_to_remove) > 0) {
    message("Removing unnecessary IPUMS system/intermediate columns: ", paste(existing_cols_to_remove, collapse=", "))
    cps_data_filtered[, (existing_cols_to_remove) := NULL]
  }

  if (debug) {
    message("Final processed data dimensions before deriving variables: ", paste(dim(cps_data_filtered), collapse = " x "))
    message("Unique individuals in final data: ", uniqueN(cps_data_filtered$cpsidp))
  }
  gc()
  return(cps_data_filtered)
}


#' Process CPS data into panel format with derived variables
#'
#' @param cps CPS data.table (expects output from get_cps_data)
#' @param national_rates_output_path Path to save the national rates CSV.
#' @return Processed CPS data as data.table
process_cps_data <- function(cps, national_rates_output_path = NULL) {
  if (!is.data.table(cps)) setDT(cps)

  message("Deriving core employment status variables...")
  cps[, `:=`(
    emp = fifelse(empstat %in% c(10L, 12L), 1L, 0L),
    unemp = fifelse(empstat %in% c(20L, 21L, 22L), 1L, 0L),
    nilf = fifelse(labforce == 1L, 1L, 0L)
  )]

  message("Deriving other categorical and numerical variables...")
  cps[, `:=`(
    mth_dim1 = round(sin(month / 12 * 2 * pi), 4) + 1,
    mth_dim2 = round(cos(month / 12 * 2 * pi), 4) + 1,
    hispan_cat = fcase(
      hispan == 0L, "Not hispanic",
      hispan > 0L & hispan < 900L, "Hispanic",
      default = NA_character_
    ),
    educ_cat = fcase(
      educ > 1L & educ <= 73L, "High school or less",
      educ > 73L & educ <= 110L, "Some college, no bachelor's degree",
      educ == 111L, "Bachelor's degree",
      educ %in% c(123L, 124L, 125L), "Advanced degree",
      default = NA_character_
    ),
    durunemp = fifelse(unemp == 0L, 0L, fifelse(durunemp == 999L, NA_integer_, as.integer(durunemp))),
    ind1990 = as.numeric(ind1990),
    relate_cat = fcase(
        relate == 101L, "Head/Householder",
        relate == 201L, "Spouse",
        relate == 301L, "Child",
        relate > 301L & relate < 1200L, "Other Relative/Non-relative",
        default = "Unknown/NIU"
    ),
    metro_cat = fcase(
        metro == 0L, "NIU",
        metro == 1L, "Not identifiable",
        metro == 2L, "Central city",
        metro == 3L, "Outside central city",
        metro == 4L, "Not in metro area",
        default = "Unknown"
    ),
    cbsasz_cat = fcase(
        cbsasz == 0L, "NIU/Non-metro",
        cbsasz == 1L, "<100k",
        cbsasz == 2L, "100k-250k",
        cbsasz == 3L, "250k-500k",
        cbsasz == 4L, "500k-1M",
        cbsasz == 5L, "1M-2.5M",
        cbsasz == 6L, "2.5M-5M",
        cbsasz == 7L, "5M+",
        default = "Unknown"
    ),
    marst_cat = fcase(
        marst == 1L, "Married, spouse present",
        marst == 2L, "Married, spouse absent",
        marst == 3L, "Separated",
        marst == 4L, "Divorced",
        marst == 5L, "Widowed",
        marst == 6L, "Never married/single",
        default = NA_character_
    ),
    citizen_cat = fcase(
        citizen == 1L, "Born in US",
        citizen == 2L, "Born in PR/Territory",
        citizen == 3L, "Born abroad, US parents",
        citizen == 4L, "Naturalized citizen",
        citizen == 5L, "Not a citizen",
        default = NA_character_
    ),
    famsize = fifelse(famsize >= 99L, NA_integer_, as.integer(famsize)),
    nchild = fifelse(nchild >= 99L, NA_integer_, as.integer(nchild)),
    diff_cat = fcase(
        diffany == 0L, "NIU",
        diffany == 1L, "No difficulty",
        diffany == 2L, "Has difficulty",
        default = "Unknown"
    ),
    classwkr_cat = fcase(
        classwkr %in% c(10L, 13L, 14L), "Self-employed/Unpaid",
        classwkr %in% c(21L, 22L), "Private",
        classwkr %in% c(25L, 27L, 28L), "Government",
        classwkr == 29L, "Government",
        default = "Other/NIU"
    ),
    occ1990 = as.numeric(occ1990),
    wtfinl = as.numeric(wtfinl)
  )]

  cps[, race_cat := as.factor(race)]

  cps[, ind_group_cat := fcase(
    ind1990 >= 10L & ind1990 <= 32L, "Agr, Forest, Fish",
    ind1990 >= 40L & ind1990 <= 50L, "Mining",
    ind1990 == 60L, "Construction",
    ind1990 >= 100L & ind1990 <= 392L, "Manufacturing",
    ind1990 >= 400L & ind1990 <= 472L, "Transport, Comm, Util",
    ind1990 >= 500L & ind1990 <= 571L, "Wholesale Trade",
    ind1990 >= 580L & ind1990 <= 691L, "Retail Trade",
    ind1990 >= 700L & ind1990 <= 712L, "Finance, Ins, Real Estate",
    ind1990 >= 721L & ind1990 <= 760L, "Business & Repair Svcs",
    ind1990 >= 761L & ind1990 <= 791L, "Personal Svcs",
    ind1990 >= 800L & ind1990 <= 810L, "Entertainment & Rec Svcs",
    ind1990 >= 812L & ind1990 <= 893L, "Professional & Related Svcs",
    ind1990 >= 900L & ind1990 <= 932L, "Public Administration",
    ind1990 >= 940L & ind1990 <= 960L, "Active Duty Military",
    ind1990 == 991L | ind1990 == 992L, "Unemployed/NIU",
    default = NA_character_
  )]
  if ("ind_group" %in% names(cps)) cps[, ind_group := NULL]


  cps[, occ_group_cat := fcase(
      occ1990 >= 3L & occ1990 <= 194L, "Managerial/Professional",
      occ1990 >= 203L & occ1990 <= 389L, "Technical/Sales/Admin Support",
      occ1990 >= 403L & occ1990 <= 469L, "Service Occupations",
      occ1990 >= 473L & occ1990 <= 498L, "Farming/Forestry/Fishing",
      occ1990 >= 503L & occ1990 <= 699L, "Precision Prod/Craft/Repair",
      occ1990 >= 703L & occ1990 <= 889L, "Operators/Fabricators/Laborers",
      occ1990 >= 900L, "Military Occupations",
      default = NA_character_
  )]
  if ("occ_group" %in% names(cps)) cps[, occ_group := NULL]

  # --- Remove original source variables EARLY ---
  message("Removing original source variables after deriving categories...")
  cols_to_remove_early <- c("year", "month", "empstat", "labforce", "race", "occ1990", "ind1990",
                            "hispan", "educ", "relate", "metro", "cbsasz", "marst", "citizen",
                            "diffany", "classwkr", # Removed nativity, diffmob
                            "mish")
  existing_cols_to_remove_early <- intersect(cols_to_remove_early, names(cps))
  if (length(existing_cols_to_remove_early) > 0) {
      message("Removing early: ", paste(existing_cols_to_remove_early, collapse=", "))
      cps[, (existing_cols_to_remove_early) := NULL]
      gc()
  }
  # -----------------------------------------------------------

  # --- Aggregate Rate Calculations ---
  message("Calculating weighted labor force aggregates using data.table...")
  cps[, lf := emp + unemp]
  cps_in_lf <- cps[lf == 1L] # Subset for efficiency

  rates_ind <- cps_in_lf[!is.na(ind_group_cat), .(
    state_ind_emp_w = sum(wtfinl * emp, na.rm = TRUE),
    state_ind_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
    state_ind_lf_w = sum(wtfinl, na.rm = TRUE)
  ), by = .(date, statefip, ind_group_cat)]

  state_rates <- cps_in_lf[, .(
    state_emp_w = sum(wtfinl * emp, na.rm = TRUE),
    state_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
    state_lf_w = sum(wtfinl, na.rm = TRUE)
  ), by = .(date, statefip)]

  state_rates[, state_unemp_rate := fifelse(state_lf_w > 0, state_unemp_w / state_lf_w, NA_real_)]
  setorder(state_rates, statefip, date)
  state_rates[, state_emp_w_lag := shift(state_emp_w, type = "lag"), by = statefip]
  state_rates[, state_emp_pctchg := fifelse(!is.na(state_emp_w_lag) & state_emp_w_lag > 0, state_emp_w / state_emp_w_lag, 1.0)]
  state_rates[is.nan(state_emp_pctchg) | is.infinite(state_emp_pctchg), state_emp_pctchg := 1.0]
  state_rates[, state_emp_w_lag := NULL]
  state_rates_to_merge <- state_rates[, .(date, statefip, state_unemp_rate, state_emp_pctchg)]
  rm(state_rates); gc()

  industry_rates <- rates_ind[, .(
    ind_group_emp_w = sum(state_ind_emp_w, na.rm = TRUE),
    ind_group_unemp_w = sum(state_ind_unemp_w, na.rm = TRUE),
    ind_group_lf_w = sum(state_ind_lf_w, na.rm = TRUE)
  ), by = .(date, ind_group_cat)]
  rm(rates_ind); gc()
  setorder(industry_rates, ind_group_cat, date)
  industry_rates[, ind_emp_w_lag := shift(ind_group_emp_w, type = "lag"), by = ind_group_cat]
  industry_rates[, ind_emp_pctchg := fifelse(!is.na(ind_emp_w_lag) & ind_emp_w_lag > 0, ind_group_emp_w / ind_emp_w_lag, 1.0)]
  industry_rates[is.nan(ind_emp_pctchg) | is.infinite(ind_emp_pctchg), ind_emp_pctchg := 1.0]
  industry_rates[, ind_emp_w_lag := NULL]
  industry_rates_to_merge <- industry_rates[, .(date, ind_group_cat, ind_emp_pctchg)]
  rm(industry_rates); gc()

  national_rates <- cps_in_lf[, .(
      national_emp_w    = sum(wtfinl * emp,   na.rm = TRUE),
      national_unemp_w  = sum(wtfinl * unemp, na.rm = TRUE),
      national_lf_w     = sum(wtfinl,         na.rm = TRUE)
  ), by = date]
  national_rates[, national_unemp_rate := fifelse(national_lf_w > 0, national_unemp_w / national_lf_w, NA_real_)]
  setorder(national_rates, date)
  national_rates[, national_emp_w_lag := shift(national_emp_w, type = "lag")]
  national_rates[, national_emp_pctchg := fifelse(!is.na(national_emp_w_lag) & national_emp_w_lag > 0, national_emp_w / national_emp_w_lag, 1.0)]
  national_rates[is.nan(national_emp_pctchg) | is.infinite(national_emp_pctchg), national_emp_pctchg := 1.0]
  national_rates[, national_emp_w_lag := NULL]
  national_rates_base <- unique(national_rates[, .(date, national_unemp_rate, national_emp_pctchg)])
  setorder(national_rates_base, date)
  rm(national_rates, cps_in_lf); gc()

  message("Calculating national transition percentages since last period...")
  cps[, state := fcase(emp == 1L, "E", unemp == 1L, "U", nilf == 1L, "NE")]
  setkey(cps, cpsidp, date)
  cps[, state_next := shift(state, type = "lead"), by = cpsidp]
  trans_natl <- cps[!is.na(state_next) & !is.na(state) & wtfinl > 0,
    .(w = sum(wtfinl, na.rm = TRUE)),
    by = .(date, state, state_next)
  ]
  trans_start_totals <- trans_natl[, .(total_w = sum(w)), by = .(date, state)]
  trans_pct <- trans_natl[trans_start_totals, on = .(date, state)]
  trans_pct[, perc := fifelse(total_w > 0, w / total_w, 0)]
  trans_pct[, `:=`(w = NULL, total_w = NULL)]
  rm(trans_natl, trans_start_totals); gc()
  # Filter out transitions starting from NE before casting
  trans_wide <- dcast(trans_pct[state != "NE"], date ~ paste0(state, "_", state_next), value.var = "perc", fill = 0)
  rm(trans_pct); gc()
  national_rates_final <- national_rates_base[trans_wide, on = "date"]
  rm(national_rates_base, trans_wide); gc()

  if (!is.null(national_rates_output_path)) {
    message("Saving national aggregate rates to: ", national_rates_output_path)
    tryCatch({
      dir.create(dirname(national_rates_output_path), recursive = TRUE, showWarnings = FALSE)
      fwrite(national_rates_final, file = national_rates_output_path)
    }, error = function(e) {
      warning("Failed to save national rates file: ", e$message)
    })
  } else {
    message("National rates output path not provided, skipping save.")
  }

  message("Removing intermediate columns (lf, state, state_next) before merging...")
  intermediate_cols <- c("lf", "state", "state_next")
  existing_intermediate_cols <- intersect(intermediate_cols, names(cps))
  if (length(existing_intermediate_cols) > 0) {
      cps[, (existing_intermediate_cols) := NULL]
  }
  gc()

  # --- Merge Rates Back using Incremental Binding --- ## MODIFIED SECTION ##
  message("Merging aggregate rates onto individual data using incremental binding...")

  # Add year column for splitting
  cps[, year := year(date)]
  years <- sort(unique(cps$year)) # Sort years for sequential processing

  # Validate if any years exist
  if (length(years) == 0) {
      warning("No years found in the data to process for incremental binding.")
      # Depending on desired behavior, could return empty table or stop
      return(cps[, year := NULL]) # Return table with year column removed if it was added
  }

  # Initialize the final table with the first year's processed data
  first_year <- years[1]
  message("  Processing and initializing with first chunk: ", first_year)

  # Use explicit filtering to create the initial table and the remainder
  cps_final_combined <- cps[year == first_year]
  if (length(years) > 1) {
    cps <- cps[year != first_year] # Keep the rest only if there are more years
  } else {
    cps <- data.table() # Empty table if only one year
  }
  gc() # Clean up memory

  # --- Process the initial chunk ---
  if (nrow(cps_final_combined) > 0) { # Check if first chunk has data
      # Define transition columns dynamically (needed inside the scope accessing national_rates_final)
      base_national_cols <- c("date", "national_unemp_rate", "national_emp_pctchg")
      transition_cols_to_merge <- setdiff(names(national_rates_final), base_national_cols)
      cols_to_add_national <- c("national_unemp_rate", "national_emp_pctchg", transition_cols_to_merge)
      cols_from_i_national <- paste0("i.", cols_to_add_national) # Prefix for join source cols

      cols_to_add_state <- c("state_unemp_rate", "state_emp_pctchg")
      cols_from_i_state <- paste0("i.", cols_to_add_state) # Prefix for join source cols


      setkey(cps_final_combined, date) # Key for national join
      # Ensure columns exist before trying to merge them
      valid_national_cols <- intersect(cols_to_add_national, names(national_rates_final))
      valid_national_i_cols <- paste0("i.", valid_national_cols)
      if(length(valid_national_cols) > 0) {
          cps_final_combined[national_rates_final, on = "date", (valid_national_cols) := mget(valid_national_i_cols)]
      }

      setkey(cps_final_combined, date, statefip) # Key for state join
      valid_state_cols <- intersect(cols_to_add_state, names(state_rates_to_merge))
      valid_state_i_cols <- paste0("i.", valid_state_cols)
       if(length(valid_state_cols) > 0) {
          cps_final_combined[state_rates_to_merge, on = c("date", "statefip"), (valid_state_cols) := mget(valid_state_i_cols)]
      }

      setkey(cps_final_combined, date, ind_group_cat) # Key for industry join
      if ("ind_emp_pctchg" %in% names(industry_rates_to_merge)) {
         cps_final_combined[industry_rates_to_merge, on = c("date", "ind_group_cat"), `:=`(ind_emp_pctchg = i.ind_emp_pctchg)]
      }
  } else {
       message("    First chunk (year ", first_year, ") was empty. Initializing empty result table.")
       # Ensure cps_final_combined is an empty data.table if the first chunk was empty
       cps_final_combined <- data.table()
  }
  # --- End processing initial chunk ---


  # Loop through remaining years
  if (length(years) > 1 && nrow(cps) > 0) { # Check if there's remaining data
      # Re-define merge columns here in case they weren't defined due to empty first chunk
      base_national_cols <- c("date", "national_unemp_rate", "national_emp_pctchg")
      transition_cols_to_merge <- setdiff(names(national_rates_final), base_national_cols)
      cols_to_add_national <- c("national_unemp_rate", "national_emp_pctchg", transition_cols_to_merge)
      cols_from_i_national <- paste0("i.", cols_to_add_national)
      cols_to_add_state <- c("state_unemp_rate", "state_emp_pctchg")
      cols_from_i_state <- paste0("i.", cols_to_add_state)

      for (yr in years[-1]) {
          message("  Processing and binding chunk for year: ", yr)
          chunk <- cps[year == yr]
          # Check if this slice actually produced data
          if (nrow(chunk) == 0) {
              message("    Chunk for year ", yr, " is empty. Skipping.")
              # Explicitly remove the year from the remaining data even if empty
              cps <- cps[year != yr]
              gc()
              next # Skip to the next year
          }
          # Remove the processed year from the remaining pool
          cps <- cps[year != yr]
          gc()

          # Process the current chunk (same merge logic as above)
          setkey(chunk, date)
          valid_national_cols <- intersect(cols_to_add_national, names(national_rates_final))
          valid_national_i_cols <- paste0("i.", valid_national_cols)
          if(length(valid_national_cols) > 0) {
            chunk[national_rates_final, on = "date", (valid_national_cols) := mget(valid_national_i_cols)]
          }

          setkey(chunk, date, statefip)
          valid_state_cols <- intersect(cols_to_add_state, names(state_rates_to_merge))
          valid_state_i_cols <- paste0("i.", valid_state_cols)
           if(length(valid_state_cols) > 0) {
            chunk[state_rates_to_merge, on = c("date", "statefip"), (valid_state_cols) := mget(valid_state_i_cols)]
          }

          setkey(chunk, date, ind_group_cat)
           if ("ind_emp_pctchg" %in% names(industry_rates_to_merge)) {
             chunk[industry_rates_to_merge, on = c("date", "ind_group_cat"), `:=`(ind_emp_pctchg = i.ind_emp_pctchg)]
           }

          # Bind the processed chunk to the main table
          message("    Binding chunk...")
          # Ensure cps_final_combined exists before binding
          if (nrow(cps_final_combined) == 0) {
              cps_final_combined <- chunk # Assign first non-empty chunk
          } else {
              cps_final_combined <- rbindlist(list(cps_final_combined, chunk), use.names = TRUE, fill = FALSE)
          }
          message("    Done binding. Total rows: ", nrow(cps_final_combined))
          rm(chunk); gc() # Clean up aggressively
      }
  }

  # Clean up intermediate rate tables and remaining original data remnants
  rm(national_rates_final, state_rates_to_merge, industry_rates_to_merge, cps); gc()

  # Remove temporary year column if it exists
  if ("year" %in% names(cps_final_combined)) {
      cps_final_combined[, year := NULL]
  }
  message("Finished merging rates via incremental binding.")

  # --- Validation and Cleanup (use cps_final_combined now) ---
  message("Performing final validation and cleanup...")
  cps <- cps_final_combined # Assign back to 'cps' for rest of function
  rm(cps_final_combined); gc()
  # Check if cps actually contains data before proceeding
   if (nrow(cps) == 0) {
      warning("Data table is empty after merging process. Skipping validation and returning empty table.")
      return(cps) # Return the empty table
  }

  # Validation columns
  cps[, status_check := emp + unemp + nilf]
  cps[, status_error := fifelse(status_check != 1L, 1L, 0L)]

  error_count <- sum(cps$status_error, na.rm = TRUE)
  if(error_count > 0) {
    warning(paste0("Found ", error_count, " records with inconsistent employment classifications (emp+unemp+nilf != 1)."))
  }

  cols_to_remove_final <- c("status_check", "status_error")
  existing_cols_to_remove_final <- intersect(cols_to_remove_final, names(cps))
  if (length(existing_cols_to_remove_final) > 0) {
      message("Removing final intermediate columns: ", paste(existing_cols_to_remove_final, collapse=", "))
      cps[, (existing_cols_to_remove_final) := NULL]
  }

  message("De-duplicating data (if any duplicates remain)...")
  n_before <- nrow(cps)
  setkey(cps, cpsidp, date)
  cps <- unique(cps, by = key(cps))
  n_after <- nrow(cps)
  if (n_before > n_after) {
      message("Removed ", n_before - n_after, " duplicate observations (based on cpsidp, date)")
  }

  message("Finished processing variables. Returning data.table.")
  return(cps)
}


#' Add lead variables and filter to matched individuals using data.table
#'
#' @param cps Processed CPS data.table (output from process_cps_data)
#' @return Filtered CPS data.table with lead variables and employment states
add_lead_variables <- function(cps) {
  if (!is.data.table(cps)) setDT(cps)
   # Check if input is empty
  if (nrow(cps) == 0) {
    warning("Input data to add_lead_variables is empty. Returning empty data.table.")
    # Need to ensure expected output columns exist, even if empty
    expected_cols <- c("cpsidp", "date", "wtfinl", "age", "sex", "race_cat", "statefip", "mth_dim1", "mth_dim2",
                       "hispan_cat", "educ_cat", "durunemp", "ind_group_cat", "occ_group_cat", "relate_cat",
                       "metro_cat", "cbsasz_cat", "marst_cat", "citizen_cat",
                       "famsize", "nchild", "diff_cat", "classwkr_cat", "national_unemp_rate",
                       "state_unemp_rate", "state_emp_pctchg", "ind_emp_pctchg", "national_emp_pctchg",
                       "emp_state", "emp_state_f1") # Add known transition cols if predictable, else they won't exist
     # Find transition columns if they exist in the (empty) input
     # Exclude NE_E and NE_U from expected columns even if present in input (they shouldn't be)
    transition_cols_in_input <- grep("^[EU]{1}_[EUN]{1,2}$", names(cps), value = TRUE) # Only E_* and U_*
    final_expected_cols <- c(expected_cols, transition_cols_in_input)
    # Create an empty data table with these columns
    empty_dt <- data.table(matrix(ncol = length(final_expected_cols), nrow = 0))
    setnames(empty_dt, final_expected_cols)
    return(empty_dt)
  }
  setkey(cps, cpsidp, date)

  message("Adding lead variables using data.table...")
  lead_cols_demog <- c("age", "sex", "race_cat", "hispan_cat")
  lead_cols_status <- c("emp", "unemp", "nilf") # Still need these binary flags temporarily
  lead_cols <- c(lead_cols_demog, lead_cols_status)
  lead_cols_new_names <- paste0(lead_cols, "_f1")

  # Ensure lead columns exist before trying to operate on them
  missing_lead_cols <- setdiff(lead_cols, names(cps))
   if(length(missing_lead_cols) > 0){
      stop("Error in add_lead_variables: Missing required columns to create leads: ", paste(missing_lead_cols, collapse=", "))
   }


  cps[, (lead_cols_new_names) := shift(.SD, n = 1L, type = "lead"), .SDcols = lead_cols, by = cpsidp]

  message("Creating current and future employment state categories...")
  cps[, `:=`(
    emp_state = fcase(
      emp == 1L, "Employed",
      unemp == 1L, "Unemployed",
      nilf == 1L, "Not in Labor Force",
      default = NA_character_
    ),
    emp_state_f1 = fcase(
      emp_f1 == 1L, "Employed",
      unemp_f1 == 1L, "Unemployed",
      nilf_f1 == 1L, "Not in Labor Force",
      default = NA_character_
    )
  )]

  message("Filtering to matched individuals with valid future state and consistent demographics...")
  rows_before <- nrow(cps)

  is_equal_or_both_na <- function(a, b) {
    (a == b) | (is.na(a) & is.na(b))
  }

  # Need to handle cases where lead variables (_f1) might not exist if shift failed or original col missing
  # Check required _f1 columns exist before filtering
  required_f1_cols <- c("emp_state_f1", "age_f1", "sex_f1", "race_cat_f1", "hispan_cat_f1")
  missing_f1_cols <- setdiff(required_f1_cols, names(cps))
   if(length(missing_f1_cols) > 0){
       warning("Missing expected lead (_f1) columns for filtering: ", paste(missing_f1_cols, collapse=", "), ". Filtering might be incomplete.")
       # Handle missing columns gracefully in the filter expression if necessary, e.g., default to FALSE
       # For simplicity here, we'll assume the columns *should* exist if the leads were created.
       # A more robust approach might involve conditional filtering.
   }

  # Proceed with filtering, assuming columns exist (or handle NAs appropriately)
  cps_final <- cps[
    !is.na(emp_state_f1) &
    age == age_f1 & # Direct comparison
    sex == sex_f1 & # Direct comparison
    is_equal_or_both_na(race_cat, race_cat_f1) & # Use helper for potential NAs
    is_equal_or_both_na(hispan_cat, hispan_cat_f1) # Use helper for potential NAs
  ]


  rows_after <- nrow(cps_final)
  reduction_pct <- if (rows_before > 0) round((rows_before - rows_after) / rows_before * 100, 1) else 0
  message("Rows removed by matching/filtering: ", rows_before - rows_after,
          " (", reduction_pct, "% reduction)")

  # Remove temporary lead columns and intermediate binary status indicators
  cols_to_remove <- c(lead_cols_new_names, "emp", "unemp", "nilf")
  existing_cols_to_remove <- intersect(names(cps_final), cols_to_remove)
  if (length(existing_cols_to_remove) > 0) {
    message("Removing temporary lead/intermediate columns: ", paste(existing_cols_to_remove, collapse=", "))
    cps_final[, (existing_cols_to_remove) := NULL]
  }
  gc()

  message("Finished creating lead variables and filtering.")
  return(cps_final)
}


#' Label CPS variables for clarity
#'
#' @param data CPS data.table (output from add_lead_variables)
#' @return Data.table with labeled variables using attributes
label_variables <- function(data) {
    if (!is.data.table(data)) setDT(data)
    if (nrow(data) == 0) {
        message("Input data is empty, skipping labeling.")
        return(data) # Return empty data as is
    }

    message("Adding variable labels...")

    # Define labels in a list
    var_labels <- list(
        cpsidp = "IPUMS CPS Person Identifier (Longitudinal)",
        date = "Reference Date (YYYY-MM-01)",
        wtfinl = "Final Person Weight",
        age = "Age",
        sex = "Sex (1=Male, 2=Female)",
        race_cat = "Race (Categorical, based on IPUMS codes)",
        statefip = "State FIPS Code",
        mth_dim1 = "Month Sine Component (Cyclical Encoding)",
        mth_dim2 = "Month Cosine Component (Cyclical Encoding)",
        hispan_cat = "Hispanic Origin (Categorical)",
        educ_cat = "Educational Attainment (Categorical)",
        durunemp = "Duration of Unemployment in Weeks (0 if Employed/NILF)",
        ind_group_cat = "Industry Group (Categorical, based on IND1990)",
        occ_group_cat = "Occupation Group (Categorical, based on OCC1990)",
        relate_cat = "Relationship to Household Head (Categorical)",
        metro_cat = "Metropolitan Status (Categorical)",
        cbsasz_cat = "CBSA Size Category (Categorical)",
        marst_cat = "Marital Status (Categorical)",
        citizen_cat = "Citizenship Status (Categorical)",
        famsize = "Family Size (Numeric)",
        nchild = "Number of Own Children in Household (Numeric)",
        diff_cat = "Any Difficulty Reported (Yes/No/NIU)",
        classwkr_cat = "Class of Worker (Categorical)",
        national_unemp_rate = "Monthly National Unemployment Rate (Weighted)",
        state_unemp_rate = "Monthly State Unemployment Rate (Weighted)",
        state_emp_pctchg = "Monthly State Employment Level Ratio (Current/Previous)",
        ind_emp_pctchg = "Monthly Industry Group Employment Level Ratio (Current/Previous)",
        national_emp_pctchg = "Monthly National Employment Level Ratio (Current/Previous)",
        # Dynamic transition labels
        E_E = "National E->E Transition Pct (% of prior E pop)",
        E_U = "National E->U Transition Pct (% of prior E pop)",
        E_NE = "National E->NE Transition Pct (% of prior E pop)",
        U_E = "National U->E Transition Pct (% of prior U pop)",
        U_U = "National U->U Transition Pct (% of prior U pop)",
        U_NE = "National U->NE Transition Pct (% of prior U pop)",
        # NE_E label removed
        # NE_U label removed
        NE_NE = "National NE->NE Transition Pct (% of prior NE pop)", # Keep NE_NE if needed, otherwise remove
        # Final state variables
        emp_state = "Current Month Employment Status (Categorical)",
        emp_state_f1 = "Next Month Employment Status (Categorical)"
    )

    # Apply labels using setattr
    for (var_name in names(var_labels)) {
        if (var_name %in% names(data)) {
            setattr(data[[var_name]], "label", var_labels[[var_name]])
        }
    }

    message("Finished labeling variables.")
    return(data)
}