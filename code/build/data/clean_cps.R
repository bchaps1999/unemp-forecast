# Module for CPS data cleaning and processing
# =========================================

#' Load required packages for CPS data processing
#' @return NULL
load_cps_packages <- function() {
  packages <- c("ipumsr", "janitor", "lubridate", "plm", "tidyverse", "dotenv", "data.table")
  missing_packages <- setdiff(packages, rownames(installed.packages()))
  if (length(missing_packages) > 0) install.packages(missing_packages)
  invisible(lapply(packages, library, character.only = TRUE))
  
  options(scipen = 999, digits = 4)
  set.seed(8675309)
}

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
  
  # Scrape if needed
  if (!file.exists(samples_file) || force_scrape) {
    message("Scraping sample IDs from IPUMS...")
    scraper_path <- file.path(project_root, "code", "build", "data", "scrape_cps_samples.R")
    if (!file.exists(scraper_path)) {
      stop("Cannot find scrape_cps_samples.R script")
    }
    
    source(scraper_path)
    samples_file <- scrape_cps_samples(metadata_dir, force = force_scrape)
    if (is.null(samples_file)) stop("Failed to scrape CPS sample IDs")
  }
  
  # Parse dates and read samples
  start_date <- as.Date(paste0(start_date, "-01"))
  end_date <- as.Date(paste0(end_date, "-01"))
  
  message("Loading CPS sample IDs from ", samples_file)
  samples_df <- read_csv(samples_file, show_col_types = FALSE) %>%
    mutate(
      sample_year = as.numeric(sample_year),
      sample_month = as.numeric(sample_month),
      date = as.Date(paste0(sample_year, "-", sample_month, "-01"))
    ) %>%
    filter(date >= start_date, date <= end_date)
  
  # Apply ASEC filter if requested
  if (!include_asec) {
    samples_df <- samples_df %>% filter(!asec)
  }
  
  return(samples_df)
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
#' @return CPS data frame
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
  message("refresh_extract parameter is: ", refresh_extract)
  
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
    message("Forcing refresh extract, ignoring existing files")
    data_files <- character(0)
    ddi_files <- character(0)
  }
  
  # Use existing files only if refresh_extract is FALSE and files exist
  if (length(data_files) > 0 && length(ddi_files) > 0) {
    message("Using existing extract files")
    cps_data <- read_ipums_micro(ddi_files[1])
    if (debug) message("Raw data dimensions (from existing files): ", paste(dim(cps_data), collapse = " x "))
  } else {
    message("Requesting new extract from IPUMS API...")
    
    # Define variables for extract - added WHYUNEMP and WTFINL
    variables <- c("CPSIDP", "AGE", "SEX", "RACE", "HISPAN", 
                  "LABFORCE", "EMPSTAT", "AHRSWORKT", "FAMINC", 
                  "EDUC", "DURUNEMP", "IND1990", "OCC1990", "STATEFIP",
                  "MONTH", "YEAR", "MISH", "WHYUNEMP", "WTFINL") # Added WTFINL
    
    message("Submitting new extract request...")
    extract <- define_extract_micro(
      collection = "cps",
      description = "CPS extract for unemployment analysis",
      samples = samples,
      variables = variables,
      data_format = "csv",
      data_structure = "rectangular",
      rectangular_on = "P"
    )
    
    submitted_extract <- submit_extract(extract)
    message("Extract number: ", submitted_extract$number)
    
    wait_for_extract(submitted_extract)
    data_files <- download_extract(submitted_extract, download_dir = download_dir)
    cps_data <- read_ipums_micro(data_files)
    if (debug) message("Raw data dimensions (from new extract): ", paste(dim(cps_data), collapse = " x "))
  }
  
  # Initial cleaning and date column creation
  message("Creating date column and basic cleaning...")
  cps_data <- cps_data %>%
    clean_names() %>%
    as.data.frame() %>% # Ensure it's a data.frame before data.table conversion
    mutate(date = ymd(paste0(year, "-", month, "-01")))
  
  # Convert to data.table for efficiency
  setDT(cps_data)
  if (debug) message("Converted raw data to data.table")
  
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
      message("Unique individuals after date subsetting: ", length(unique(cps_data$cpsidp)))
    }
  }
  
  # Continue with remaining processing - more efficient now with smaller dataset
  message("Processing employment status and filtering for longitudinal analysis using data.table...")
  rows_before <- nrow(cps_data)
  
  # Ensure sorting for lead calculation
  setkey(cps_data, cpsidp, date) 
  
  # Calculate month difference using data.table's shift
  cps_data[, lead_date := shift(date, type = "lead"), by = cpsidp]
  cps_data[, month_diff := interval(date, lead_date) %/% months(1)]
  
  # Apply filters using data.table syntax
  cps_data <- cps_data[age >= 16 & labforce != 0 & month_diff == 1]
  
  # Remove temporary columns
  cps_data[, `:=`(lead_date = NULL, month_diff = NULL)]
  
  rows_after <- nrow(cps_data)
  message("Rows removed by age/labor force/follow-up filtering: ", rows_before - rows_after,
          " (", round((rows_before - rows_after) / rows_before * 100, 1), "% reduction)")
  
  # Remove unnecessary columns - ensure wtfinl is NOT removed
  cols_to_remove <- c("serial", "asecflag", "lfproxy", "pernum", "cpsid", "d_year", "cpsidv")
  existing_cols_to_remove <- intersect(cols_to_remove, names(cps_data))
  if (length(existing_cols_to_remove) > 0) {
    message("Removing unnecessary columns: ", paste(existing_cols_to_remove, collapse=", "))
    # Use data.table's way to remove columns by reference
    cps_data[, (existing_cols_to_remove) := NULL]
  }
  
  if (debug) {
    message("Final processed data dimensions: ", paste(dim(cps_data), collapse = " x "))
    message("Unique individuals in final data: ", length(unique(cps_data$cpsidp)))
  }
  
  # Return as data.table
  return(cps_data) 
}

#' Process CPS data into panel format with derived variables
#'
#' @param cps CPS data frame (now expecting a data.table)
#' @return Processed CPS data as data.table
process_cps_data <- function(cps) {
  # Ensure input is a data.table
  setDT(cps) 
  
  # Create core employment status variables first
  cps[, `:=`(
    emp = fifelse(empstat %in% c(10, 12), 1, 0), # Use fifelse for speed
    unemp = fifelse(empstat %in% c(20, 21, 22), 1, 0),
    nilf = fifelse(labforce == 1, 1, 0) # Assuming labforce=1 means NILF based on context
  )]

  # Create other derived variables using data.table syntax
  cps[, `:=`(
    mth_dim1 = round(sin(month / 12 * 2 * pi), 4) + 1,
    mth_dim2 = round(cos(month / 12 * 2 * pi), 4) + 1,
    hispan = fcase( # Use fcase for speed
      hispan == 0, "Not hispanic",
      hispan > 0 & hispan < 900, "Hispanic",
      default = NA_character_
    ),
    ahrsworkt = fcase(
      ahrsworkt < 20, "<20",
      ahrsworkt < 40 & ahrsworkt >= 20, "20-40",
      ahrsworkt >= 40 & ahrsworkt != 999, "40+",
      default = NA_character_
    ),
    faminc = fcase(
      faminc <= 350, "< $10,000",
      faminc > 350 & faminc <= 600, "$10,000 - $24,999",
      faminc > 600 & faminc <= 830, "$25,000 - $74,999",
      faminc > 840 & faminc <= 842, "$75,000 - $149,999",
      faminc == 843, "$150,000+",
      default = NA_character_
    ),
    educ = fcase(
      educ > 1 & educ <= 73, "High school or less",
      educ > 73 & educ <= 122 & educ != 111, "Some college, no bachelor's degree",
      educ == 111, "Bachelor's degree",
      educ %in% c(123, 124, 125), "Advanced degree",
      default = NA_character_
    ),
    # Ensure type consistency: both yes (0L) and no (as.integer(durunemp)) are integers
    durunemp = fifelse(durunemp == 999 & unemp == 0, 0L, as.integer(durunemp)), 
    ind1990 = as.numeric(ind1990) # Keep as.numeric for now
  )]
  
  # Derive ind_group after ind1990 is numeric
  cps[, ind_group := fcase(
    ind1990 > 0 & ind1990 <= 32, "Agriculture, forestry, & fisheries",
    ind1990 > 32 & ind1990 <= 50, "Mining",
    ind1990 == 60, "Construction",
    ind1990 > 60 & ind1990 <= 392, "Manufacturing",
    ind1990 > 392 & ind1990 <= 472, "Transportation & utilities",
    ind1990 > 472 & ind1990 <= 571, "Wholesale trade",
    ind1990 > 571 & ind1990 <= 691, "Retail trade",
    ind1990 > 691 & ind1990 <= 712, "Finance, insurance, & real estate",
    ind1990 > 712 & ind1990 <= 760, "Business & repair services",
    ind1990 > 760 & ind1990 <= 791, "Personal services",
    ind1990 > 791 & ind1990 <= 810, "Entertainment & recreation services",
    ind1990 > 810 & ind1990 <= 893, "Professional & related services",
    ind1990 > 893 & ind1990 <= 932, "Public administration",
    ind1990 > 932 & ind1990 <= 960, "Active duty military",
    default = NA_character_
  )]
  
  # Derive occ_2dig and ensure wtfinl is numeric
  cps[, `:=`(
    occ_2dig = floor(as.numeric(occ1990)/10),
    wtfinl = as.numeric(wtfinl) # Ensure wtfinl is numeric
  )]

  # Calculate weighted labor force aggregates using data.table
  message("Calculating weighted labor force aggregates using data.table...")
  
  # Calculate state and industry rates efficiently
  rates <- cps[!is.na(ind_group), .(
    state_ind_emp_w = sum(wtfinl * emp, na.rm = TRUE),
    state_ind_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
    state_ind_lf_w = sum(wtfinl * (emp + unemp), na.rm = TRUE)
  ), by = .(date, statefip, ind_group)]
    
  # Aggregate to state level
  state_rates <- rates[, .(
    state_emp_w = sum(state_ind_emp_w, na.rm = TRUE),
    state_unemp_w = sum(state_ind_unemp_w, na.rm = TRUE),
    state_lf_w = sum(state_ind_lf_w, na.rm = TRUE)
  ), by = .(date, statefip)]
  state_rates[, `:=`(
    state_unemp_rate = fifelse(state_lf_w > 0, state_unemp_w / state_lf_w, NA_real_),
    state_emp_rate = fifelse(state_lf_w > 0, state_emp_w / state_lf_w, NA_real_)
  )]
  state_rates <- state_rates[, .(date, statefip, state_unemp_rate, state_emp_rate)] # Select columns

  # Aggregate to industry level
  industry_rates <- rates[, .(
    ind_group_emp_w = sum(state_ind_emp_w, na.rm = TRUE),
    ind_group_unemp_w = sum(state_ind_unemp_w, na.rm = TRUE),
    ind_group_lf_w = sum(state_ind_lf_w, na.rm = TRUE)
  ), by = .(date, ind_group)]
  industry_rates[, `:=`(
    ind_group_unemp_rate = fifelse(ind_group_lf_w > 0, ind_group_unemp_w / ind_group_lf_w, NA_real_),
    ind_group_emp_rate = fifelse(ind_group_lf_w > 0, ind_group_emp_w / ind_group_lf_w, NA_real_)
  )]
  industry_rates <- industry_rates[, .(date, ind_group, ind_group_unemp_rate, ind_group_emp_rate)] # Select columns

  # Aggregate to national level directly from cps for accuracy
  national_rates <- cps[, .(
      national_emp_w = sum(wtfinl * emp, na.rm = TRUE),
      national_unemp_w = sum(wtfinl * unemp, na.rm = TRUE),
      national_lf_w = sum(wtfinl * (emp + unemp), na.rm = TRUE)
  ), by = date]
  national_rates[, `:=`(
      national_unemp_rate = fifelse(national_lf_w > 0, national_unemp_w / national_lf_w, NA_real_),
      national_emp_rate = fifelse(national_lf_w > 0, national_emp_w / national_lf_w, NA_real_)
  )]
  national_rates <- unique(national_rates[, .(date, national_unemp_rate, national_emp_rate)]) # Select and unique

  # Merge rates back onto the main dataset using data.table joins
  message("Merging aggregate rates onto individual data using data.table joins...")
  cps[national_rates, on = "date", `:=`(national_unemp_rate = i.national_unemp_rate, national_emp_rate = i.national_emp_rate)]
  cps[state_rates, on = c("date", "statefip"), `:=`(state_unemp_rate = i.state_unemp_rate, state_emp_rate = i.state_emp_rate)]
  cps[industry_rates, on = c("date", "ind_group"), `:=`(ind_group_unemp_rate = i.ind_group_unemp_rate, ind_group_emp_rate = i.ind_group_emp_rate)]
  
  # Create validation column using data.table
  cps[, status_check := emp + unemp + nilf]
  cps[, status_error := fifelse(status_check != 1, 1, 0)]
  
  # Report any errors in classification
  error_count <- sum(cps$status_error, na.rm = TRUE)
  if(error_count > 0) {
    warning(paste0("Found ", error_count, " records with inconsistent employment classifications"))
  }
  
  # Remove validation columns and redundant source columns
  # Added hwtfinl to this list
  cols_to_remove <- c("status_check", "status_error", "year", "month", "empstat", "labforce", "occ1990", "ind1990", "hwtfinl") 
  existing_cols_to_remove <- intersect(cols_to_remove, names(cps))
  if (length(existing_cols_to_remove) > 0) {
      message("Removing intermediate/redundant columns: ", paste(existing_cols_to_remove, collapse=", "))
      cps[, (existing_cols_to_remove) := NULL]
  }

  # Check and remove duplicates using data.table::unique
  message("De-duplicating data using data.table...")
  n_before <- nrow(cps)
  # Ensure keys are set for unique, or specify by argument
  setkey(cps, cpsidp, date) 
  cps <- unique(cps, by = key(cps)) 
  n_after <- nrow(cps)
  if (n_before > n_after) {
      message("Removed ", n_before - n_after, " duplicate observations")
  }
  
  # Return data.table (removed pdata.frame conversion)
  message("Finished processing. Returning data.table.")
  return(cps) 
}

#' Add lead variables and filter to matched individuals using data.table
#'
#' @param cps CPS data.table
#' @return Filtered CPS data.table with lead variables
add_lead_variables <- function(cps) {
  # Ensure it's a data.table and sorted for lead calculation
  setDT(cps)
  setkey(cps, cpsidp, date) 
  
  message("Adding lead variables using data.table...")
  
  # Define columns for which to create leads
  lead_cols <- c("age", "sex", "race", "hispan", "emp", "unemp", "nilf")
  lead_cols_new <- paste0(lead_cols, "_f1")
  
  # Create lead variables using data.table::shift
  cps[, (lead_cols_new) := shift(.SD, n = 1L, type = "lead"), .SDcols = lead_cols, by = cpsidp]
  
  message("Creating current and future employment state categories...")
  # Create categorical employment state variables
  cps[, `:=`(
    emp_state = fcase(
      emp == 1, "employed",
      unemp == 1, "unemployed",
      nilf == 1, "not_in_labor_force",
      default = NA_character_
    ),
    emp_state_f1 = fcase(
      emp_f1 == 1, "employed",
      unemp_f1 == 1, "unemployed",
      nilf_f1 == 1, "not_in_labor_force",
      default = NA_character_
    )
  )]
  
  message("Filtering to matched individuals with valid future state...")
  # Filter using data.table syntax
  rows_before <- nrow(cps)
  cps_final <- cps[
    !is.na(emp_state_f1) & # Ensure future state is known
    age == age_f1 &        # Check demographic consistency
    sex == sex_f1 &
    race == race_f1 &
    hispan == hispan_f1
  ]
  rows_after <- nrow(cps_final)
  message("Rows removed by matching/filtering: ", rows_before - rows_after,
          " (", round((rows_before - rows_after) / rows_before * 100, 1), "% reduction)")
  
  # Remove temporary lead columns, original mish, and intermediate emp/unemp/nilf
  # Added emp, unemp, nilf to this list
  cols_to_remove <- c(lead_cols_new, "mish", "emp", "unemp", "nilf") 
  existing_cols_to_remove <- intersect(cols_to_remove, names(cps_final))
  if (length(existing_cols_to_remove) > 0) {
    message("Removing temporary/unnecessary columns: ", paste(existing_cols_to_remove, collapse=", "))
    cps_final[, (existing_cols_to_remove) := NULL]
  }
  
  return(cps_final)
}

#' Label CPS employment status variables
#' 
#' @param data CPS data frame
#' @return Data frame with labeled employment status variables
label_employment_status <- function(data) {
  # Add variable label attributes for documentation and analysis
  # Removed labels for emp, unemp, nilf as they are dropped later
  # attr(data$emp, "label") <- "Employed (EMPSTAT in 10,12)"
  # attr(data$unemp, "label") <- "Unemployed (EMPSTAT in 20,21,22)"
  # attr(data$nilf, "label") <- "Not in labor force (LABFORCE=0)"
  attr(data$emp_state, "label") <- "Current employment state category"
  attr(data$emp_state_f1, "label") <- "Next month employment state category"
  
  # Add labels for new rate variables if they exist
  if ("national_unemp_rate" %in% names(data)) attr(data$national_unemp_rate, "label") <- "Monthly national unemployment rate (weighted)"
  if ("national_emp_rate" %in% names(data)) attr(data$national_emp_rate, "label") <- "Monthly national employment rate (weighted)"
  if ("state_unemp_rate" %in% names(data)) attr(data$state_unemp_rate, "label") <- "Monthly state unemployment rate (weighted)"
  if ("state_emp_rate" %in% names(data)) attr(data$state_emp_rate, "label") <- "Monthly state employment rate (weighted)"
  if ("ind_group_unemp_rate" %in% names(data)) attr(data$ind_group_unemp_rate, "label") <- "Monthly industry group unemployment rate (weighted)"
  if ("ind_group_emp_rate" %in% names(data)) attr(data$ind_group_emp_rate, "label") <- "Monthly industry group employment rate (weighted)"
  
  return(data)
}