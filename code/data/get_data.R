#' CPS Data Pipeline
#'
#' This script provides a workflow for fetching, processing, and analyzing CPS data
#' from IPUMS, creating a panel dataset with employment state transitions.
#' When sourced, it runs the pipeline using the parameters defined below.
#'
#' @author Brendan Chapuis

message("--- get_data.R script started ---") # Add initial message

# Ensure 'here' package is available for robust path management
if (!requireNamespace("here", quietly = TRUE)) {
  # Installation handled by renv, just stop if not found after renv::load
  stop("Package 'here' is required but not found. Please ensure renv environment is active and restored.")
}
library(here)

# Set project root using here() - Finds the .Rproj file or .git folder
project_root <- here::here()
message("Project Root (using here): ", project_root)

# Source required modules using here() for robust paths
source(here::here("code", "data", "scrape_cps_samples.R"))
source(here::here("code", "data", "clean_cps.R"))

# --- Load Required Packages Directly ---
# Load packages needed by clean_cps.R here, after renv is loaded by main.R
message("--- Loading required CPS processing packages ---")
load_cps_packages() # Call the function defined in clean_cps.R

# --- Define Pipeline Parameters --- 
# These parameters will be used when the script is sourced.
output_file_param <- "data/processed/cps_transitions.csv"
# Add parameter for national rates file
national_rates_output_file_param <- "data/processed/national_unemployment_rate.csv"
# refresh_extract_param: If TRUE, forces a new download request to IPUMS.
# If FALSE, uses existing downloaded files if they cover the extract date range.
# Re-processing of the data happens whenever this script is sourced, regardless of this flag.
refresh_extract_param <- FALSE 
force_scrape_param <- FALSE
# Dates for the IPUMS extract request
extract_start_date_param <- "2000-01" 
extract_end_date_param <- "2025-03"
# Dates to subset the downloaded data for processing/analysis
subset_start_date_param <- "2005-01"
subset_end_date_param <- "2025-03"
include_asec_param <- FALSE
debug_param <- TRUE # Set to TRUE for verbose output during sourced run
# --- End Parameter Definition ---

#' Run the CPS data pipeline
#'
#' @param project_root Path to the project root directory
#' @param output_file Path to save the main output CSV (relative to project_root)
#' @param national_rates_output_file Path to save the national rates CSV (relative to project_root)
#' @param refresh_extract Whether to force retrieving a new extract from IPUMS
#' @param force_scrape Whether to force re-scraping sample IDs
#' @param extract_start_date Start date for IPUMS extract request (YYYY-MM)
#' @param extract_end_date End date for IPUMS extract request (YYYY-MM)
#' @param subset_start_date Start date for filtering the data subset (YYYY-MM)
#' @param subset_end_date End date for filtering the data subset (YYYY-MM)
#' @param include_asec Whether to include ASEC samples
#' @param debug Print debug information
#' @return Path to the created main output file
run_cps_pipeline <- function(
  project_root,
  output_file = "data/processed/cps_transitions.csv",
  national_rates_output_file = "data/processed/national_unemployment_rate.csv", # Add new parameter
  refresh_extract = FALSE,
  force_scrape = FALSE,
  extract_start_date = "2018-01",
  extract_end_date = "2020-12",
  subset_start_date = "2019-01",
  subset_end_date = "2019-06",
  include_asec = FALSE,
  debug = FALSE
) {
  # Validate date parameters
  validate_date <- function(date_str, date_name) {
    if (!grepl("^\\d{4}-\\d{2}$", date_str)) {
      stop(date_name, " must be in YYYY-MM format: ", date_str)
    }
    as.Date(paste0(date_str, "-01"))
  }
  
  extract_start_date_parsed <- validate_date(extract_start_date, "extract_start_date")
  extract_end_date_parsed <- validate_date(extract_end_date, "extract_end_date")
  subset_start_date_parsed <- validate_date(subset_start_date, "subset_start_date")
  subset_end_date_parsed <- validate_date(subset_end_date, "subset_end_date")
  
  if (extract_start_date_parsed >= extract_end_date_parsed) {
    stop("extract_start_date must be before extract_end_date")
  }
  if (subset_start_date_parsed >= subset_end_date_parsed) {
    stop("subset_start_date must be before subset_end_date")
  }
  if (subset_start_date_parsed < extract_start_date_parsed || subset_end_date_parsed > extract_end_date_parsed) {
    stop("Subset dates must be within or equal to extract dates")
  }
  
  if (debug) {
    message("Extract date range: ", extract_start_date, " to ", extract_end_date)
    message("Subset date range:  ", subset_start_date, " to ", subset_end_date)
  }

  # Start timing
  start_time <- Sys.time()
  message("Starting CPS data pipeline at ", format(start_time))
  message("refresh_extract set to: ", refresh_extract)
  
  # Setup
  output_path <- file.path(project_root, output_file)
  national_rates_path <- file.path(project_root, national_rates_output_file) # Define full path
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  # Ensure directory for national rates exists too
  dir.create(dirname(national_rates_path), recursive = TRUE, showWarnings = FALSE)
  init_ipums_api(project_root)
  
  # Step 1: Identify CPS samples (using extract dates)
  message("\n===== STEP 1: Identifying CPS samples =====")
  samples_df <- get_cps_samples(
    project_root = project_root,
    start_date = extract_start_date, # Use extract dates
    end_date = extract_end_date,     # Use extract dates
    include_asec = include_asec,
    force_scrape = force_scrape
  )
  
  if (debug) {
    message("Found ", nrow(samples_df), " samples")
    print(head(samples_df))
  }
  
  # Force refresh_extract to be a logical value to ensure correct type
  refresh_extract <- as.logical(refresh_extract)
  message("Using refresh_extract = ", refresh_extract)
  
  # Step 2: Get CPS data (passing both extract and subset dates)
  message("\n===== STEP 2: Fetching and subsetting CPS data =====")
  cps_data <- get_cps_data(
    project_root = project_root,
    refresh_extract = refresh_extract, 
    start_date = extract_start_date, 
    end_date = extract_end_date,
    subset_start_date = subset_start_date, # Pass subset dates directly
    subset_end_date = subset_end_date,     # Pass subset dates directly
    include_asec = include_asec,
    force_scrape = force_scrape,
    debug = debug
  )
  
  if (debug) {
    message("Data dimensions after fetch and subset: ", paste(dim(cps_data), collapse = " x "))
    message("Individuals in dataset: ", length(unique(cps_data$cpsidp)))
  }

  # Step 3: Process data into panel format (using already subsetted data)
  message("\n===== STEP 3: Creating panel dataset & Saving National Rates =====")
  # Pass the national_rates_path to process_cps_data
  cps_panel <- process_cps_data(cps_data, national_rates_output_path = national_rates_path)
  rm(cps_data); gc()
  
  if (debug) {
    message("Panel dimensions: ", paste(dim(cps_panel), collapse = " x "))
  }
  
  # Step 4: Add lead variables and finalize dataset
  message("\n===== STEP 4: Adding lead variables & finalizing dataset =====")
  final_data <- add_lead_variables(cps_panel)
  rm(cps_panel); gc()
  
  if (debug) {
    message("After matching: ", paste(dim(final_data), collapse = " x "))
    message("Matched individuals: ", length(unique(final_data$cpsidp)))
  }
  
  # Step 4.5: Add variable labels
  message("\n===== STEP 4.5: Labeling variables =====")
  final_data <- label_variables(final_data) 
  
  # Step 5: Save the processed data (individual level)
  message("\n===== STEP 5: Saving processed individual-level data =====")
  # Convert necessary columns before saving
  # Use data.table::fwrite for potentially faster writing
  final_dt <- as.data.table(final_data) # Ensure it's a data.table
  # Simplified conversions: fwrite handles numeric types well. Ensure date is Date.
  final_dt[, `:=`(
    date = as_date(date) # Keep date as Date object, fwrite handles it
  )]
  
  fwrite(final_dt, file = output_path)
  
  # Report completion
  end_time <- Sys.time()
  elapsed <- difftime(end_time, start_time, units = "mins")
  message("\nPipeline completed in ", round(as.numeric(elapsed), 2), " minutes")
  message("Individual-level output saved to: ", output_path)
  if (file.exists(national_rates_path)) {
      message("National rates output saved to: ", national_rates_path)
  } else {
      message("National rates file was not saved (check path or function).")
  }

  rm(final_data, final_dt); gc() # Clean up both objects
  return(output_path)
}

# --- Automatically Run Pipeline When Sourced ---
# Call the pipeline function directly using the parameters defined above
run_cps_pipeline(
  project_root = project_root,
  output_file = output_file_param,
  national_rates_output_file = national_rates_output_file_param, # Pass new param
  refresh_extract = refresh_extract_param,
  force_scrape = force_scrape_param,
  extract_start_date = extract_start_date_param,
  extract_end_date = extract_end_date_param,
  subset_start_date = subset_start_date_param,
  subset_end_date = subset_end_date_param,
  include_asec = include_asec_param,
  debug = debug_param
)
# --- End Automatic Run ---