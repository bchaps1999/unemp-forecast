#!/usr/bin/env Rscript

# --- Main Orchestration Script ---

# --- 1. Setup ---
message("===== Starting Main R Orchestration Script =====")
start_time <- Sys.time()

# Ensure 'here' package is available (renv should handle installation)
if (!requireNamespace("here", quietly = TRUE)) {
  stop("Package 'here' is required but not found. Please ensure renv environment is active and restored.")
}
library(here)

# Set project root using here() - Finds the .Rproj file or .git folder
project_root <- here::here()
message("Project Root (using here): ", project_root)
# No need to setwd() globally, here() manages relative paths

# --- 2. Load renv Environment ---
message("\n--- Loading renv environment ---")
# Ensure 'renv' package is available (renv should handle installation)
if (!requireNamespace("renv", quietly = TRUE)) {
  stop("Package 'renv' is required but not found. Has 'renv::init()' been run for this project?")
}
library(renv) # Load renv

tryCatch({
  # renv::load expects project path, here() provides it
  renv::load(project = project_root)
  message("renv environment loaded successfully.")
}, error = function(e) {
  message("Error loading renv environment: ", e$message)
  stop("Failed to load renv environment. Ensure it's initialized ('renv::init()') and potentially run 'renv::restore()'.")
})

# --- 3. Run Data Preparation Script ---
message("\n--- Running Data Preparation (code/data/get_data.R) ---")
# Use here() to construct the path relative to the project root
data_script_path <- here::here("code", "data", "get_data.R")
if (!file.exists(data_script_path)) {
  stop("Data preparation script not found: ", data_script_path)
}
tryCatch({
  # Source the script using the absolute path from here()
  # Sourcing changes the working directory temporarily, which is fine here.
  message("Attempting to source: ", data_script_path) # Add message before sourcing
  source(data_script_path, chdir = TRUE) # chdir=TRUE ensures it runs relative to its own dir if needed
  message("Data preparation script sourced successfully.") # Changed message
}, error = function(e) {
  # Print the specific error message from the source call
  message("Error occurred while sourcing ", data_script_path, ":")
  message(e$message)
  stop("Data preparation script failed during sourcing.")
})

# --- 4. Prepare and Run Model Pipeline Shell Script ---
# ...existing code...
