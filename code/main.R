#!/usr/bin/env Rscript

# --- Main Orchestration Script ---

# --- 1. Setup ---
message("===== Starting Main R Orchestration Script =====")
start_time <- Sys.time()

# Ensure 'here' package is available for robust path management
if (!requireNamespace("here", quietly = TRUE)) {
  message("Installing 'here' package for robust path management...")
  install.packages("here")
}
library(here)

# Set project root using here() - Finds the .Rproj file or .git folder
project_root <- here::here()
message("Project Root (using here): ", project_root)
# No need to setwd() globally, here() manages relative paths

# --- 2. Load renv Environment ---
message("\n--- Loading renv environment ---")
tryCatch({
  if (!requireNamespace("renv", quietly = TRUE)) {
    stop("renv package not found. Please install it.")
  }
  # renv::load expects project path, here() provides it
  renv::load(project = project_root)
  message("renv environment loaded successfully.")
}, error = function(e) {
  message("Error loading renv environment: ", e$message)
  stop("Failed to load renv environment. Ensure it's initialized ('renv::init()').")
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
  source(data_script_path, chdir = TRUE) # chdir=TRUE ensures it runs relative to its own dir if needed
  message("Data preparation script executed successfully.")
}, error = function(e) {
  message("Error running data preparation script: ", e$message)
  stop("Data preparation failed.")
})

# --- 4. Prepare and Run Model Pipeline Shell Script ---
message("\n--- Preparing and Running Model Pipeline Shell Script (code/model/run_model_pipeline.sh) ---")
# Use here() to construct the path relative to the project root
model_pipeline_script <- here::here("code", "model", "run_model_pipeline.sh")

if (!file.exists(model_pipeline_script)) {
  stop("Model pipeline shell script not found: ", model_pipeline_script)
}

# Make the shell script executable
message("Setting execute permissions for shell script...")
tryCatch({
  # Sys.chmod might behave differently on Windows vs. Unix-like systems
  # mode="0755" is typical for executable scripts on Unix-like systems
  Sys.chmod(model_pipeline_script, mode = "0755", use_umask = FALSE)
  message("Execute permissions set for: ", model_pipeline_script)
}, warning = function(w) {
  message("Warning setting permissions (might be ignorable on Windows): ", w$message)
}, error = function(e) {
  # Don't stop if chmod fails (e.g., permissions issue, Windows), let system2 handle it
  message("Error setting execute permissions: ", e$message)
  message("Attempting to run the script anyway...")
})

# Run the shell script from the project root directory
message("Executing model pipeline shell script...")
original_wd <- getwd() # Store current WD
setwd(project_root)   # Change to project root
message("Temporarily changed WD to project root for system2: ", getwd())
result <- system2(command = "bash", args = shQuote(model_pipeline_script), stdout = TRUE, stderr = TRUE)
setwd(original_wd)   # Change back to original WD
message("Restored WD: ", getwd())

# Check execution status
if (length(attr(result, "status")) > 0 && attr(result, "status") != 0) {
  message("--- Model Pipeline Script FAILED ---")
  message("Exit Status: ", attr(result, "status"))
  message("Output/Error:")
  cat(result, sep = "\n")
  stop("Model pipeline script execution failed.")
} else {
  message("--- Model Pipeline Script FINISHED Successfully ---")
  # Print output if desired
  # message("Output:")
  # cat(result, sep = "\n")
}

# --- 5. Completion ---
end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")
message("\n===== Main R Orchestration Script Finished =====")
message("Total Elapsed Time: ", round(as.numeric(elapsed_time), 2), " minutes")
