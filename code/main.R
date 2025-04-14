# Main Orchestration Script

# --- 1. Setup ---
message("===== Starting Main R Orchestration Script =====")
start_time <- Sys.time()

# --- 2. Load renv Environment ---
message("\n--- Loading renv environment ---")
if (!requireNamespace("renv", quietly = TRUE)) {
  stop("Package 'renv' is required but not found. Run 'renv::init()'.")
}
library(renv)

# Use getwd() initially for project root. Confirm with here() later.
project_root <- getwd()
message("Initial Project Root guess (using getwd): ", project_root)

tryCatch({
  # Load renv environment
  renv::load(project = project_root)
  message("renv::load() completed.")

  # Explicitly set library path
  renv_lib_path <- renv::paths$library(project = project_root)
  message("renv library path: ", renv_lib_path)
  .libPaths(c(renv_lib_path, .libPaths())) # Prepend renv lib path
  message("Updated .libPaths(): ", paste(.libPaths(), collapse = "; "))

  # --- 3. Load 'here' package ---
  message("\n--- Loading 'here' package ---")
  if (!requireNamespace("here", quietly = TRUE)) {
    message("Search paths (.libPaths()): ", paste(.libPaths(), collapse = "\n"))
    message("Files in renv library path (", renv_lib_path, "):")
    print(list.files(renv_lib_path, recursive = TRUE, all.files = TRUE))
    stop("Package 'here' not found after setting lib paths. Run 'renv::restore()'.")
  }
  library(here)
  message("'here' package loaded successfully.")

}, error = function(e) {
  message("Error during renv setup or 'here' loading: ", e$message)
  stop("Failed to load renv environment or 'here' package. Run 'renv::init()' and 'renv::restore()'.")
})

# --- 3.1 Confirm 'here' and set project root ---
if (!"here" %in% .packages()) {
    stop("Failed to load 'here' package. Check renv status.")
}
project_root <- here::here() # Use here() now
message("Project Root confirmed (using here): ", project_root)

# --- 4. Run Data Preparation Script ---
message("\n--- Running Data Preparation (code/data/get_data.R) ---")
data_script_path <- here::here("code", "data", "get_data.R")
if (!file.exists(data_script_path)) {
  stop("Data preparation script not found: ", data_script_path)
}
tryCatch({
  message("Attempting to source: ", data_script_path)
  source(data_script_path, chdir = TRUE) # chdir=TRUE runs script relative to its own dir
  message("Data preparation script sourced successfully.")
}, error = function(e) {
  message("Error occurred while sourcing ", data_script_path, ":")
  message(e$message)
  stop("Data preparation script failed during sourcing.")
})

# --- 5. Prepare and Run Model Pipeline Shell Script ---
message("\n--- Preparing and Running Model Pipeline (code/model/run_model_pipeline.sh) ---")
model_script_path <- here::here("code", "model", "run_model_pipeline.sh")

if (!file.exists(model_script_path)) {
  stop("Model pipeline script not found: ", model_script_path)
}

# Ensure the script has execute permissions
message("Setting execute permissions for: ", model_script_path)
chmod_cmd <- paste("chmod +x", shQuote(model_script_path)) # Use shQuote for paths with spaces
system(chmod_cmd)

# Execute the shell script
message("Executing model pipeline script...")
exit_code <- system(shQuote(model_script_path)) # Use shQuote

if (exit_code != 0) {
  stop("Model pipeline script failed with exit code: ", exit_code)
} else {
  message("Model pipeline script executed successfully.")
}

# --- 6. Completion ---
end_time <- Sys.time()
elapsed_total <- difftime(end_time, start_time, units = "mins")
message("\n===== Main R Orchestration Script Finished =====")
message("Total execution time: ", round(as.numeric(elapsed_total), 2), " minutes.")
