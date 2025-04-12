# Module to scrape IPUMS CPS sample IDs table
# =====================================================

# Load required packages
load_scraper_packages <- function() {
  pkgs <- c("rvest", "dplyr", "stringr", "readr")
  missing <- setdiff(pkgs, rownames(installed.packages()))
  if (length(missing) > 0) install.packages(missing)
  invisible(lapply(pkgs, library, character.only = TRUE))
}

#' Scrape CPS sample IDs from IPUMS website
#'
#' @param output_dir Directory to save the sample IDs CSV file
#' @param force Whether to force scraping even if the file exists
#' @return Path to the created CSV file or NULL if failed
#' @export
scrape_cps_samples <- function(output_dir, force = FALSE) {
  load_scraper_packages()
  
  # Setup output
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_file <- file.path(output_dir, "cps_sample_ids.csv")
  
  # Check if file exists and force is FALSE
  if (!force && file.exists(output_file)) {
    message("CPS sample IDs file already exists. Use force=TRUE to re-scrape.")
    return(output_file)
  }
  
  # Define URL to scrape
  url <- "https://cps.ipums.org/cps-action/samples/sample_ids"
  message("Scraping CPS sample IDs from: ", url)
  
  # Scrape the sample table
  tryCatch({
    # Read and extract tables
    page <- read_html(url)
    tables <- html_table(page)
    
    # Find the table with Sample ID column
    sample_table <- NULL
    for (i in seq_along(tables)) {
      if (any(grepl("Sample ID", colnames(tables[[i]]), ignore.case = TRUE))) {
        sample_table <- tables[[i]]
        break
      }
    }
    
    if (is.null(sample_table)) {
      stop("Could not find the sample IDs table")
    }
    
    # Clean and process
    names(sample_table) <- tolower(gsub(" ", "_", names(sample_table)))
    samples_df <- sample_table %>%
      mutate(
        sample_year = str_extract(sample_id, "(?<=cps)\\d{4}"),
        sample_month = str_extract(sample_id, "(?<=_)\\d{2}"),
        asec = grepl("ASEC|Annual Social and Economic|March Supplement", description, ignore.case = TRUE)
      ) %>%
      select(sample_id, description, sample_year, sample_month, asec)
    
    # Save and return
    write_csv(samples_df, output_file)
    message("Saved ", nrow(samples_df), " CPS sample IDs")
    return(output_file)
    
  }, error = function(e) {
    message("Error scraping CPS samples: ", e$message)
    return(NULL)
  })
}
