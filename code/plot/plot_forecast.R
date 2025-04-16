# R script to batch-generate forecast plots

# --- Load Required Packages ---
# Ensure these packages are installed: install.packages(c("ggplot2", "dplyr", "lubridate", "tidyr", "scales", "here"))
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(lubridate)
  library(tidyr)
  library(scales)
  library(here) # Using here() for potentially better path management
})

#' Plot Unemployment Forecast (Internal Function)
#' @noRd # Indicate it's primarily for internal use within the script
plot_unemployment_forecast_internal <- function(national_rates_file,
                                                forecast_agg_df,
                                                sample_urs_over_time = NULL,
                                                output_path,
                                                simulation_start_period) { # Keep param for context if needed elsewhere

  message(paste("  Generating plot for period:", simulation_start_period))

  # --- 1. Load and Prepare Historical Data ---
  historical_df_full <- NULL # Keep the full historical data
  if (!is.null(national_rates_file) && file.exists(national_rates_file)) {
      tryCatch({
          historical_df_full <- read.csv(national_rates_file) %>%
              mutate(date = floor_date(as.Date(date), "month")) %>%
              rename(unemployment_rate = national_unemp_rate) %>%
              select(date, unemployment_rate) %>%
              filter(!is.na(date), !is.na(unemployment_rate)) %>%
              arrange(date)
      }, error = function(e) {
          warning(paste("  Warning: Failed to load/process historical rates:", e$message))
          historical_df_full <<- NULL
      })
  } else {
      warning("  Warning: Historical national rates file not found or not provided.")
  }

  # --- 2. Prepare Forecast Data ---
  if (is.null(forecast_agg_df) || nrow(forecast_agg_df) == 0) {
    stop("Aggregated forecast data frame is NULL or empty.")
  }

  forecast_agg_df <- forecast_agg_df %>%
    mutate(
      date = floor_date(as.Date(date), "month") # Ensure first of month
    ) %>%
    filter(!is.na(date), !is.na(unemployment_rate_forecast),
           !is.na(unemployment_rate_p10), !is.na(unemployment_rate_p90)) %>%
    arrange(date)

  if (nrow(forecast_agg_df) == 0) {
    stop("No valid aggregated forecast data after cleaning.")
  }
  forecast_start_date <- min(forecast_agg_df$date) # This is the actual start date of forecast
  forecast_end_date <- max(forecast_agg_df$date)
  message(paste("Forecast data starts from:", forecast_start_date))

  # --- 3. Prepare Sample Data (if provided) ---
  plot_sample_df <- NULL
  if (!is.null(sample_urs_over_time) && nrow(sample_urs_over_time) > 0 && ncol(sample_urs_over_time) > 0) {
      message("  Preparing sample trajectory data for plotting...")
      # Assume columns are periods (YYYYMM), potentially prefixed with 'X' by R.
      # Handle potential first column being row names or index (done in batch function)
      period_cols <- grep("^(X)?\\d{6}$", names(sample_urs_over_time), value = TRUE) # Adjusted grep pattern
      if (length(period_cols) == 0) {
          warning("  Warning: Could not find period columns (YYYYMM format, possibly X-prefixed) in sample data. Skipping samples.")
          plot_sample_df <- NULL
      } else {
          # Ensure column names used in pivot_longer match the actual names found
          sample_data_long <- sample_urs_over_time %>%
              select(all_of(period_cols)) %>% # Select only period columns
              mutate(sample_id = row_number()) %>%
              # Use names_pattern if names might have 'X' prefix to extract numeric part
              pivot_longer(
                  cols = all_of(period_cols),
                  names_to = "period_str_raw", # Keep original name temporarily
                  values_to = "unemployment_rate"
              ) %>%
              mutate(
                  # Extract numeric period string, removing potential 'X' prefix
                  period_str = sub("^X", "", period_str_raw),
                  year = floor(as.numeric(period_str) / 100),
                  month = as.numeric(period_str) %% 100,
                  # Create date as first of the month
                  date = floor_date(as.Date(paste0(year, "-", sprintf("%02d", month), "-01")), "month")
              ) %>%
              filter(!is.na(date)) %>%
              select(sample_id, date, unemployment_rate) %>%
              arrange(sample_id, date)

          # Prepend last historical point to each sample trajectory for continuity
          if (!is.null(historical_df_full)) { # Use full history to find last point before forecast
              last_hist_point_df <- historical_df_full %>%
                  filter(date < forecast_start_date) %>% # Find points strictly before forecast
                  filter(date == max(date))

              if(nrow(last_hist_point_df) > 0 && nrow(sample_data_long) > 0 && min(sample_data_long$date) > last_hist_point_df$date[1]) {
                  connection_points_samples <- sample_data_long %>% distinct(sample_id) %>%
                      mutate(
                          date = last_hist_point_df$date[1],
                          unemployment_rate = last_hist_point_df$unemployment_rate[1]
                      )
                  plot_sample_df <- bind_rows(connection_points_samples, sample_data_long) %>% arrange(sample_id, date)
              } else {
                  plot_sample_df <- sample_data_long
              }
          } else {
               plot_sample_df <- sample_data_long # Cannot prepend if no history
          }
      }
  }

  # --- 4. Combine Data and Determine Plot Range ---
  # Keep forecast_start_date calculation
  forecast_end_date <- max(forecast_agg_df$date)

  # Define desired display window: 1 year before forecast start, 1 year after forecast end
  display_start_date <- forecast_start_date %m-% years(1)
  display_end_date <- forecast_end_date %m+% years(1)

  # Prepend last historical point to mean forecast for continuity
  plot_forecast_df_with_connection <- forecast_agg_df # Start with original forecast data
  if (!is.null(historical_df_full)) { # Use full history to find last point
      last_hist_point_df <- historical_df_full %>%
          filter(date < forecast_start_date) %>%
          filter(date == max(date))

      if (!is.null(last_hist_point_df) && nrow(last_hist_point_df) > 0 && nrow(forecast_agg_df) > 0 && forecast_agg_df$date[1] > last_hist_point_df$date[1]) {
          connection_point <- data.frame(
              date = last_hist_point_df$date[1],
              unemployment_rate_forecast = last_hist_point_df$unemployment_rate[1],
              unemployment_rate_p10 = last_hist_point_df$unemployment_rate[1],
              unemployment_rate_p90 = last_hist_point_df$unemployment_rate[1]
          )
          plot_forecast_df_with_connection <- bind_rows(connection_point, forecast_agg_df) %>% arrange(date)
      }
  }
  # --- Filter ALL data to the display window BEFORE plotting ---
  # Filter historical data for display
  historical_df_display <- NULL
  historical_df_post_display <- NULL # For overplotting dashed line
  if (!is.null(historical_df_full)) {
      historical_df_display <- historical_df_full %>% filter(date >= display_start_date & date <= display_end_date)
      # Create the data for the dashed overlay, including the connection point
      last_pre_forecast_point_for_hist <- historical_df_full %>%
          filter(date < forecast_start_date) %>%
          filter(date == max(date))
      historical_df_post_overlay_data <- historical_df_full %>% filter(date >= forecast_start_date)

      # Add connection point if needed and filter to display window
      if(nrow(last_pre_forecast_point_for_hist) > 0 && nrow(historical_df_post_overlay_data) > 0 && historical_df_post_overlay_data$date[1] > last_pre_forecast_point_for_hist$date[1]) {
           historical_df_post_overlay_data <- bind_rows(last_pre_forecast_point_for_hist, historical_df_post_overlay_data)
      }
      historical_df_post_display <- historical_df_post_overlay_data %>% filter(date >= display_start_date & date <= display_end_date)

  }

  # Filter forecast data (mean line includes connection point, ribbon does not)
  if (!is.null(plot_forecast_df_with_connection)) {
    plot_forecast_mean_line_display <- plot_forecast_df_with_connection %>% filter(date >= display_start_date & date <= display_end_date)
  } else {
    plot_forecast_mean_line_display <- NULL
  }

  # Filter the ORIGINAL forecast data (no connection point) for the RIBBON
  if (!is.null(forecast_agg_df)) {
    plot_forecast_ribbon_display <- forecast_agg_df %>% filter(date >= display_start_date & date <= display_end_date)
  } else {
    plot_forecast_ribbon_display <- NULL # Should not happen
  }

  if (!is.null(plot_sample_df)) {
    plot_sample_df_display <- plot_sample_df %>% filter(date >= display_start_date & date <= display_end_date)
  } else {
    plot_sample_df_display <- NULL
  }
  # --- End Filtering ---


  # --- 5. Create Plot ---
  gg <- ggplot() +
    theme_minimal(base_size = 12)

  # Layer 1: Sample Trajectories (Optional, plotted first, lighter)
  if (!is.null(plot_sample_df_display) && nrow(plot_sample_df_display) > 0) {
    gg <- gg + geom_line(data = plot_sample_df_display, aes(x = date, y = unemployment_rate, group = sample_id), color = "steelblue", alpha = 0.15, linewidth = 0.4)
  }

  # Layer 2a: Full Historical Data (within window) - Solid Line
  if (!is.null(historical_df_display) && nrow(historical_df_display) > 0) {
    gg <- gg + geom_line(data = historical_df_display, aes(x = date, y = unemployment_rate, color = "Actual"), linetype = "solid", linewidth = 0.8)
  }

  # Layer 2b: Post-Forecast Historical Data (within window) - Dashed Line (Overplotted)
  if (!is.null(historical_df_post_display) && nrow(historical_df_post_display) > 0) {
    # Use the same color mapping but change linetype
    gg <- gg + geom_line(data = historical_df_post_display, aes(x = date, y = unemployment_rate, color = "Actual"), linetype = "dashed", linewidth = 0.8)
  }

  # Layer 3: Confidence Interval Ribbon - Use RIBBON specific filtered data (no connection point)
  if (!is.null(plot_forecast_ribbon_display) && nrow(plot_forecast_ribbon_display) > 0) {
    gg <- gg + geom_ribbon(data = plot_forecast_ribbon_display, aes(x = date, ymin = unemployment_rate_p10, ymax = unemployment_rate_p90, fill = "Forecast (80% CI)"), alpha = 0.3)
  }

  # Layer 4: Mean Forecast Line - Use MEAN LINE specific filtered data (with connection point)
  if (!is.null(plot_forecast_mean_line_display) && nrow(plot_forecast_mean_line_display) > 0) {
    gg <- gg + geom_line(data = plot_forecast_mean_line_display, aes(x = date, y = unemployment_rate_forecast, color = "Forecast (Mean)"), linetype = "dashed", linewidth = 1.0)
  }

  # --- 6. Formatting ---
  # Format the ACTUAL forecast_start_date for the title
  if (!is.null(forecast_start_date)) {
      # Format as "Mon. YYYY" e.g., "Jan. 2022"
      formatted_start_period <- format(forecast_start_date, "%b. %Y")
      plot_title <- paste("Unemployment Rate Forecast (Starting", formatted_start_period, ")")
  } else {
      # Fallback title if forecast_start_date is somehow NULL (shouldn't happen after checks)
      plot_title <- paste("Unemployment Rate Forecast")
  }


  # Define colors and fills
  color_values <- c("Actual" = "black", "Forecast (Mean)" = "blue")
  fill_values <- c("Forecast (80% CI)" = "skyblue")

  # Add color for samples if plotted, but don't add dummy point
  if (!is.null(plot_sample_df) && nrow(plot_sample_df) > 0) {
      # We still need the color defined if we want to potentially reference it,
      # but we won't map it via aes() for the legend.
      # The actual sample lines use color = "steelblue" directly in geom_line.
      # color_values <- c(color_values, "Samples" = "steelblue") # Keep if needed elsewhere, otherwise remove
  }

  gg <- gg +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 0.1),
      # Reduce upper expansion slightly
      expand = expansion(mult = c(0.05, 0.05)) # Add 5% padding below, 5% padding above data range
    ) +
    scale_x_date(
      date_breaks = "3 months", # Ticks every 3 months
      date_labels = "%b %Y",   # Format: Jan 2023
      limits = c(display_start_date, display_end_date), # Set the display window
      expand = expansion(mult = c(0.01, 0.01)) # Minimal padding on x-axis
    ) +
    scale_color_manual(
      name = "Series",
      values = color_values, # Will only contain "Actual" and "Forecast (Mean)" now
      guide = guide_legend(override.aes = list(
        # Legend guide only needs entries for mapped aesthetics ("Actual", "Forecast (Mean)")
        linetype = c(if ("Actual" %in% names(color_values)) "solid" else NULL,
                     if ("Forecast (Mean)" %in% names(color_values)) "dashed" else NULL),
        linewidth = c(if ("Actual" %in% names(color_values)) 0.8 else NULL,
                      if ("Forecast (Mean)" %in% names(color_values)) 1.0 else NULL),
        alpha = c(if ("Actual" %in% names(color_values)) 1 else NULL,
                  if ("Forecast (Mean)" %in% names(color_values)) 1 else NULL)
      ))
    ) +
    scale_fill_manual(
      name = "Interval",
      values = fill_values
    ) +
    labs(
      title = plot_title, # Use the newly formatted title
      x = "Date",
      y = "Unemployment Rate"
    ) +
    theme_minimal(base_size = 14) + # Slightly larger base font size
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11), # Adjust size if needed
      axis.text.y = element_text(size = 11),
      axis.title = element_text(size = 13, face = "bold"), # Combined axis title styling
      legend.position = "bottom",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16, margin = margin(b = 10)), # Add bottom margin to title
      panel.grid.major = element_line(color = "grey90", linewidth = 0.5), # Slightly lighter grid lines
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(10, 10, 10, 10) # Add some margin around the plot
    )

  # --- 7. Save Plot ---
  tryCatch({
    ggsave(output_path, plot = gg, width = 12, height = 6, dpi = 300, units = "in") # Slightly smaller default size
    message(paste("  Plot saved successfully to:", basename(output_path)))
  }, error = function(e) {
    warning(paste("  Error saving plot:", e$message))
  })

  invisible(gg) # Return the plot object
}

#' Find all forecast files in a directory (Internal Function)
#' @noRd
find_forecast_files <- function(forecast_dir, pattern = "transformer_forecast_results_\\d+\\.csv$") {
  # Ensure the directory exists
  if (!dir.exists(forecast_dir)) {
    stop("Forecast directory does not exist: ", forecast_dir)
  }

  # Find all files matching the pattern
  all_files <- list.files(path = forecast_dir,
                         pattern = pattern,
                         full.names = TRUE)

  if (length(all_files) == 0) {
    warning("No forecast files found matching pattern in directory: ", forecast_dir)
    return(data.frame())
  }

  # Extract start periods from filenames
  start_periods <- gsub(".*transformer_forecast_results_(\\d+)\\.csv$", "\\1", basename(all_files))

  # Check for corresponding raw samples files
  samples_files_potential <- gsub("forecast_results", "forecast_raw_samples_ur", all_files)
  samples_exist <- file.exists(samples_files_potential)

  # Create a data frame with file info
  result <- data.frame(
    forecast_file = all_files,
    samples_file = ifelse(samples_exist, samples_files_potential, NA_character_), # Store path only if exists
    start_period = as.numeric(start_periods),
    has_samples = samples_exist,
    stringsAsFactors = FALSE
  )

  # Sort by start period
  result <- result[order(result$start_period), ]

  message("Found ", nrow(result), " forecast files for periods: ",
         paste(result$start_period, collapse = ", "))

  return(result)
}

#' Process all available forecast files in a directory (Internal Function)
#' @noRd
batch_process_forecasts_internal <- function(forecast_dir,
                                             national_rates_file,
                                             output_dir = NULL,
                                             pattern = "transformer_forecast_results_\\d+\\.csv$") {

  # Use same directory as input if not specified
  if (is.null(output_dir)) {
    output_dir <- forecast_dir
  }

  # Ensure output directory exists
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Find all forecast files
  forecast_files <- find_forecast_files(forecast_dir, pattern)

  if (nrow(forecast_files) == 0) {
    message("No forecast files found to process.")
    return(character(0))
  }

  # Process each file
  output_files <- character(length = nrow(forecast_files))

  for (i in 1:nrow(forecast_files)) {
    file_info <- forecast_files[i, ]
    message("\nProcessing forecast ", i, " of ", nrow(forecast_files),
           ": Period ", file_info$start_period)

    tryCatch({
      # Read forecast data
      forecast_agg_df <- read.csv(file_info$forecast_file)

      # Read samples data if available
      sample_urs_over_time <- NULL
      if (file_info$has_samples && !is.na(file_info$samples_file)) {
        message("  Reading sample trajectories from ", basename(file_info$samples_file))
        # Handle potential empty first column from CSV write (e.g., row names)
        temp_samples <- read.csv(file_info$samples_file)
        # Check if the first column is unnamed and likely an index
        if (names(temp_samples)[1] %in% c("", "X")) {
            sample_urs_over_time <- temp_samples[, -1]
            message("  (Removed unnamed first column from samples file)")
        } else {
            sample_urs_over_time <- temp_samples
        }
      } else {
        message("  No sample trajectory file found or specified for this period.")
      }

      # Set output path for plot
      output_path <- file.path(output_dir,
                             paste0("transformer_unemployment_forecast_",
                                   file_info$start_period, ".png"))

      # Generate plot using the internal function name
      plot_unemployment_forecast_internal(
        national_rates_file = national_rates_file,
        forecast_agg_df = forecast_agg_df,
        sample_urs_over_time = sample_urs_over_time,
        output_path = output_path,
        simulation_start_period = file_info$start_period
      )

      output_files[i] <- output_path
      # message("  Plot saved to: ", basename(output_path)) # Message moved inside plotting function

    }, error = function(e) {
      warning("  Error processing forecast file ", basename(file_info$forecast_file), ": ", e$message)
      # print(sys.calls()) # Uncomment for detailed debugging
      output_files[i] <- NA_character_
    })
  }

  # Report completion
  successful <- sum(!is.na(output_files))
  message("\nCompleted batch processing: ", successful, " of ", length(output_files),
         " forecasts successfully plotted.")

  return(output_files[!is.na(output_files)])
}

# --- Main Execution Block ---

# Define file paths (adjust as necessary)
# Using here() assumes the script is run from within the project root or similar structure
forecast_dir <- here::here("output", "forecast_transformer")
national_rates_file <- here::here("data", "processed", "national_unemployment_rate.csv")
output_dir <- here::here("output", "forecast_plots") # Save plots in a dedicated subfolder

# Create the output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  message(paste("Creating output directory:", output_dir))
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

message("\nStarting batch forecast plotting...")
message(paste("Forecast data directory:", forecast_dir))
message(paste("Historical data file:", national_rates_file))
message(paste("Output plot directory:", output_dir))

# Run the batch processing
generated_plots <- batch_process_forecasts_internal(
  forecast_dir = forecast_dir,
  national_rates_file = national_rates_file,
  output_dir = output_dir
  # pattern = "transformer_forecast_results_\\d+\\.csv$" # Default pattern is usually fine
)

if (length(generated_plots) > 0) {
  message("\nSuccessfully generated plots:")
  # Print relative paths for cleaner output if possible
  tryCatch({
      relative_paths <- fs::path_rel(generated_plots, start = here::here())
      for(p in relative_paths) message(paste(" -", p))
  }, error = function(e) { # Fallback if fs not available or paths are complex
      for(p in generated_plots) message(paste(" -", p))
  })

} else {
  message("\nNo plots were generated.")
}

message("\nScript finished.")