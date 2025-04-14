import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from pathlib import Path
import os
import sys
import gc # Import garbage collector

# Import scikit-learn components
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# --- Import Config ---
# Add parent directory to sys.path to allow config import
# sys.path.append(str(Path(__file__).parent)) # Removed - Assume config.py is importable
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Make sure it's in the same directory or sys.path is configured correctly.")
    sys.exit(1)

# --- Parameters & Configuration ---

def preprocess_data():
    """Main function to load, clean, preprocess, split, and save data."""
    # --- Setup Paths ---
    project_root_path = config.PROJECT_ROOT
    input_data_path = config.PREPROCESS_INPUT_FILE
    output_dir_path = config.PREPROCESS_OUTPUT_DIR
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Output files for different splits
    hpt_train_file = output_dir_path / config.HPT_TRAIN_DATA_FILENAME
    hpt_val_file = output_dir_path / config.HPT_VAL_DATA_FILENAME
    full_train_file = output_dir_path / config.FULL_TRAIN_DATA_FILENAME
    full_val_file = output_dir_path / config.FULL_VAL_DATA_FILENAME
    test_file = output_dir_path / config.TEST_DATA_FILENAME
    hpt_interval_data_file = output_dir_path / config.HPT_INTERVAL_DATA_FILENAME # Data for HPT objective metric
    # Other outputs
    preprocessor_file = output_dir_path / config.RECIPE_FILENAME
    metadata_file = output_dir_path / config.METADATA_FILENAME

    # --- Seed ---
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # --- 1. Load and Clean Raw Data ---
    print("===== STEP 1: Loading and Cleaning Raw Data =====")
    try:
        cps_df_raw = pd.read_csv(input_data_path, dtype={'cpsidp': str})
        print(f"Loaded raw data: {cps_df_raw.shape}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_data_path}")
        return
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    df = cps_df_raw.copy()
    del cps_df_raw # Free memory
    gc.collect()

    try:
        df['cpsidp'] = df['cpsidp'].astype(str)
        df['date'] = pd.to_datetime(df['date'])

        # Apply overall start/end date filters first
        if config.PREPROCESS_START_DATE:
            start_dt = pd.to_datetime(config.PREPROCESS_START_DATE)
            df = df[df['date'] >= start_dt]
            print(f"Filtered data after overall start date ({config.PREPROCESS_START_DATE}): {df.shape}")
        if config.PREPROCESS_END_DATE:
            end_dt = pd.to_datetime(config.PREPROCESS_END_DATE)
            df = df[df['date'] <= end_dt]
            print(f"Filtered data before overall end date ({config.PREPROCESS_END_DATE}): {df.shape}")

        # --- Time-based Split Information (Dates Only) ---
        # Get HPT interval dates and train_end_date_preprocess for later use
        print("\nProcessing time-based split parameters...")
        if not hasattr(config, 'HPT_VALIDATION_INTERVALS') or not config.HPT_VALIDATION_INTERVALS:
            print("ERROR: config.HPT_VALIDATION_INTERVALS is not defined or empty. Cannot perform time-based split.")
            sys.exit(1)

        # Convert interval strings to datetime and find the earliest start date
        hpt_intervals = []
        earliest_hpt_start_dt = None
        latest_hpt_end_dt = None # Track latest end date
        try:
            for start_str, end_str in config.HPT_VALIDATION_INTERVALS:
                start_dt = pd.to_datetime(start_str)
                end_dt = pd.to_datetime(end_str)
                if start_dt > end_dt:
                    raise ValueError(f"Start date {start_str} is after end date {end_str} in HPT interval.")
                hpt_intervals.append((start_dt, end_dt))
                if earliest_hpt_start_dt is None or start_dt < earliest_hpt_start_dt:
                    earliest_hpt_start_dt = start_dt
                if latest_hpt_end_dt is None or end_dt > latest_hpt_end_dt:
                    latest_hpt_end_dt = end_dt
            print(f"Using HPT validation intervals: {[(s.date(), e.date()) for s, e in hpt_intervals]}")
            print(f"Earliest HPT interval start date: {earliest_hpt_start_dt.date()}")
        except Exception as e:
            print(f"ERROR processing HPT_VALIDATION_INTERVALS: {e}")
            sys.exit(1)

        # Check TRAIN_END_DATE_PREPROCESS relative to HPT intervals
        train_end_dt_preprocess = None
        if config.TRAIN_END_DATE_PREPROCESS:
             train_end_dt_preprocess = pd.to_datetime(config.TRAIN_END_DATE_PREPROCESS)
             print(f"Using TRAIN_END_DATE_PREPROCESS: {train_end_dt_preprocess.date()} to separate test data.")
             if latest_hpt_end_dt and train_end_dt_preprocess <= latest_hpt_end_dt:
                  print(f"Warning: TRAIN_END_DATE_PREPROCESS ({train_end_dt_preprocess.date()}) is on or before the latest HPT interval end date ({latest_hpt_end_dt.date()}). This might lead to unexpected test set composition.")
        else:
             print("Warning: TRAIN_END_DATE_PREPROCESS not set in config. Test set will be empty.")

        # --- Apply Cleaning and Feature Engineering to Entire DataFrame ---
        def apply_cleaning(input_df):
            """
            Applies minimal cleaning/type conversion assuming input CSV is pre-cleaned by R script.
            Focuses on creating target_state, ensuring correct dtypes for pipeline,
            and calculating time differences.
            """
            temp_df = input_df.copy()

            # 1. Create numerical target state from the pre-cleaned emp_state_f1
            # Ensure emp_state_f1 exists
            if 'emp_state_f1' not in temp_df.columns:
                raise ValueError("Input data missing required 'emp_state_f1' column from R script.")
            target_map = {"Employed": 0, "Unemployed": 1, "Not in Labor Force": 2} # Canonical names
            temp_df['target_state'] = temp_df['emp_state_f1'].map(target_map).astype('Int32') # Use pandas nullable integer

            # 2. Create current_state categorical from pre-cleaned emp_state
            if 'emp_state' not in temp_df.columns:
                raise ValueError("Input data missing required 'emp_state' column from R script.")

            # Explicitly map raw emp_state values to the canonical names from target_map keys
            # This handles potential capitalization inconsistencies in the raw data.
            # Create a reverse map for easier lookup (value -> canonical key)
            # Handle potential variations in raw data (e.g., lowercase)
            raw_to_canonical_map = {
                'employed': 'Employed',
                'unemployed': 'Unemployed',
                'not in labor force': 'Not in Labor Force',
                'nilf': 'Not in Labor Force', # Handle abbreviation if present
                # Add the canonical names themselves in case they are already correct
                'Employed': 'Employed',
                'Unemployed': 'Unemployed',
                'Not in Labor Force': 'Not in Labor Force'
            }
            # Apply the mapping, converting raw state to lowercase first for robustness
            temp_df['current_state_canonical'] = temp_df['emp_state'].str.lower().map(raw_to_canonical_map)
            # Check for any values that didn't map (shouldn't happen if map is comprehensive)
            if temp_df['current_state_canonical'].isnull().any():
                unmapped_values = temp_df.loc[temp_df['current_state_canonical'].isnull(), 'emp_state'].unique()
                print(f"Warning: Unmapped emp_state values found: {unmapped_values}. Setting to 'Not in Labor Force'.")
                temp_df['current_state_canonical'].fillna('Not in Labor Force', inplace=True)

            # Ensure the final column is categorical with the exact categories from target_map.keys()
            temp_df['current_state'] = pd.Categorical(temp_df['current_state_canonical'], categories=target_map.keys())
            temp_df = temp_df.drop(columns=['current_state_canonical'], errors='ignore') # Drop intermediate column

            # 3. Ensure numeric types for relevant columns (might be redundant if R saved correctly, but safe)
            numeric_cols = ['age', 'durunemp', 'famsize', 'nchild', 'wtfinl',
                            'mth_dim1', 'mth_dim2', # Already derived in R
                            'national_unemp_rate', 'national_emp_rate',
                            'state_unemp_rate', 'state_emp_rate',
                            'ind_group_unemp_rate', 'ind_group_emp_rate']
            for col in numeric_cols:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                else:
                    print(f"Warning: Expected numeric column '{col}' not found in input data.")

            # Handle potential NaNs introduced by coercion if necessary (e.g., fill wtfinl)
            if 'wtfinl' in temp_df.columns:
                 temp_df['wtfinl'] = temp_df['wtfinl'].fillna(0)
            # Ensure durunemp is integer after potential float conversion from NaN
            if 'durunemp' in temp_df.columns:
                 temp_df['durunemp'] = temp_df['durunemp'].fillna(0).astype(int)


            # 4. Ensure categorical types for features expected by the pipeline
            # These columns should already exist with the '_cat' suffix from the R script
            categorical_cols = [
                'sex', # Assuming R script outputs 'sex' as 1/2 or Male/Female string
                'race_cat', 'hispan_cat', 'educ_cat', 'statefip',
                'relate_cat', 'metro_cat', 'cbsasz_cat', 'marst_cat', 'citizen_cat',
                'nativity_cat', 'vetstat_cat', 'diff_cat', 'profcert_cat', 'classwkr_cat', # Use diff_cat from R
                'ind_group_cat', 'occ_group_cat' # Use grouped versions from R
            ]
             # Adjust 'sex' handling based on R output format
            if 'sex' in temp_df.columns:
                 # If R outputs 1/2, map it here. If it outputs "Male"/"Female", ensure categorical.
                 if temp_df['sex'].dtype == 'object' or pd.api.types.is_categorical_dtype(temp_df['sex']):
                      temp_df['sex'] = pd.Categorical(temp_df['sex']) # Ensure categorical if string
                 elif pd.api.types.is_numeric_dtype(temp_df['sex']):
                      sex_map = {1: "Male", 2: "Female"}
                      temp_df['sex'] = pd.Categorical(temp_df['sex'].map(sex_map))
                 else:
                      print(f"Warning: Unexpected dtype for 'sex' column: {temp_df['sex'].dtype}")


            for col in categorical_cols:
                if col in temp_df.columns:
                    # Convert object columns or ensure existing categories are recognized
                    if temp_df[col].dtype == 'object':
                         temp_df[col] = pd.Categorical(temp_df[col].fillna("Unknown")) # Fill NA before making categorical
                    elif pd.api.types.is_categorical_dtype(temp_df[col]):
                         # If already categorical, ensure 'Unknown' is a category if NAs existed
                         if temp_df[col].isnull().any():
                              if "Unknown" not in temp_df[col].cat.categories:
                                   temp_df[col] = temp_df[col].cat.add_categories("Unknown")
                              temp_df[col] = temp_df[col].fillna("Unknown")
                    else:
                         # Handle numeric codes if R script didn't convert them (should not happen ideally)
                         print(f"Warning: Column '{col}' expected to be categorical/object, found {temp_df[col].dtype}. Converting.")
                         temp_df[col] = pd.Categorical(temp_df[col].astype(str).fillna("Unknown"))
                else:
                    # Only print warning if it's a core category expected from R
                    core_cats = ['race_cat', 'hispan_cat', 'educ_cat', 'statefip', 'ind_group_cat', 'occ_group_cat']
                    if col in core_cats:
                        print(f"Warning: Expected categorical column '{col}' not found in input data.")


            # 5. Calculate time difference (months_since_last) - This needs to be done here
            temp_df = temp_df.sort_values(['cpsidp', 'date'])
            temp_df['time_diff_days'] = temp_df.groupby('cpsidp')['date'].diff().dt.days
            temp_df['months_since_last'] = (temp_df['time_diff_days'] / 30.4375).round()
            temp_df['months_since_last'] = temp_df['months_since_last'].fillna(0).astype(int)
            temp_df = temp_df.drop(columns=['time_diff_days'], errors='ignore') # Clean up intermediate column

            # 6. Define inverse map for metadata
            target_map_inverse = {v: k for k, v in target_map.items()}

            # Remove original state columns if they won't be used directly as predictors
            # temp_df = temp_df.drop(columns=['emp_state', 'emp_state_f1'], errors='ignore')

            return temp_df, target_map, target_map_inverse # Return both maps

        print("Applying cleaning steps to the entire dataframe...")
        df_clean, target_map, target_map_inverse = apply_cleaning(df)
        print(f"Data shape after cleaning: {df_clean.shape}")
        del df # Free memory
        gc.collect()

    except Exception as e:
        print(f"ERROR during data cleaning: {e}")
        raise

    # --- 2. Define Columns ---
    print("\n===== STEP 2: Defining Columns =====")
    # Update predictor_cols list to use the variable names expected from the R script's output
    # These should align with the columns processed/checked in the simplified apply_cleaning
    predictor_cols = [
        'current_state', # Derived in Python from emp_state
        'age', 'sex', # 'sex' is now categorical string "Male"/"Female" or handled above
        'race_cat', 'hispan_cat', 'educ_cat', 'statefip', # Categorical from R
        'durunemp', # Numeric from R
        'ind_group_cat', 'occ_group_cat', # Categorical groups from R
        'mth_dim1', 'mth_dim2', # Numeric cyclical month from R
        'months_since_last', # Derived in Python
        'national_unemp_rate', 'national_emp_rate', # Numeric rates from R
        'state_unemp_rate', 'state_emp_rate', # Numeric rates from R
        'ind_group_unemp_rate', 'ind_group_emp_rate', # Numeric rates from R
        # Categorical variables from R
        'relate_cat', 'metro_cat', 'cbsasz_cat', 'marst_cat', 'citizen_cat',
        'nativity_cat', 'vetstat_cat',
        'famsize', 'nchild', # Numeric from R
        'diff_cat', # Use diff_cat from R
        'profcert_cat', 'classwkr_cat' # Categorical from R
    ]

    # Original ID columns kept for potential analysis/debugging (should exist in input)
    # These are the *cleaned* categorical versions used for merging rates in R
    original_id_cols = ['statefip', 'ind_group_cat', 'occ_group_cat']
    # Base ID columns needed
    base_id_cols = ['cpsidp', 'date', 'target_state', 'wtfinl'] # target_state created in Python

    # Ensure all required columns exist after cleaning
    all_needed_cols = list(dict.fromkeys(base_id_cols + predictor_cols + original_id_cols))
    missing_cols = [col for col in all_needed_cols if col not in df_clean.columns]
    if missing_cols:
         raise ValueError(f"Missing required columns in cleaned data: {missing_cols}")

    print(f"Predictor columns for pipeline: {predictor_cols}")
    print(f"Original ID columns kept: {original_id_cols}")

    # --- 3. Handle Sparse Categories (on entire cleaned data) ---
    print("\n===== STEP 3: Grouping Sparse Categorical Features (on Entire Cleaned Data) =====")
    potential_categorical_features = df_clean[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Identified potential categorical features for sparsity check: {potential_categorical_features}")
    sparsity_threshold = getattr(config, 'SPARSITY_THRESHOLD', 0.01)
    print(f"Using sparsity threshold: {sparsity_threshold}")

    if sparsity_threshold > 0 and not df_clean.empty:
        for col in potential_categorical_features:
            frequencies = df_clean[col].value_counts(normalize=True)
            sparse_cats = frequencies[frequencies < sparsity_threshold].index.tolist()
            if sparse_cats:
                print(f"  Grouping sparse categories in '{col}': {sparse_cats}")
                other_category = "_OTHER_"
                # Apply grouping directly to df_clean
                if pd.api.types.is_categorical_dtype(df_clean[col]):
                    if other_category not in df_clean[col].cat.categories:
                        df_clean[col] = df_clean[col].cat.add_categories([other_category])
                    df_clean[col] = df_clean[col].replace(sparse_cats, other_category)
                else: # Object type
                    df_clean[col] = df_clean[col].replace(sparse_cats, other_category)
                    all_cats = df_clean[col].unique().tolist()
                    df_clean[col] = pd.Categorical(df_clean[col], categories=all_cats)
    else:
        print("Sparsity threshold is 0 or data is empty, skipping grouping.")

    # --- 4. Define and Fit Preprocessing Pipeline (on entire cleaned data) ---
    print("\n===== STEP 4: Defining and Fitting Preprocessing Pipeline (on Entire Cleaned Data) =====")
    numeric_features = df_clean[predictor_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_clean[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Define transformers and pipeline
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    full_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('variance_threshold', VarianceThreshold(threshold=0))])

    # Fit the pipeline ONLY on the predictor columns of the entire cleaned data
    print("Fitting pipeline on all cleaned data...")
    try:
        if 'target_state' not in df_clean.columns or df_clean['target_state'].isnull().all():
             raise ValueError("Target state column is missing or all NaN in the cleaned data.")
        # Use predictor columns and target from df_clean
        full_pipeline.fit(df_clean[predictor_cols], df_clean['target_state'])
    except Exception as e:
        print(f"ERROR fitting pipeline: {e}")
        raise
    print("Pipeline fitted.")

    # Save the fitted pipeline
    try:
        with open(preprocessor_file, 'wb') as f: pickle.dump(full_pipeline, f)
        print(f"Preprocessing pipeline saved to: {preprocessor_file}")
    except Exception as e: print(f"ERROR saving preprocessor: {e}")

    # --- 5. Apply Pipeline to Create Full Baked DataFrame ---
    print("\n===== STEP 5: Applying Pipeline to Create Full Baked DataFrame =====")
    print("Applying pipeline to all cleaned data...")
    try:
        X_baked_np = full_pipeline.transform(df_clean[predictor_cols])
    except Exception as e:
        print(f"ERROR applying pipeline: {e}")
        raise

    # Get feature names after transformation
    try:
        col_transformer = full_pipeline.named_steps['preprocess']
        final_feature_names = col_transformer.get_feature_names_out()
        vt_step = full_pipeline.named_steps['variance_threshold']
        final_feature_indices = vt_step.get_support(indices=True)
        final_feature_names_after_vt = final_feature_names[final_feature_indices]
        print(f"Number of features after preprocessing: {len(final_feature_names_after_vt)}")
    except Exception as e:
        print(f"Warning: Could not retrieve feature names from pipeline: {e}")
        final_feature_names_after_vt = [f"feature_{i}" for i in range(X_baked_np.shape[1])]

    # Create the single, fully baked DataFrame
    full_baked_df = pd.DataFrame(X_baked_np, columns=final_feature_names_after_vt, index=df_clean.index)

    # Add back base IDs and original IDs from df_clean
    id_cols_to_add = list(dict.fromkeys(base_id_cols + original_id_cols)) # Ensure unique
    full_baked_df = pd.concat([df_clean[id_cols_to_add], full_baked_df], axis=1)

    print(f"Full baked data shape: {full_baked_df.shape}")

    # Clean up intermediate data
    del df_clean, X_baked_np
    gc.collect()

    # --- 6. Time-Based Splitting of full_baked_df ---
    print("\n===== STEP 6: Splitting Baked Data Based on Time =====")
    hpt_interval_data_baked = pd.DataFrame(columns=full_baked_df.columns)
    test_data_baked = pd.DataFrame(columns=full_baked_df.columns)
    train_val_pool_baked = pd.DataFrame(columns=full_baked_df.columns)

    # Ensure date column is datetime
    full_baked_df['date'] = pd.to_datetime(full_baked_df['date'])

    # --- Extract HPT Interval Data (including lookback) ---
    print("Extracting HPT interval data...")
    hpt_extended_masks_baked = []
    lookback_months = getattr(config, 'SEQUENCE_LENGTH', 24) # Use sequence length for lookback
    for start_dt, end_dt in hpt_intervals:
        # Define the actual interval period
        interval_mask = (full_baked_df['date'] >= start_dt) & (full_baked_df['date'] <= end_dt)
        # Find individuals present *within* this interval
        individuals_this_interval = full_baked_df.loc[interval_mask, 'cpsidp'].unique()
        # Define the lookback period for these individuals
        lookback_start_dt = start_dt - pd.DateOffset(months=lookback_months)
        lookback_mask = (
            (full_baked_df['date'] >= lookback_start_dt) &
            (full_baked_df['date'] < start_dt) &
            (full_baked_df['cpsidp'].isin(individuals_this_interval))
        )
        # Combine interval observations and their lookback
        extended_interval_mask = interval_mask | lookback_mask
        hpt_extended_masks_baked.append(extended_interval_mask)

    # Combine masks for all HPT intervals
    combined_hpt_interval_mask_baked = pd.concat(hpt_extended_masks_baked, axis=1).any(axis=1) if hpt_extended_masks_baked else pd.Series([False]*len(full_baked_df), index=full_baked_df.index)
    hpt_interval_data_baked = full_baked_df[combined_hpt_interval_mask_baked].copy()
    print(f"HPT interval data baked shape (including lookback): {hpt_interval_data_baked.shape}") # Clarified shape description

    # --- Extract Test Data (including lookback) ---
    print("Extracting Test data...")
    test_set_indices = pd.Index([]) # Initialize empty index for test set rows
    if train_end_dt_preprocess:
        test_start_dt = train_end_dt_preprocess + pd.Timedelta(days=1)
        print(f"Test period starts on/after: {test_start_dt.date()}")

        # Identify ALL individuals with ANY observation within the test period
        test_period_observations_mask = (full_baked_df['date'] >= test_start_dt)
        test_period_individuals = full_baked_df.loc[test_period_observations_mask, 'cpsidp'].unique()
        print(f"Found {len(test_period_individuals)} unique individuals with observations on/after {test_start_dt.date()}.")

        if len(test_period_individuals) > 0:
            # Re-select the actual test observations for these specific individuals
            test_observations_mask_final = (full_baked_df['cpsidp'].isin(test_period_individuals)) & \
                                           (full_baked_df['date'] >= test_start_dt)
            test_observations = full_baked_df[test_observations_mask_final]

            # Extract lookback history for these individuals from full_baked_df
            lookback_months_test = getattr(config, 'SEQUENCE_LENGTH', 24) # Use sequence length for lookback
            lookback_start_dt_test = test_start_dt - pd.DateOffset(months=lookback_months_test)

            lookback_history_mask = (full_baked_df['cpsidp'].isin(test_period_individuals)) & \
                                    (full_baked_df['date'] < test_start_dt) & \
                                    (full_baked_df['date'] >= lookback_start_dt_test) # Ensure lookback doesn't go too far back
            lookback_history = full_baked_df[lookback_history_mask]

            # Combine lookback and actual test observations
            test_data_baked = pd.concat([lookback_history, test_observations], ignore_index=True).sort_values(['cpsidp', 'date'])
            print(f"Test data baked shape (including lookback): {test_data_baked.shape}")

            # Identify rows belonging to the test set (observations + lookback) in the original index
            test_set_indices = full_baked_df.index[test_observations_mask_final | lookback_history_mask]
        else:
            print("No individuals found in the test period. Test set is empty.")
            # test_set_indices remains empty
    else:
        print("TRAIN_END_DATE_PREPROCESS not set. Test set is empty.")
        # test_set_indices remains empty

    # --- Create Train/Val Pool ---
    print("Creating Train/Val pool...")
    # Exclude HPT interval rows and Test set rows (including lookback)
    hpt_interval_indices = full_baked_df.index[combined_hpt_interval_mask_baked]
    # Ensure test_set_indices is valid before union
    exclude_indices = hpt_interval_indices.union(test_set_indices if not test_set_indices.empty else pd.Index([]))
    train_val_pool_baked = full_baked_df.drop(exclude_indices).copy()
    print(f"Train/Val pool baked shape: {train_val_pool_baked.shape}")

    # Verification (Optional but recommended)
    # Check for overlap (should be empty)
    pool_excluded_overlap = train_val_pool_baked.index.intersection(exclude_indices)
    if not pool_excluded_overlap.empty:
         print(f"WARNING: Overlap detected between train/val pool and excluded indices! Count: {len(pool_excluded_overlap)}")
    # Check row counts (sum may not perfectly match due to overlapping lookback periods)
    print(f"Row counts - HPT Interval: {len(hpt_interval_data_baked)}, Test: {len(test_data_baked)}, Train/Val Pool: {len(train_val_pool_baked)}")
    print(f"Total rows in full_baked_df before splitting: {len(full_baked_df.index)}") # Use original index count

    # Clean up full_baked_df
    del full_baked_df
    gc.collect()

    # --- 7. Create FULL Train/Validation Splits ---
    print("\n===== STEP 7: Creating FULL Train/Validation Splits =====")
    all_person_ids_full_pool = train_val_pool_baked['cpsidp'].unique()
    n_all_persons_full_pool = len(all_person_ids_full_pool)
    sampled_ids_full = all_person_ids_full_pool # Default to all

    if config.PREPROCESS_NUM_INDIVIDUALS_FULL is not None and config.PREPROCESS_NUM_INDIVIDUALS_FULL > 0:
        if config.PREPROCESS_NUM_INDIVIDUALS_FULL >= n_all_persons_full_pool:
            print(f"Requested {config.PREPROCESS_NUM_INDIVIDUALS_FULL} individuals for FULL splits, have {n_all_persons_full_pool}. Using all.")
        else:
            print(f"Sampling {config.PREPROCESS_NUM_INDIVIDUALS_FULL} individuals from {n_all_persons_full_pool} for FULL splits...")
            sampled_ids_full = np.random.choice(all_person_ids_full_pool, config.PREPROCESS_NUM_INDIVIDUALS_FULL, replace=False)
            print(f"Selected {len(sampled_ids_full)} individuals.")
    else:
        print("PREPROCESS_NUM_INDIVIDUALS_FULL not set or <= 0. Using all individuals from Train/Val Pool for FULL splits.")

    n_sampled_persons_full = len(sampled_ids_full)

    # Split sampled IDs into Train/Val
    full_train_ids, full_val_ids = np.array([]), np.array([])
    if n_sampled_persons_full > 0:
        train_size_full = int(config.TRAIN_SPLIT * n_sampled_persons_full)
        # Ensure validation set is not empty if test split is 0
        val_size_full = max(1, int(config.VAL_SPLIT * n_sampled_persons_full)) if (config.TRAIN_SPLIT + config.VAL_SPLIT) < 1 else (n_sampled_persons_full - train_size_full)
        val_size_full = min(val_size_full, n_sampled_persons_full - train_size_full) # Ensure val_size doesn't exceed remaining

        if train_size_full + val_size_full > n_sampled_persons_full:
             print(f"Warning: Full Train ({train_size_full}) + Val ({val_size_full}) > Total ({n_sampled_persons_full}). Adjusting Val.")
             val_size_full = n_sampled_persons_full - train_size_full

        full_train_ids, full_val_ids = train_test_split(sampled_ids_full, train_size=train_size_full, test_size=val_size_full, random_state=config.RANDOM_SEED)
        print(f"FULL Individuals Split - Train: {len(full_train_ids)}, Validation: {len(full_val_ids)}")
    else:
        print("No individuals available for FULL splits.")

    # Filter data
    full_train_data_baked = train_val_pool_baked[train_val_pool_baked['cpsidp'].isin(full_train_ids)].copy() if len(full_train_ids) > 0 else pd.DataFrame(columns=train_val_pool_baked.columns)
    full_val_data_baked = train_val_pool_baked[train_val_pool_baked['cpsidp'].isin(full_val_ids)].copy() if len(full_val_ids) > 0 else pd.DataFrame(columns=train_val_pool_baked.columns)

    print(f"Shape of final FULL train data: {full_train_data_baked.shape}")
    print(f"Shape of final FULL val data: {full_val_data_baked.shape}")

    # Save FULL splits
    print("\nSaving FULL Train/Validation Splits...")
    try:
        full_train_data_baked.to_parquet(full_train_file, index=False)
        full_val_data_baked.to_parquet(full_val_file, index=False)
        print(f" - FULL Train: {full_train_file}")
        print(f" - FULL Validation: {full_val_file}")
    except Exception as e:
        print(f"ERROR saving FULL baked data splits: {e}")

    # --- 8. Create HPT Train/Validation Splits ---
    print("\n===== STEP 8: Creating HPT Train/Validation Splits =====")
    # Use the same pool as the FULL splits (data before TRAIN_END_DATE_PREPROCESS, excluding HPT intervals)
    # Apply HPT-specific sampling (PREPROCESS_NUM_INDIVIDUALS_HPT) to this pool.
    hpt_train_val_pool_baked = train_val_pool_baked # Use the full pool defined in Step 7
    print(f"Pool for HPT splits (data before {train_end_dt_preprocess.date() if train_end_dt_preprocess else 'end of data'}, excluding HPT intervals): {hpt_train_val_pool_baked.shape}") # Updated print statement

    # Use the *same* pool of individuals as the FULL splits pool
    all_person_ids_hpt_pool = all_person_ids_full_pool # IDs from train_val_pool_baked
    n_all_persons_hpt_pool = len(all_person_ids_hpt_pool)
    sampled_ids_hpt = all_person_ids_hpt_pool # Default to all

    if config.PREPROCESS_NUM_INDIVIDUALS_HPT is not None and config.PREPROCESS_NUM_INDIVIDUALS_HPT > 0:
        if config.PREPROCESS_NUM_INDIVIDUALS_HPT >= n_all_persons_hpt_pool:
            print(f"Requested {config.PREPROCESS_NUM_INDIVIDUALS_HPT} individuals for HPT splits, have {n_all_persons_hpt_pool}. Using all.")
        else:
            print(f"Sampling {config.PREPROCESS_NUM_INDIVIDUALS_HPT} individuals from {n_all_persons_hpt_pool} available in the pool for HPT splits...") # Clarified print
            sampled_ids_hpt = np.random.choice(all_person_ids_hpt_pool, config.PREPROCESS_NUM_INDIVIDUALS_HPT, replace=False)
            print(f"Selected {len(sampled_ids_hpt)} individuals.")
    else:
        print("PREPROCESS_NUM_INDIVIDUALS_HPT not set or <= 0. Using all individuals from the pool for HPT splits.")

    n_sampled_persons_hpt = len(sampled_ids_hpt)

    # Split sampled IDs into Train/Val
    hpt_train_ids, hpt_val_ids = np.array([]), np.array([])
    if n_sampled_persons_hpt > 0:
        train_size_hpt = int(config.TRAIN_SPLIT * n_sampled_persons_hpt)
        val_size_hpt = max(1, int(config.VAL_SPLIT * n_sampled_persons_hpt)) if (config.TRAIN_SPLIT + config.VAL_SPLIT) < 1 else (n_sampled_persons_hpt - train_size_hpt)
        val_size_hpt = min(val_size_hpt, n_sampled_persons_hpt - train_size_hpt)

        if train_size_hpt + val_size_hpt > n_sampled_persons_hpt:
             print(f"Warning: HPT Train ({train_size_hpt}) + Val ({val_size_hpt}) > Total ({n_sampled_persons_hpt}). Adjusting Val.")
             val_size_hpt = n_sampled_persons_hpt - train_size_hpt

        hpt_train_ids, hpt_val_ids = train_test_split(sampled_ids_hpt, train_size=train_size_hpt, test_size=val_size_hpt, random_state=config.RANDOM_SEED)
        print(f"HPT Individuals Split - Train: {len(hpt_train_ids)}, Validation: {len(hpt_val_ids)}")
    else:
        print("No individuals available for HPT splits.")


    # Filter data from the hpt_train_val_pool_baked (which is = train_val_pool_baked)
    hpt_train_data_baked = hpt_train_val_pool_baked[hpt_train_val_pool_baked['cpsidp'].isin(hpt_train_ids)].copy() if len(hpt_train_ids) > 0 else pd.DataFrame(columns=hpt_train_val_pool_baked.columns)
    hpt_val_data_baked = hpt_train_val_pool_baked[hpt_train_val_pool_baked['cpsidp'].isin(hpt_val_ids)].copy() if len(hpt_val_ids) > 0 else pd.DataFrame(columns=hpt_train_val_pool_baked.columns)

    print(f"Shape of final HPT train data: {hpt_train_data_baked.shape}")
    print(f"Shape of final HPT val data: {hpt_val_data_baked.shape}")

    # Save HPT splits
    print("\nSaving HPT Train/Validation Splits...")
    try:
        hpt_train_data_baked.to_parquet(hpt_train_file, index=False)
        hpt_val_data_baked.to_parquet(hpt_val_file, index=False)
        print(f" - HPT Train: {hpt_train_file}")
        print(f" - HPT Validation: {hpt_val_file}")
    except Exception as e:
        print(f"ERROR saving HPT baked data splits: {e}")

    # Clean up pool
    del train_val_pool_baked
    gc.collect()

    # --- 9. Save Splits ---
    print("\n===== STEP 9: Saving Final Data Splits =====")
    # Save HPT Interval Data
    if not hpt_interval_data_baked.empty:
        try:
            hpt_interval_data_baked.to_parquet(hpt_interval_data_file, index=False)
            print(f" - HPT Interval Data: {hpt_interval_data_file}")
        except Exception as e: print(f"ERROR saving HPT interval data: {e}")
    else: print(" - HPT Interval Data: Empty, not saved.")

    # Save Test Data
    if not test_data_baked.empty:
        try:
            test_data_baked.to_parquet(test_file, index=False)
            print(f" - Test Data: {test_file}")
        except Exception as e: print(f"ERROR saving test data: {e}")
    else: print(" - Test Data: Empty, not saved.")

    # Save FULL Splits
    try:
        if not full_train_data_baked.empty:
            full_train_data_baked.to_parquet(full_train_file, index=False)
            print(f" - FULL Train: {full_train_file}")
        else: print(" - FULL Train: Empty, not saved.")
        if not full_val_data_baked.empty:
            full_val_data_baked.to_parquet(full_val_file, index=False)
            print(f" - FULL Validation: {full_val_file}")
        else: print(" - FULL Validation: Empty, not saved.")
    except Exception as e: print(f"ERROR saving FULL splits: {e}")

    # Save HPT Splits
    try:
        if not hpt_train_data_baked.empty:
            hpt_train_data_baked.to_parquet(hpt_train_file, index=False)
            print(f" - HPT Train: {hpt_train_file}")
        else: print(" - HPT Train: Empty, not saved.")
        if not hpt_val_data_baked.empty:
            hpt_val_data_baked.to_parquet(hpt_val_file, index=False)
            print(f" - HPT Validation: {hpt_val_file}")
        else: print(" - HPT Validation: Empty, not saved.")
    except Exception as e: print(f"ERROR saving HPT splits: {e}")

    # --- 10. Save Metadata ---
    print("\n===== STEP 10: Saving Metadata =====")
    n_features = len(final_feature_names_after_vt)
    n_classes = len(target_map)

    # Add a check that the actual values in target_state match the map keys (check on a non-empty split if possible)
    check_df_for_targets = full_train_data_baked if not full_train_data_baked.empty else hpt_train_data_baked
    if not check_df_for_targets.empty:
        actual_targets = check_df_for_targets['target_state'].dropna().unique()
        expected_targets = target_map.values()
        if not set(actual_targets).issubset(set(expected_targets)):
             print(f"WARNING: Found target_state values {actual_targets} not fully contained in expected map values {expected_targets}.")
    if n_classes != 3:
        print(f"ERROR: Expected 3 classes based on target_map, but map has {n_classes} entries.")
        print("STOPPING: Incorrect number of target classes determined from map.")
        return # Stop execution
    else:
        print("Confirmed n_classes = 3.")

    metadata = {
        'feature_names': final_feature_names_after_vt.tolist(),
        'n_features': n_features,
        'n_classes': n_classes,
        'target_state_map': target_map,
        'target_state_map_inverse': target_map_inverse,
        'original_identifier_columns': original_id_cols,
        'pad_value': config.PAD_VALUE,
        'hpt_validation_intervals': config.HPT_VALIDATION_INTERVALS,
        'train_end_date_preprocess': str(train_end_dt_preprocess.date()) if train_end_dt_preprocess else None,
        # Counts for FULL splits
        'full_train_individuals': len(full_train_ids),
        'full_val_individuals': len(full_val_ids),
        'full_train_rows_baked': len(full_train_data_baked),
        'full_val_rows_baked': len(full_val_data_baked),
        # Counts for HPT splits
        'hpt_train_individuals': len(hpt_train_ids),
        'hpt_val_individuals': len(hpt_val_ids),
        'hpt_train_rows_baked': len(hpt_train_data_baked),
        'hpt_val_rows_baked': len(hpt_val_data_baked),
        # Counts for other splits
        'test_individuals': test_data_baked['cpsidp'].nunique() if not test_data_baked.empty else 0,
        'test_rows_baked': len(test_data_baked),
        'hpt_interval_individuals': hpt_interval_data_baked['cpsidp'].nunique() if not hpt_interval_data_baked.empty else 0,
        'hpt_interval_rows_baked': len(hpt_interval_data_baked),
        # File paths
        'preprocessing_pipeline_path': str(preprocessor_file),
        'full_train_data_path': str(full_train_file),
        'full_val_data_path': str(full_val_file),
        'hpt_train_data_path': str(hpt_train_file),
        'hpt_val_data_path': str(hpt_val_file),
        'test_data_path': str(test_file) if not test_data_baked.empty else None,
        'hpt_interval_data_path': str(hpt_interval_data_file) if not hpt_interval_data_baked.empty else None,
        'weight_column': config.WEIGHT_COL
    }

    try:
        with open(metadata_file, 'wb') as f: pickle.dump(metadata, f)
        print(f"Metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"ERROR saving metadata: {e}")

    print("\n--- Preprocessing Script Finished ---")

if __name__ == "__main__":
    preprocess_data()