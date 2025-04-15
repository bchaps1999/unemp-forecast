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
# Assume config.py is importable
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
        # Specify potentially problematic dtypes during load
        dtypes = {'cpsidp': str}
        cps_df_raw = pd.read_csv(input_data_path, dtype=dtypes)
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
        df['cpsidp'] = df['cpsidp'].astype(str) # Ensure cpsidp is string
        df['date'] = pd.to_datetime(df['date'])

        # Apply overall start/end date filters first
        start_dt = pd.to_datetime(config.PREPROCESS_START_DATE) if config.PREPROCESS_START_DATE else None
        end_dt = pd.to_datetime(config.PREPROCESS_END_DATE) if config.PREPROCESS_END_DATE else None
        if start_dt:
            df = df[df['date'] >= start_dt]
            print(f"Filtered data after overall start date ({config.PREPROCESS_START_DATE}): {df.shape}")
        if end_dt:
            df = df[df['date'] <= end_dt]
            print(f"Filtered data before overall end date ({config.PREPROCESS_END_DATE}): {df.shape}")

        # --- Time-based Split Information (Dates Only) ---
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
                start_dt_interval = pd.to_datetime(start_str)
                end_dt_interval = pd.to_datetime(end_str)
                if start_dt_interval > end_dt_interval:
                    raise ValueError(f"Start date {start_str} is after end date {end_str} in HPT interval.")
                hpt_intervals.append((start_dt_interval, end_dt_interval))
                if earliest_hpt_start_dt is None or start_dt_interval < earliest_hpt_start_dt:
                    earliest_hpt_start_dt = start_dt_interval
                if latest_hpt_end_dt is None or end_dt_interval > latest_hpt_end_dt:
                    latest_hpt_end_dt = end_dt_interval
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
                  print(f"Warning: TRAIN_END_DATE_PREPROCESS ({train_end_dt_preprocess.date()}) <= latest HPT interval end date ({latest_hpt_end_dt.date()}). Test set might overlap HPT intervals.")
        else:
             print("Warning: TRAIN_END_DATE_PREPROCESS not set in config. Test set will be empty.")

        # --- Apply Cleaning and Feature Engineering to Entire DataFrame ---
        def apply_cleaning(input_df):
            """
            Applies cleaning, type conversion, target creation, and time difference calculation.
            Assumes input CSV is pre-cleaned by R script.
            """
            temp_df = input_df.copy()

            # 1. Create numerical target state from emp_state_f1
            if 'emp_state_f1' not in temp_df.columns:
                raise ValueError("Input data missing required 'emp_state_f1' column.")
            target_map = {"Employed": 0, "Unemployed": 1, "Not in Labor Force": 2}
            temp_df['target_state'] = temp_df['emp_state_f1'].map(target_map).astype('Int32')

            # 2. Create current_state categorical from emp_state
            if 'emp_state' not in temp_df.columns:
                raise ValueError("Input data missing required 'emp_state' column.")
            # Map raw emp_state values to canonical names robustly
            raw_to_canonical_map = {
                'employed': 'Employed', 'unemployed': 'Unemployed',
                'not in labor force': 'Not in Labor Force', 'nilf': 'Not in Labor Force',
                'Employed': 'Employed', 'Unemployed': 'Unemployed', 'Not in Labor Force': 'Not in Labor Force'
            }
            temp_df['current_state_canonical'] = temp_df['emp_state'].str.lower().map(raw_to_canonical_map)
            if temp_df['current_state_canonical'].isnull().any():
                unmapped_values = temp_df.loc[temp_df['current_state_canonical'].isnull(), 'emp_state'].unique()
                print(f"Warning: Unmapped emp_state values found: {unmapped_values}. Setting to 'Not in Labor Force'.")
                temp_df['current_state_canonical'].fillna('Not in Labor Force', inplace=True)
            # Ensure the final column is categorical with the exact categories
            temp_df['current_state'] = pd.Categorical(temp_df['current_state_canonical'], categories=target_map.keys())
            temp_df.drop(columns=['current_state_canonical'], inplace=True)

            # 3. Ensure numeric types
            numeric_cols = ['age', 'durunemp', 'famsize', 'nchild', 'wtfinl',
                            'mth_dim1', 'mth_dim2', 'national_unemp_rate', 'national_emp_rate',
                            'state_unemp_rate', 'state_emp_rate', 'ind_group_unemp_rate', 'ind_group_emp_rate']
            for col in numeric_cols:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                else:
                    print(f"Warning: Expected numeric column '{col}' not found.")
            # Handle potential NaNs (example for wtfinl and durunemp)
            if 'wtfinl' in temp_df.columns: temp_df['wtfinl'] = temp_df['wtfinl'].fillna(0)
            if 'durunemp' in temp_df.columns: temp_df['durunemp'] = temp_df['durunemp'].fillna(0).astype(int)

            # 4. Ensure categorical types (using R script outputs with '_cat' suffix)
            categorical_cols = [
                'sex', 'race_cat', 'hispan_cat', 'educ_cat', 'statefip',
                'relate_cat', 'metro_cat', 'cbsasz_cat', 'marst_cat', 'citizen_cat',
                'nativity_cat', 'vetstat_cat', 'diff_cat', 'profcert_cat', 'classwkr_cat',
                'ind_group_cat', 'occ_group_cat'
            ]
            # Handle 'sex' specifically based on potential formats
            if 'sex' in temp_df.columns:
                 if pd.api.types.is_numeric_dtype(temp_df['sex']):
                      sex_map = {1: "Male", 2: "Female"}
                      temp_df['sex'] = temp_df['sex'].map(sex_map)
                 # Ensure categorical regardless of original format (string or mapped numeric)
                 temp_df['sex'] = pd.Categorical(temp_df['sex'].fillna("Unknown"))

            for col in categorical_cols:
                if col == 'sex': continue # Already handled
                if col in temp_df.columns:
                    # Convert object/numeric to categorical, filling NAs first
                    if temp_df[col].isnull().any():
                        temp_df[col] = temp_df[col].fillna("Unknown")
                    # Ensure categorical type
                    if not isinstance(temp_df[col].dtype, pd.CategoricalDtype):
                        temp_df[col] = pd.Categorical(temp_df[col].astype(str))
                    # Ensure 'Unknown' is a category if it was added
                    if "Unknown" in temp_df[col].unique() and "Unknown" not in temp_df[col].cat.categories:
                         temp_df[col] = temp_df[col].cat.add_categories("Unknown")
                else:
                    # Only warn for core categories expected from R
                    core_cats = ['race_cat', 'hispan_cat', 'educ_cat', 'statefip', 'ind_group_cat', 'occ_group_cat']
                    if col in core_cats:
                        print(f"Warning: Expected categorical column '{col}' not found.")

            # 5. Calculate time difference (months_since_last)
            temp_df = temp_df.sort_values(['cpsidp', 'date'])
            # Use .loc for assignment to avoid SettingWithCopyWarning
            temp_df['time_diff_days'] = temp_df.groupby('cpsidp')['date'].diff().dt.days
            temp_df['months_since_last'] = (temp_df['time_diff_days'] / 30.4375).round().fillna(0).astype(int)
            temp_df.drop(columns=['time_diff_days'], inplace=True)

            # 6. Define inverse map for metadata
            target_map_inverse = {v: k for k, v in target_map.items()}

            # Optional: Drop original state columns if not used as predictors
            # temp_df.drop(columns=['emp_state', 'emp_state_f1'], errors='ignore', inplace=True)

            return temp_df, target_map, target_map_inverse

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
    # Predictor columns should align with cleaned data
    predictor_cols = [
        'current_state', 'age', 'sex', 'race_cat', 'hispan_cat', 'educ_cat', 'statefip',
        'durunemp', 'ind_group_cat', 'occ_group_cat', 'mth_dim1', 'mth_dim2',
        'months_since_last', 'national_unemp_rate', 'national_emp_rate',
        'state_unemp_rate', 'state_emp_rate', 'ind_group_unemp_rate', 'ind_group_emp_rate',
        'relate_cat', 'metro_cat', 'cbsasz_cat', 'marst_cat', 'citizen_cat',
        'nativity_cat', 'vetstat_cat', 'famsize', 'nchild', 'diff_cat',
        'profcert_cat', 'classwkr_cat'
    ]
    # Filter predictor_cols to only include columns present in df_clean
    predictor_cols = [col for col in predictor_cols if col in df_clean.columns]

    # Base ID columns needed
    base_id_cols = ['cpsidp', 'date', 'target_state', config.WEIGHT_COL] # Use config for weight col
    # Original ID columns kept for potential analysis/debugging
    original_id_cols = ['statefip', 'ind_group_cat', 'occ_group_cat']
    original_id_cols = [col for col in original_id_cols if col in df_clean.columns] # Ensure they exist

    # Ensure all required columns exist after cleaning
    all_needed_cols = list(dict.fromkeys(base_id_cols + predictor_cols + original_id_cols))
    missing_cols = [col for col in all_needed_cols if col not in df_clean.columns]
    if missing_cols:
         raise ValueError(f"Missing required columns in cleaned data: {missing_cols}")

    print(f"Predictor columns for pipeline: {predictor_cols}")
    print(f"Original ID columns kept: {original_id_cols}")

    # --- 3. Time-Based Separation (Indices First) ---
    print("\n===== STEP 3: Identifying Indices for Time-Based Separation =====")
    # Ensure date column is datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])

    # --- Identify HPT Interval Indices (including lookback) ---
    print("Identifying HPT interval indices...")
    combined_hpt_interval_mask_clean = pd.Series(False, index=df_clean.index)
    lookback_months = getattr(config, 'SEQUENCE_LENGTH', 24) # Use sequence length for lookback

    for start_dt, end_dt in hpt_intervals:
        interval_mask = (df_clean['date'] >= start_dt) & (df_clean['date'] <= end_dt)
        individuals_this_interval = df_clean.loc[interval_mask, 'cpsidp'].unique()
        if len(individuals_this_interval) > 0:
            lookback_start_dt = start_dt - pd.DateOffset(months=lookback_months)
            lookback_mask = (
                (df_clean['date'] >= lookback_start_dt) &
                (df_clean['date'] < start_dt) &
                (df_clean['cpsidp'].isin(individuals_this_interval))
            )
            combined_hpt_interval_mask_clean |= interval_mask | lookback_mask

    hpt_interval_indices = df_clean.index[combined_hpt_interval_mask_clean]
    print(f"Identified {len(hpt_interval_indices)} rows for HPT intervals (including lookback).")

    # --- Identify Test Data Indices (including lookback) ---
    print("Identifying Test data indices...")
    test_set_indices = pd.Index([]) # Initialize empty index for test set rows
    if train_end_dt_preprocess:
        test_start_dt = train_end_dt_preprocess + pd.Timedelta(days=1)
        print(f"Test period starts on/after: {test_start_dt.date()}")

        test_period_observations_mask = (df_clean['date'] >= test_start_dt)
        test_period_individuals = df_clean.loc[test_period_observations_mask, 'cpsidp'].unique()
        print(f"Found {len(test_period_individuals)} unique individuals with observations on/after {test_start_dt.date()}.")

        if len(test_period_individuals) > 0:
            test_observations_mask_final = (df_clean['cpsidp'].isin(test_period_individuals)) & \
                                           (df_clean['date'] >= test_start_dt)
            lookback_months_test = getattr(config, 'SEQUENCE_LENGTH', 24)
            lookback_start_dt_test = test_start_dt - pd.DateOffset(months=lookback_months_test)
            lookback_history_mask = (df_clean['cpsidp'].isin(test_period_individuals)) & \
                                    (df_clean['date'] < test_start_dt) & \
                                    (df_clean['date'] >= lookback_start_dt_test)
            test_data_mask = test_observations_mask_final | lookback_history_mask
            test_set_indices = df_clean.index[test_data_mask] # Store indices
            print(f"Identified {len(test_set_indices)} rows for Test set (including lookback).")
        else:
            print("No individuals found in the test period. Test set is empty.")
    else:
        print("TRAIN_END_DATE_PREPROCESS not set. Test set is empty.")

    # --- Create Train/Val Pool (Cleaned, Not Baked Yet) ---
    print("Creating Train/Val pool (cleaned data)...")
    exclude_indices = hpt_interval_indices.union(test_set_indices) # Union of indices to exclude
    train_val_pool_clean = df_clean.drop(exclude_indices).copy()
    print(f"Train/Val pool clean shape: {train_val_pool_clean.shape}")

    # Extract the actual data subsets for HPT and Test using the identified indices
    hpt_interval_data_clean = df_clean.loc[hpt_interval_indices].copy()
    test_data_clean = df_clean.loc[test_set_indices].copy().sort_values(['cpsidp', 'date'])

    # Verification (Optional)
    pool_excluded_overlap = train_val_pool_clean.index.intersection(exclude_indices)
    if not pool_excluded_overlap.empty:
         print(f"WARNING: Overlap detected between train/val pool and excluded indices! Count: {len(pool_excluded_overlap)}")
    print(f"Row counts (Clean) - HPT Interval: {len(hpt_interval_data_clean)}, Test: {len(test_data_clean)}, Train/Val Pool: {len(train_val_pool_clean)}")
    print(f"Total rows in df_clean before splitting: {len(df_clean)}")

    # Clean up df_clean
    del df_clean
    gc.collect()

    # --- 4. Handle Sparse Categories (on Train/Val Pool ONLY) ---
    print("\n===== STEP 4: Grouping Sparse Categorical Features (on Train/Val Pool ONLY) =====")
    potential_categorical_features = train_val_pool_clean[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Identified potential categorical features for sparsity check: {potential_categorical_features}")
    sparsity_threshold = getattr(config, 'SPARSITY_THRESHOLD', 0.01)
    print(f"Using sparsity threshold: {sparsity_threshold}")

    if sparsity_threshold > 0 and not train_val_pool_clean.empty:
        other_category = "_OTHER_"
        for col in potential_categorical_features:
            if not isinstance(train_val_pool_clean[col].dtype, pd.CategoricalDtype):
                 train_val_pool_clean[col] = pd.Categorical(train_val_pool_clean[col])

            frequencies = train_val_pool_clean[col].value_counts(normalize=True)
            sparse_cats = frequencies[frequencies < sparsity_threshold].index.tolist()

            if sparse_cats:
                print(f"  Grouping sparse categories in '{col}': {sparse_cats}")
                if other_category not in train_val_pool_clean[col].cat.categories:
                    train_val_pool_clean[col] = train_val_pool_clean[col].cat.add_categories([other_category])
                train_val_pool_clean.loc[train_val_pool_clean[col].isin(sparse_cats), col] = other_category
                train_val_pool_clean[col] = train_val_pool_clean[col].cat.remove_unused_categories()
                # --- Apply the SAME grouping to HPT and Test data ---
                # This ensures consistency in categories across splits
                if col in hpt_interval_data_clean.columns:
                    if not isinstance(hpt_interval_data_clean[col].dtype, pd.CategoricalDtype):
                         hpt_interval_data_clean[col] = pd.Categorical(hpt_interval_data_clean[col])
                    if other_category not in hpt_interval_data_clean[col].cat.categories:
                         hpt_interval_data_clean[col] = hpt_interval_data_clean[col].cat.add_categories([other_category])
                    hpt_interval_data_clean.loc[hpt_interval_data_clean[col].isin(sparse_cats), col] = other_category
                    # Do NOT remove unused categories here, keep all categories seen in train pool
                if col in test_data_clean.columns:
                    if not isinstance(test_data_clean[col].dtype, pd.CategoricalDtype):
                         test_data_clean[col] = pd.Categorical(test_data_clean[col])
                    if other_category not in test_data_clean[col].cat.categories:
                         test_data_clean[col] = test_data_clean[col].cat.add_categories([other_category])
                    test_data_clean.loc[test_data_clean[col].isin(sparse_cats), col] = other_category
                    # Do NOT remove unused categories here
    else:
        print("Sparsity threshold is 0 or train/val pool is empty, skipping grouping.")


    # --- 5. Define and Fit Preprocessing Pipeline (on Train/Val Pool ONLY) ---
    print("\n===== STEP 5: Defining and Fitting Preprocessing Pipeline (on Train/Val Pool ONLY) =====")
    numeric_features = train_val_pool_clean[predictor_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = train_val_pool_clean[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Define transformers and pipeline (same definition as before)
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                  ('cat', categorical_transformer, categorical_features)],
                                     remainder='drop')
    full_pipeline = Pipeline(steps=[('preprocess', preprocessor),
                                    ('variance_threshold', VarianceThreshold(threshold=0))])

    # Fit the pipeline ONLY on the predictor columns of the train/val pool
    print("Fitting pipeline on train/val pool data...")
    try:
        if train_val_pool_clean.empty:
             print("Warning: Train/Val pool is empty. Cannot fit pipeline.")
             # Handle case where pipeline cannot be fitted (e.g., create dummy or raise error)
             # For now, we'll proceed, but applying the unfitted pipeline will fail later.
        elif 'target_state' not in train_val_pool_clean.columns or train_val_pool_clean['target_state'].isnull().all():
             raise ValueError("Target state column is missing or all NaN in the train/val pool.")
        else:
            # Fit on predictors of the pool
            full_pipeline.fit(train_val_pool_clean[predictor_cols], train_val_pool_clean['target_state'])
            print("Pipeline fitted successfully on train/val pool.")
    except Exception as e:
        print(f"ERROR fitting pipeline: {e}")
        raise

    # Save the fitted pipeline
    try:
        with open(preprocessor_file, 'wb') as f: pickle.dump(full_pipeline, f)
        print(f"Preprocessing pipeline saved to: {preprocessor_file}")
    except Exception as e: print(f"ERROR saving preprocessor: {e}")

    # --- 6. Apply Fitted Pipeline to Create Baked DataFrames for All Splits ---
    print("\n===== STEP 6: Applying Pipeline to Create Baked DataFrames for All Splits =====")
    final_feature_names_after_vt = []
    try:
        # Get feature names after transformation (from the fitted pipeline)
        col_transformer = full_pipeline.named_steps['preprocess']
        raw_feature_names = col_transformer.get_feature_names_out()
        vt_step = full_pipeline.named_steps['variance_threshold']
        vt_mask = vt_step.get_support()
        final_feature_names_after_vt = raw_feature_names[vt_mask]
        print(f"Number of features after preprocessing and variance threshold: {len(final_feature_names_after_vt)}")
    except Exception as e:
        print(f"Warning: Could not retrieve feature names from pipeline: {e}")
        # Fallback if names can't be retrieved (shouldn't happen if fitted)
        # Need to know the expected number of columns if pipeline wasn't fitted
        # This part might need adjustment if fitting failed. Assuming it succeeded for now.
        # If fitting failed, X_baked_np shape below would be unknown.
        # Let's assume fitting worked and we have feature names.

    # Helper function to apply pipeline and create baked df
    def apply_pipeline_and_bake(df_clean_split, name):
        print(f"Applying pipeline to {name} data...")
        if df_clean_split.empty:
            print(f"{name} data is empty. Returning empty baked DataFrame.")
            return pd.DataFrame(columns=list(dict.fromkeys(base_id_cols + original_id_cols)) + final_feature_names_after_vt.tolist()) # Ensure correct columns

        try:
            # Apply transform using the *already fitted* pipeline
            X_baked_np = full_pipeline.transform(df_clean_split[predictor_cols])
            # Create baked DataFrame
            baked_df = pd.DataFrame(X_baked_np, columns=final_feature_names_after_vt, index=df_clean_split.index)
            # Add back ID columns
            id_cols_to_add = list(dict.fromkeys(base_id_cols + original_id_cols))
            baked_df = df_clean_split[id_cols_to_add].join(baked_df, how='inner')
            print(f"Baked {name} data shape: {baked_df.shape}")
            return baked_df
        except Exception as e:
            print(f"ERROR applying pipeline to {name} data: {e}")
            # Return empty DataFrame with correct columns on error
            return pd.DataFrame(columns=list(dict.fromkeys(base_id_cols + original_id_cols)) + final_feature_names_after_vt.tolist())

    # Apply to each split
    train_val_pool_baked = apply_pipeline_and_bake(train_val_pool_clean, "Train/Val Pool")
    hpt_interval_data_baked = apply_pipeline_and_bake(hpt_interval_data_clean, "HPT Interval")
    test_data_baked = apply_pipeline_and_bake(test_data_clean, "Test")

    # Clean up intermediate clean splits
    del train_val_pool_clean, hpt_interval_data_clean, test_data_clean
    gc.collect()

    # --- 7. Create FULL Train/Validation Splits (from train_val_pool_baked) ---
    print("\n===== STEP 7: Creating FULL Train/Validation Splits (from Baked Pool) =====")
    # This part remains largely the same, but operates on train_val_pool_baked
    all_person_ids_full_pool = train_val_pool_baked['cpsidp'].unique()
    n_all_persons_full_pool = len(all_person_ids_full_pool)
    sampled_ids_full = all_person_ids_full_pool # Default to all

    # Sample individuals if requested and possible
    num_individuals_full = config.PREPROCESS_NUM_INDIVIDUALS_FULL
    if num_individuals_full is not None and 0 < num_individuals_full < n_all_persons_full_pool:
        print(f"Sampling {num_individuals_full} individuals from {n_all_persons_full_pool} for FULL splits...")
        sampled_ids_full = np.random.choice(all_person_ids_full_pool, num_individuals_full, replace=False)
        print(f"Selected {len(sampled_ids_full)} individuals.")
    elif num_individuals_full is not None and num_individuals_full >= n_all_persons_full_pool:
        print(f"Requested {num_individuals_full} individuals for FULL splits, have {n_all_persons_full_pool}. Using all.")
    else:
        print("PREPROCESS_NUM_INDIVIDUALS_FULL not set or <= 0. Using all individuals from Train/Val Pool for FULL splits.")

    n_sampled_persons_full = len(sampled_ids_full)

    # Split sampled IDs into Train/Val
    full_train_ids, full_val_ids = np.array([]), np.array([])
    if n_sampled_persons_full > 0:
        # Ensure splits sum to <= 1.0
        train_prop = config.TRAIN_SPLIT
        val_prop = config.VAL_SPLIT
        if train_prop + val_prop > 1.0:
            print(f"Warning: TRAIN_SPLIT ({train_prop}) + VAL_SPLIT ({val_prop}) > 1.0. Adjusting VAL_SPLIT.")
            val_prop = max(0, 1.0 - train_prop)

        train_size_full = int(train_prop * n_sampled_persons_full)
        # Calculate val_size based on remaining proportion, ensuring at least 1 if possible
        val_size_full = n_sampled_persons_full - train_size_full if val_prop > 0 else 0
        if val_size_full == 0 and train_size_full < n_sampled_persons_full: # Ensure val gets at least 1 if there are leftovers
             val_size_full = 1
             train_size_full = n_sampled_persons_full - 1 # Adjust train size slightly

        # Use test_size for the validation set size in train_test_split
        if train_size_full > 0 and val_size_full > 0:
            full_train_ids, full_val_ids = train_test_split(sampled_ids_full, train_size=train_size_full, test_size=val_size_full, random_state=config.RANDOM_SEED)
        elif train_size_full > 0: # Only train set
             full_train_ids = sampled_ids_full
        elif val_size_full > 0: # Only val set (unlikely)
             full_val_ids = sampled_ids_full

        print(f"FULL Individuals Split - Train: {len(full_train_ids)}, Validation: {len(full_val_ids)}")
    else:
        print("No individuals available for FULL splits.")

    # Filter data using isin for efficiency
    full_train_data_baked = train_val_pool_baked[train_val_pool_baked['cpsidp'].isin(full_train_ids)].copy() if len(full_train_ids) > 0 else pd.DataFrame(columns=train_val_pool_baked.columns)
    full_val_data_baked = train_val_pool_baked[train_val_pool_baked['cpsidp'].isin(full_val_ids)].copy() if len(full_val_ids) > 0 else pd.DataFrame(columns=train_val_pool_baked.columns)

    print(f"Shape of final FULL train data: {full_train_data_baked.shape}")
    print(f"Shape of final FULL val data: {full_val_data_baked.shape}")


    # --- 8. Create HPT Train/Validation Splits (from train_val_pool_baked) ---
    print("\n===== STEP 8: Creating HPT Train/Validation Splits (from Baked Pool) =====")
    # This part remains largely the same, but operates on train_val_pool_baked
    all_person_ids_hpt_pool = all_person_ids_full_pool # Use same pool as FULL
    n_all_persons_hpt_pool = len(all_person_ids_hpt_pool)
    sampled_ids_hpt = all_person_ids_hpt_pool # Default to all

    # Sample individuals if requested and possible
    num_individuals_hpt = config.PREPROCESS_NUM_INDIVIDUALS_HPT
    if num_individuals_hpt is not None and 0 < num_individuals_hpt < n_all_persons_hpt_pool:
        print(f"Sampling {num_individuals_hpt} individuals from {n_all_persons_hpt_pool} available in the pool for HPT splits...")
        sampled_ids_hpt = np.random.choice(all_person_ids_hpt_pool, num_individuals_hpt, replace=False)
        print(f"Selected {len(sampled_ids_hpt)} individuals.")
    elif num_individuals_hpt is not None and num_individuals_hpt >= n_all_persons_hpt_pool:
        print(f"Requested {num_individuals_hpt} individuals for HPT splits, have {n_all_persons_hpt_pool}. Using all.")
    else:
        print("PREPROCESS_NUM_INDIVIDUALS_HPT not set or <= 0. Using all individuals from the pool for HPT splits.")

    n_sampled_persons_hpt = len(sampled_ids_hpt)

    # Split sampled IDs into Train/Val (using same logic as FULL splits)
    hpt_train_ids, hpt_val_ids = np.array([]), np.array([])
    if n_sampled_persons_hpt > 0:
        train_prop = config.TRAIN_SPLIT
        val_prop = config.VAL_SPLIT
        if train_prop + val_prop > 1.0: val_prop = max(0, 1.0 - train_prop) # Adjust val_prop if needed

        train_size_hpt = int(train_prop * n_sampled_persons_hpt)
        val_size_hpt = n_sampled_persons_hpt - train_size_hpt if val_prop > 0 else 0
        if val_size_hpt == 0 and train_size_hpt < n_sampled_persons_hpt: # Ensure val gets at least 1 if possible
             val_size_hpt = 1
             train_size_hpt = n_sampled_persons_hpt - 1

        if train_size_hpt > 0 and val_size_hpt > 0:
            hpt_train_ids, hpt_val_ids = train_test_split(sampled_ids_hpt, train_size=train_size_hpt, test_size=val_size_hpt, random_state=config.RANDOM_SEED)
        elif train_size_hpt > 0: hpt_train_ids = sampled_ids_hpt
        elif val_size_hpt > 0: hpt_val_ids = sampled_ids_hpt

        print(f"HPT Individuals Split - Train: {len(hpt_train_ids)}, Validation: {len(hpt_val_ids)}")
    else:
        print("No individuals available for HPT splits.")

    # Filter data from the train_val_pool_baked
    hpt_train_data_baked = train_val_pool_baked[train_val_pool_baked['cpsidp'].isin(hpt_train_ids)].copy() if len(hpt_train_ids) > 0 else pd.DataFrame(columns=train_val_pool_baked.columns)
    hpt_val_data_baked = train_val_pool_baked[train_val_pool_baked['cpsidp'].isin(hpt_val_ids)].copy() if len(hpt_val_ids) > 0 else pd.DataFrame(columns=train_val_pool_baked.columns)

    print(f"Shape of final HPT train data: {hpt_train_data_baked.shape}")
    print(f"Shape of final HPT val data: {hpt_val_data_baked.shape}")

    # Clean up pool
    del train_val_pool_baked # Now delete the baked pool
    gc.collect()

    # --- 9. Save Splits ---
    print("\n===== STEP 9: Saving Final Data Splits =====")
    # Helper function to save parquet safely
    def save_split(df, file_path, name):
        if not df.empty:
            try:
                # Specify pyarrow engine for potential speedup
                df.to_parquet(file_path, index=False, engine='pyarrow')
                print(f" - {name}: Saved to {file_path} ({df.shape})")
            except ImportError:
                print(f"Warning: pyarrow not installed. Falling back to default parquet engine for {name}.")
                try:
                    df.to_parquet(file_path, index=False) # Use default engine
                    print(f" - {name}: Saved to {file_path} ({df.shape}) using default engine.")
                except Exception as e: print(f"ERROR saving {name} data with default engine: {e}")
            except Exception as e: print(f"ERROR saving {name} data: {e}")
        else: print(f" - {name}: Empty, not saved.")

    save_split(hpt_interval_data_baked, hpt_interval_data_file, "HPT Interval Data")
    save_split(test_data_baked, test_file, "Test Data")
    save_split(full_train_data_baked, full_train_file, "FULL Train")
    save_split(full_val_data_baked, full_val_file, "FULL Validation")
    save_split(hpt_train_data_baked, hpt_train_file, "HPT Train")
    save_split(hpt_val_data_baked, hpt_val_file, "HPT Validation")

    # --- 10. Save Metadata ---
    print("\n===== STEP 10: Saving Metadata =====")
    n_features = len(final_feature_names_after_vt)
    n_classes = len(target_map)

    # Check target state consistency
    check_df_for_targets = full_train_data_baked if not full_train_data_baked.empty else hpt_train_data_baked
    if not check_df_for_targets.empty:
        actual_targets = check_df_for_targets['target_state'].dropna().unique()
        expected_targets = target_map.values()
        if not set(actual_targets).issubset(set(expected_targets)):
             print(f"WARNING: Found target_state values {actual_targets} not fully contained in expected map values {expected_targets}.")
    if n_classes != 3:
        print(f"ERROR: Expected 3 classes based on target_map, but map has {n_classes} entries. STOPPING.")
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
        # File paths (convert Path objects to strings for serialization)
        'preprocessing_pipeline_path': str(preprocessor_file),
        'full_train_data_path': str(full_train_file) if not full_train_data_baked.empty else None,
        'full_val_data_path': str(full_val_file) if not full_val_data_baked.empty else None,
        'hpt_train_data_path': str(hpt_train_file) if not hpt_train_data_baked.empty else None,
        'hpt_val_data_path': str(hpt_val_file) if not hpt_val_data_baked.empty else None,
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