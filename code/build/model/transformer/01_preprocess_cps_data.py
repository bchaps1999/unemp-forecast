import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from pathlib import Path
import os
import sys

# Import scikit-learn components
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# --- Import Config ---
# Add parent directory to sys.path to allow config import
sys.path.append(str(Path(__file__).parent))
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Make sure it's in the same directory or sys.path is configured correctly.")
    sys.exit(1)

# --- Parameters & Configuration ---

def clean_faminc(series):
    """Helper function to map faminc strings to ordered categories."""
    mapping = {
        "$10,000 - $24,999": 1, "10000-24999": 1,
        "$25,000 - $74,999": 2, "25000-74999": 2,
        "$75,000 - $149,999": 3, "75000-149999": 3,
        "$150,000+": 4, "150000+": 4,
        "<$10,000": 0, "<10000": 0,
    }
    cleaned = series.str.replace(r'[$, ]', '', regex=True).str.strip().map(mapping)
    categories = sorted(list(set(mapping.values())))
    return pd.Categorical(cleaned, categories=categories, ordered=True)

def preprocess_data():
    """Main function to load, clean, preprocess all, save all, sample, split, and save splits."""
    # --- Setup Paths ---
    project_root_path = config.PROJECT_ROOT
    input_data_path = config.PREPROCESS_INPUT_FILE
    output_dir_path = config.PREPROCESS_OUTPUT_DIR
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Output files for sampled splits (used by training)
    train_file = output_dir_path / config.TRAIN_DATA_FILENAME
    val_file = output_dir_path / config.VAL_DATA_FILENAME
    test_file = output_dir_path / config.TEST_DATA_FILENAME
    # Output file for HPT validation data
    hpt_val_file = output_dir_path / config.HPT_VAL_DATA_FILENAME # New path for HPT val data
    # Output file for ALL processed data (potentially used by forecasting)
    full_baked_file = output_dir_path / config.FULL_DATA_FILENAME # New path
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

        # --- Time-based Split for HPT Validation ---
        print("\nSplitting data for HPT validation...")
        if not hasattr(config, 'HPT_VALIDATION_START_DATE') or not config.HPT_VALIDATION_START_DATE:
            print("ERROR: config.HPT_VALIDATION_START_DATE is not defined. Cannot perform time-based split.")
            sys.exit(1)

        hpt_val_start_dt = pd.to_datetime(config.HPT_VALIDATION_START_DATE)
        print(f"Using HPT validation start date: {hpt_val_start_dt.date()}")

        data_for_fitting_raw = df[df['date'] < hpt_val_start_dt].copy()
        hpt_val_data_raw = df[df['date'] >= hpt_val_start_dt].copy()

        print(f"Data for fitting/standard splits (before {hpt_val_start_dt.date()}): {data_for_fitting_raw.shape}")
        print(f"Data for HPT validation (on/after {hpt_val_start_dt.date()}): {hpt_val_data_raw.shape}")

        if data_for_fitting_raw.empty:
            print("ERROR: No data available before HPT_VALIDATION_START_DATE. Cannot proceed.")
            sys.exit(1)
        if hpt_val_data_raw.empty:
            print("WARNING: No data available on/after HPT_VALIDATION_START_DATE for HPT validation set.")
            # Proceed, but HPT evaluation might yield trivial results.

        # --- Continue Cleaning and Feature Engineering (on both splits if needed, or apply functions) ---
        # Apply cleaning/feature engineering steps consistently.
        # It's often easier to do this on the combined dataframe 'df' before splitting,
        # but ensure features like 'months_since_last' are calculated correctly after sorting.
        # Let's re-apply the cleaning steps to the split dataframes to be safe.

        def apply_cleaning(input_df):
            temp_df = input_df.copy()
            target_map = {"employed": 0, "unemployed": 1, "not_in_labor_force": 2}
            temp_df['target_state'] = temp_df['emp_state_f1'].map(target_map).astype('Int32')
            temp_df['current_state'] = pd.Categorical(temp_df['emp_state'], categories=target_map.keys())
            temp_df['age'] = pd.to_numeric(temp_df['age'], errors='coerce')
            temp_df['sex'] = pd.Categorical(temp_df['sex'].map({1: "Male", 2: "Female"}))
            race_map = {
                "100": "White", "200": "Black", "300": "NativeAmerican",
                "651": "Asian", "652": "PacificIslander"
            }
            temp_df['race_cat'] = pd.Categorical(temp_df['race'].astype(str).map(race_map).fillna("Other/Mixed"))
            temp_df['hispan_cat'] = np.select(
                [
                    temp_df['hispan'].astype(str).str.contains("not hispanic", case=False, na=False),
                    temp_df['hispan'].notna() & ~temp_df['hispan'].astype(str).str.contains("not hispanic", case=False, na=False)
                ],
                ["NotHispanic", "Hispanic"], default="Unknown"
            )
            temp_df['hispan_cat'] = pd.Categorical(temp_df['hispan_cat'])
            conditions = [
                temp_df['educ'].astype(str).str.contains("Advanced degree", case=False, na=False),
                temp_df['educ'].astype(str).str.contains("Bachelor's degree", case=False, na=False),
                temp_df['educ'].astype(str).str.contains("Some college", case=False, na=False),
                temp_df['educ'].notna()
            ]
            choices = ["Advanced", "Bachelors", "Some_College", "HS_or_less"]
            temp_df['educ_cat'] = np.select(conditions, choices, default="HS_or_less")
            educ_categories = ["HS_or_less", "Some_College", "Bachelors", "Advanced"]
            temp_df['educ_cat'] = pd.Categorical(temp_df['educ_cat'], categories=educ_categories, ordered=False)
            temp_df['statefip'] = temp_df['statefip'].astype(float).astype(int).astype(str).str.zfill(2)
            temp_df['statefip'] = pd.Categorical(temp_df['statefip'])
            temp_df['faminc_cat'] = clean_faminc(temp_df['faminc'].astype(str))
            temp_df['whyunemp'] = "Not_Unemployed"
            unemployed_mask = temp_df['current_state'] == "unemployed"
            temp_df.loc[unemployed_mask, 'whyunemp'] = temp_df.loc[unemployed_mask, 'whyunemp'].astype(str) # Ensure string type before categorizing
            temp_df['whyunemp'] = pd.Categorical(temp_df['whyunemp'])
            temp_df['ahrsworkt_cat'] = "Not_Employed"
            employed_mask = temp_df['current_state'] == "employed"
            temp_df.loc[employed_mask, 'ahrsworkt_cat'] = temp_df.loc[employed_mask, 'ahrsworkt'].astype(str)
            temp_df.loc[temp_df['ahrsworkt_cat'].isin(['nan', 'NA', '']), 'ahrsworkt_cat'] = "Unknown_Hours"
            temp_df['ahrsworkt_cat'] = temp_df['ahrsworkt_cat'].str.replace("40+", "40plus", regex=False)
            temp_df['ahrsworkt_cat'] = pd.Categorical(temp_df['ahrsworkt_cat'])
            temp_df['durunemp'] = pd.to_numeric(temp_df['durunemp'], errors='coerce')
            temp_df.loc[temp_df['current_state'] != "unemployed", 'durunemp'] = 0
            temp_df['ind_group_cat'] = temp_df['ind_group'].astype(str).replace(['', 'nan'], 'Unknown')
            temp_df['ind_group_cat'] = pd.Categorical(temp_df['ind_group_cat'])
            temp_df['occ_2dig_cat'] = temp_df['occ_2dig'].astype(str).replace(['', 'nan', '99'], 'Unknown')
            occ_mask = temp_df['occ_2dig_cat'] != 'Unknown'
            # Ensure numeric conversion before zfill
            numeric_occ = pd.to_numeric(temp_df.loc[occ_mask, 'occ_2dig_cat'], errors='coerce')
            temp_df.loc[occ_mask & numeric_occ.notna(), 'occ_2dig_cat'] = 'Occ' + numeric_occ[numeric_occ.notna()].astype(int).astype(str).str.zfill(2)
            temp_df['occ_2dig_cat'] = pd.Categorical(temp_df['occ_2dig_cat'])
            temp_df['mth_dim1'] = pd.to_numeric(temp_df['mth_dim1'], errors='coerce')
            temp_df['mth_dim2'] = pd.to_numeric(temp_df['mth_dim2'], errors='coerce')
            temp_df = temp_df.sort_values(['cpsidp', 'date'])
            temp_df['time_diff_days'] = temp_df.groupby('cpsidp')['date'].diff().dt.days
            temp_df['months_since_last'] = (temp_df['time_diff_days'] / 30.4375).round()
            temp_df['months_since_last'] = temp_df['months_since_last'].fillna(0).astype(int)
            # Ensure wtfinl is numeric
            temp_df['wtfinl'] = pd.to_numeric(temp_df['wtfinl'], errors='coerce').fillna(0)
            return temp_df, target_map # Return map as well

        print("Applying cleaning steps to 'data_for_fitting_raw'...")
        data_for_fitting_clean, target_map = apply_cleaning(data_for_fitting_raw)
        target_map_inverse = {v: k for k, v in target_map.items()} # Get inverse map

        print("Applying cleaning steps to 'hpt_val_data_raw'...")
        hpt_val_data_clean, _ = apply_cleaning(hpt_val_data_raw)

    except Exception as e:
        print(f"ERROR during data cleaning or time splitting: {e}")
        raise

    # --- 2. Define Columns & Create Final DF ---
    print("\n===== STEP 2: Defining Columns =====")
    predictor_cols = [
        'current_state', 'age', 'sex', 'race_cat', 'hispan_cat', 'educ_cat', 'statefip', 'faminc_cat',
        'whyunemp', 'ahrsworkt_cat', 'durunemp', 'ind_group_cat', 'occ_2dig_cat',
        'mth_dim1', 'mth_dim2', 'months_since_last',
        'national_unemp_rate', 'national_emp_rate', 'state_unemp_rate', 'state_emp_rate',
        'ind_group_unemp_rate', 'ind_group_emp_rate'
    ]
    original_id_cols = ['statefip', 'ind_group_cat']
    # Add wtfinl to base_id_cols
    base_id_cols = ['cpsidp', 'date', 'target_state', 'wtfinl']

    # Ensure original ID columns and wtfinl are present in both dataframes
    missing_orig_ids_fit = [col for col in original_id_cols if col not in data_for_fitting_clean.columns]
    missing_orig_ids_hpt = [col for col in original_id_cols if col not in hpt_val_data_clean.columns]
    if 'wtfinl' not in data_for_fitting_clean.columns: missing_orig_ids_fit.append('wtfinl')
    if not hpt_val_data_clean.empty and 'wtfinl' not in hpt_val_data_clean.columns: missing_orig_ids_hpt.append('wtfinl')

    if missing_orig_ids_fit or missing_orig_ids_hpt:
        raise ValueError(f"Missing original identifier or weight columns needed: Fit({missing_orig_ids_fit}), HPT({missing_orig_ids_hpt})")

    # Columns to keep before processing
    keep_cols = list(dict.fromkeys(base_id_cols + predictor_cols + original_id_cols))

    # Select columns for both dataframes
    df_final_fit = data_for_fitting_clean[keep_cols].copy()
    df_final_hpt = hpt_val_data_clean[keep_cols].copy() if not hpt_val_data_clean.empty else pd.DataFrame(columns=keep_cols)

    print(f"Data shape for fitting before preprocessing: {df_final_fit.shape}")
    print(f"Data shape for HPT validation before preprocessing: {df_final_hpt.shape}")
    print(f"Predictor columns for pipeline: {predictor_cols}")
    print(f"Original ID columns kept: {original_id_cols}")

    # --- 3. Handle Sparse Categories (on fitting data only) ---
    print("\n===== STEP 3: Grouping Sparse Categorical Features (on Fitting Data) =====")
    potential_categorical_features = df_final_fit[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Identified potential categorical features for sparsity check: {potential_categorical_features}")
    sparsity_threshold = getattr(config, 'SPARSITY_THRESHOLD', 0.01)
    print(f"Using sparsity threshold: {sparsity_threshold}")

    sparse_categories_map = {} # Store sparse categories found in fitting data
    if sparsity_threshold > 0 and not df_final_fit.empty:
        for col in potential_categorical_features:
            frequencies = df_final_fit[col].value_counts(normalize=True)
            sparse_cats = frequencies[frequencies < sparsity_threshold].index.tolist()
            if sparse_cats:
                print(f"  Grouping sparse categories in '{col}': {sparse_cats}")
                other_category = "_OTHER_"
                sparse_categories_map[col] = sparse_cats # Store for applying to HPT data

                # Apply to fitting data
                if pd.api.types.is_categorical_dtype(df_final_fit[col]):
                    if other_category not in df_final_fit[col].cat.categories:
                        df_final_fit[col] = df_final_fit[col].cat.add_categories([other_category])
                    df_final_fit[col] = df_final_fit[col].replace(sparse_cats, other_category)
                else: # Should be categorical, but handle object type just in case
                    df_final_fit[col] = df_final_fit[col].replace(sparse_cats, other_category)
                    # Convert back to categorical if needed, ensuring _OTHER_ is included
                    all_cats = df_final_fit[col].unique().tolist()
                    df_final_fit[col] = pd.Categorical(df_final_fit[col], categories=all_cats)

        # Apply the same grouping to HPT validation data
        print("Applying sparse category grouping to HPT validation data...")
        if not df_final_hpt.empty:
            for col, sparse_cats in sparse_categories_map.items():
                if col in df_final_hpt.columns:
                    other_category = "_OTHER_"
                    # Ensure the category exists before replacing
                    if pd.api.types.is_categorical_dtype(df_final_hpt[col]):
                         if other_category not in df_final_hpt[col].cat.categories:
                              df_final_hpt[col] = df_final_hpt[col].cat.add_categories([other_category])
                         # Replace only categories that exist in this split's column
                         cats_to_replace = [cat for cat in sparse_cats if cat in df_final_hpt[col].cat.categories]
                         if cats_to_replace:
                              df_final_hpt[col] = df_final_hpt[col].replace(cats_to_replace, other_category)
                    else: # Object type
                         df_final_hpt[col] = df_final_hpt[col].replace(sparse_cats, other_category)
                         all_cats_hpt = df_final_hpt[col].unique().tolist()
                         df_final_hpt[col] = pd.Categorical(df_final_hpt[col], categories=all_cats_hpt)

    else:
        print("Skipping sparse category grouping.")

    # --- 4. Define and Fit Preprocessing Pipeline (on fitting data only) ---
    print("\n===== STEP 4: Defining and Fitting Preprocessing Pipeline (on Fitting Data) =====")
    numeric_features = df_final_fit[predictor_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_final_fit[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Define transformers and pipeline
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    full_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('variance_threshold', VarianceThreshold(threshold=0))])

    # Fit the pipeline ONLY on the fitting data's predictor columns
    print("Fitting pipeline on data before HPT validation period...")
    try:
        # Ensure target_state exists and is not all NaN before fitting
        if 'target_state' not in df_final_fit.columns or df_final_fit['target_state'].isnull().all():
             raise ValueError("Target state column is missing or all NaN in the fitting data.")
        full_pipeline.fit(df_final_fit[predictor_cols], df_final_fit['target_state'])
    except Exception as e:
        print(f"ERROR fitting pipeline: {e}")
        raise
    print("Pipeline fitted.")

    # Save the fitted pipeline
    try:
        with open(preprocessor_file, 'wb') as f: pickle.dump(full_pipeline, f)
        print(f"Preprocessing pipeline saved to: {preprocessor_file}")
    except Exception as e: print(f"ERROR saving preprocessor: {e}")

    # --- 5. Apply Pipeline to Both Splits ---
    print("\n===== STEP 5: Applying Pipeline to Fitting Data and HPT Validation Data =====")
    print("Applying pipeline to fitting data...")
    X_fit_baked_np = full_pipeline.transform(df_final_fit[predictor_cols])

    print("Applying pipeline to HPT validation data...")
    X_hpt_baked_np = np.array([]).reshape(0, X_fit_baked_np.shape[1]) # Initialize empty with correct columns
    if not df_final_hpt.empty:
        try:
            X_hpt_baked_np = full_pipeline.transform(df_final_hpt[predictor_cols])
        except Exception as e:
            print(f"ERROR applying pipeline to HPT validation data: {e}")
            # Decide how to handle: stop, or continue with empty HPT set? Let's stop.
            raise

    # Get feature names after transformation (from the fitted pipeline)
    try:
        col_transformer = full_pipeline.named_steps['preprocess']
        final_feature_names = col_transformer.get_feature_names_out()
        vt_step = full_pipeline.named_steps['variance_threshold']
        final_feature_indices = vt_step.get_support(indices=True)
        final_feature_names_after_vt = final_feature_names[final_feature_indices]
        print(f"Number of features after preprocessing: {len(final_feature_names_after_vt)}")
    except Exception as e:
        print(f"Warning: Could not retrieve feature names from pipeline: {e}")
        final_feature_names_after_vt = [f"feature_{i}" for i in range(X_fit_baked_np.shape[1])]

    # Create the baked DataFrames
    fit_baked_df = pd.DataFrame(X_fit_baked_np, columns=final_feature_names_after_vt, index=df_final_fit.index)
    hpt_baked_df = pd.DataFrame(X_hpt_baked_np, columns=final_feature_names_after_vt, index=df_final_hpt.index) if not df_final_hpt.empty else pd.DataFrame(columns=final_feature_names_after_vt)

    # Add back base IDs and original IDs
    id_cols_to_add = base_id_cols + original_id_cols
    fit_baked_df = pd.concat([df_final_fit[id_cols_to_add], fit_baked_df], axis=1)
    if not df_final_hpt.empty:
        hpt_baked_df = pd.concat([df_final_hpt[id_cols_to_add], hpt_baked_df], axis=1)

    print(f"Fitting data baked shape: {fit_baked_df.shape}")
    print(f"HPT validation data baked shape: {hpt_baked_df.shape}")

    # --- 6. Save HPT Validation Baked Data ---
    print("\n===== STEP 6: Saving HPT Validation Baked Data =====")
    if not hpt_baked_df.empty:
        try:
            hpt_baked_df.to_parquet(hpt_val_file, index=False)
            print(f"HPT validation baked data saved to: {hpt_val_file}")
        except Exception as e:
            print(f"ERROR saving HPT validation baked data: {e}")
    else:
        print("HPT validation data is empty, not saving file.")

    # --- 7. Sample Individuals (from fitting data only) ---
    print("\n===== STEP 7: Sampling Individuals for Training Splits (from Fitting Data) =====")
    all_person_ids_fit = fit_baked_df['cpsidp'].unique()
    n_all_persons_fit = len(all_person_ids_fit)
    sampled_ids = all_person_ids_fit # Default to all IDs from the fitting period

    if config.PREPROCESS_NUM_INDIVIDUALS is not None and config.PREPROCESS_NUM_INDIVIDUALS > 0:
        if config.PREPROCESS_NUM_INDIVIDUALS >= n_all_persons_fit:
            print(f"Requested {config.PREPROCESS_NUM_INDIVIDUALS} individuals, have {n_all_persons_fit} in fitting period. Using all available.")
        else:
            print(f"Sampling {config.PREPROCESS_NUM_INDIVIDUALS} individuals from {n_all_persons_fit} available in fitting period...")
            sampled_ids = np.random.choice(all_person_ids_fit, config.PREPROCESS_NUM_INDIVIDUALS, replace=False)
            print(f"Selected {len(sampled_ids)} individuals for training/val/test splits.")
    else:
        print("PREPROCESS_NUM_INDIVIDUALS not set or <= 0. Using all individuals from fitting period for splits.")

    n_sampled_persons = len(sampled_ids)

    # --- 8. Split Sampled IDs into Train/Val/Test ---
    print("\n===== STEP 8: Splitting Sampled Individual IDs =====")
    if n_sampled_persons == 0:
        print("ERROR: No individuals available from fitting data after potential sampling. Cannot create splits.")
        return

    train_size = int(config.TRAIN_SPLIT * n_sampled_persons)
    val_size = int(config.VAL_SPLIT * n_sampled_persons)
    test_size = n_sampled_persons - train_size - val_size

    if test_size < 0:
        print("Warning: Train + Validation split > 1.0. Adjusting validation size.")
        val_size = n_sampled_persons - train_size
        test_size = 0

    # Split the sampled IDs
    train_ids, remaining_ids = train_test_split(sampled_ids, train_size=train_size, random_state=config.RANDOM_SEED)
    val_ids, test_ids = np.array([]), np.array([]) # Initialize
    if len(remaining_ids) > 0 and (val_size + test_size) > 0 :
         val_prop_remaining = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0
         if val_prop_remaining > 0 and val_prop_remaining < 1:
             val_ids, test_ids = train_test_split(remaining_ids, train_size=val_prop_remaining, random_state=config.RANDOM_SEED)
         elif val_prop_remaining >= 1: val_ids = remaining_ids
         else: test_ids = remaining_ids

    print(f"Sampled Individuals Split - Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")

    # --- 9. Filter Baked Fitting Data to Create Splits ---
    print("\n===== STEP 9: Creating Data Splits from Baked Fitting Data =====")
    train_data_baked = fit_baked_df[fit_baked_df['cpsidp'].isin(train_ids)].copy()
    val_data_baked = fit_baked_df[fit_baked_df['cpsidp'].isin(val_ids)].copy() if len(val_ids) > 0 else pd.DataFrame(columns=fit_baked_df.columns)
    test_data_baked = fit_baked_df[fit_baked_df['cpsidp'].isin(test_ids)].copy() if len(test_ids) > 0 else pd.DataFrame(columns=fit_baked_df.columns)

    print(f"Shape of final baked train data (sampled): {train_data_baked.shape}")
    print(f"Shape of final baked val data (sampled): {val_data_baked.shape}")
    print(f"Shape of final baked test data (sampled): {test_data_baked.shape}")

    # --- 10. Save Sampled Data Splits ---
    print("\n===== STEP 10: Saving Sampled Data Splits (Parquet) =====")
    try:
        train_data_baked.to_parquet(train_file, index=False)
        val_data_baked.to_parquet(val_file, index=False)
        test_data_baked.to_parquet(test_file, index=False)
        print("Sampled baked data splits saved:")
        print(f" - Train: {train_file}")
        print(f" - Validation: {val_file}")
        print(f" - Test: {test_file}")
    except Exception as e:
        print(f"ERROR saving sampled baked data splits: {e}")

    # --- 11. Save Metadata ---
    print("\n===== STEP 11: Saving Metadata =====")
    n_features = len(final_feature_names_after_vt)

    print("Calculating n_classes from target_state column in fitting data...")
    unique_target_values_inc_na = df_final_fit['target_state'].unique()
    valid_target_values = unique_target_values_inc_na[~pd.isna(unique_target_values_inc_na)]
    n_classes = len(valid_target_values)
    print(f"Final calculated n_classes: {n_classes}")
    if len(df_final_fit) > 0 and n_classes == 0: # Check if df_final is not empty
        print("WARNING: Target state column appears to be all NA values or the cleaned fitting dataframe is empty.")
        print("STOPPING: Cannot determine n_classes.")
        return # Stop execution
    elif n_classes != 3:
        print(f"WARNING: Expected 3 classes (0, 1, 2) but found {n_classes} unique non-NA values: {np.sort(valid_target_values)}.")
        print("STOPPING: Incorrect number of target classes determined.")
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
        'train_individuals': len(train_ids),
        'val_individuals': len(val_ids),
        'test_individuals': len(test_ids),
        'train_rows_baked': len(train_data_baked),
        'val_rows_baked': len(val_data_baked),
        'test_rows_baked': len(test_data_baked),
        'hpt_val_rows_baked': len(hpt_baked_df), # Add count for HPT val set
        'preprocessing_pipeline_path': str(preprocessor_file),
        'hpt_validation_data_path': str(hpt_val_file) if hpt_val_file.exists() else None, # Add path to HPT val data
        'full_baked_data_path': str(full_baked_file), # Add path to the full data
        'weight_column': 'wtfinl' # Add weight column name to metadata
    }

    try:
        with open(metadata_file, 'wb') as f: pickle.dump(metadata, f)
        print(f"Metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"ERROR saving metadata: {e}")

    print("\n--- Preprocessing Script Finished ---")

if __name__ == "__main__":
    preprocess_data()