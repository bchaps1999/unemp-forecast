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

        if config.PREPROCESS_START_DATE:
            start_dt = pd.to_datetime(config.PREPROCESS_START_DATE)
            df = df[df['date'] >= start_dt]
            print(f"Filtered data after start date ({config.PREPROCESS_START_DATE}): {df.shape}")
        if config.PREPROCESS_END_DATE:
            end_dt = pd.to_datetime(config.PREPROCESS_END_DATE)
            df = df[df['date'] <= end_dt]
            print(f"Filtered data after end date ({config.PREPROCESS_END_DATE}): {df.shape}")

        target_map = {"employed": 0, "unemployed": 1, "not_in_labor_force": 2}
        target_map_inverse = {v: k for k, v in target_map.items()}
        df['target_state'] = df['emp_state_f1'].map(target_map).astype('Int32')

        df['current_state'] = pd.Categorical(df['emp_state'], categories=target_map.keys())

        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['sex'] = pd.Categorical(df['sex'].map({1: "Male", 2: "Female"}))

        race_map = {
            "100": "White", "200": "Black", "300": "NativeAmerican",
            "651": "Asian", "652": "PacificIslander"
        }
        df['race_cat'] = pd.Categorical(df['race'].astype(str).map(race_map).fillna("Other/Mixed"))

        df['hispan_cat'] = np.select(
            [
                df['hispan'].astype(str).str.contains("not hispanic", case=False, na=False),
                df['hispan'].notna() & ~df['hispan'].astype(str).str.contains("not hispanic", case=False, na=False)
            ],
            [
                "NotHispanic", "Hispanic"
            ],
            default="Unknown"
        )
        df['hispan_cat'] = pd.Categorical(df['hispan_cat'])

        conditions = [
            df['educ'].astype(str).str.contains("Advanced degree", case=False, na=False),
            df['educ'].astype(str).str.contains("Bachelor's degree", case=False, na=False),
            df['educ'].astype(str).str.contains("Some college", case=False, na=False),
            df['educ'].notna()
        ]
        choices = [
            "Advanced",
            "Bachelors",
            "Some_College",
            "HS_or_less"
        ]
        df['educ_cat'] = np.select(conditions, choices, default="HS_or_less")
        educ_categories = ["HS_or_less", "Some_College", "Bachelors", "Advanced"]
        df['educ_cat'] = pd.Categorical(df['educ_cat'], categories=educ_categories, ordered=False)

        df['statefip'] = df['statefip'].astype(float).astype(int).astype(str).str.zfill(2)
        df['statefip'] = pd.Categorical(df['statefip'])

        df['faminc_cat'] = clean_faminc(df['faminc'].astype(str))

        df['whyunemp'] = "Not_Unemployed"
        unemployed_mask = df['current_state'] == "unemployed"
        df.loc[unemployed_mask, 'whyunemp'] = df.loc[unemployed_mask, 'whyunemp'].astype(str)
        df['whyunemp'] = pd.Categorical(df['whyunemp'])

        df['ahrsworkt_cat'] = "Not_Employed"
        employed_mask = df['current_state'] == "employed"
        df.loc[employed_mask, 'ahrsworkt_cat'] = df.loc[employed_mask, 'ahrsworkt'].astype(str)
        df.loc[df['ahrsworkt_cat'].isin(['nan', 'NA', '']), 'ahrsworkt_cat'] = "Unknown_Hours"
        df['ahrsworkt_cat'] = df['ahrsworkt_cat'].str.replace("40+", "40plus", regex=False)
        df['ahrsworkt_cat'] = pd.Categorical(df['ahrsworkt_cat'])

        df['durunemp'] = pd.to_numeric(df['durunemp'], errors='coerce')
        df.loc[df['current_state'] != "unemployed", 'durunemp'] = 0

        df['ind_group_cat'] = df['ind_group'].astype(str).replace(['', 'nan'], 'Unknown')
        df['ind_group_cat'] = pd.Categorical(df['ind_group_cat'])

        df['occ_2dig_cat'] = df['occ_2dig'].astype(str).replace(['', 'nan', '99'], 'Unknown')
        occ_mask = df['occ_2dig_cat'] != 'Unknown'
        df.loc[occ_mask, 'occ_2dig_cat'] = 'Occ' + pd.to_numeric(df.loc[occ_mask, 'occ_2dig_cat'], errors='coerce').astype(int).astype(str).str.zfill(2)
        df['occ_2dig_cat'] = pd.Categorical(df['occ_2dig_cat'])

        df['mth_dim1'] = pd.to_numeric(df['mth_dim1'], errors='coerce')
        df['mth_dim2'] = pd.to_numeric(df['mth_dim2'], errors='coerce')

        df = df.sort_values(['cpsidp', 'date'])
        df['time_diff_days'] = df.groupby('cpsidp')['date'].diff().dt.days
        df['months_since_last'] = (df['time_diff_days'] / 30.4375).round()
        df['months_since_last'] = df['months_since_last'].fillna(0).astype(int)

    except Exception as e:
        print(f"ERROR during data cleaning: {e}")
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
    base_id_cols = ['cpsidp', 'date', 'target_state']

    # Ensure original ID columns are present
    missing_orig_ids = [col for col in original_id_cols if col not in df.columns]
    if missing_orig_ids:
        raise ValueError(f"Missing original identifier columns needed for forecasting: {missing_orig_ids}")

    # Columns to keep before processing
    keep_cols = list(dict.fromkeys(base_id_cols + predictor_cols + original_id_cols))
    df_final = df[keep_cols].copy()
    print(f"Data shape before preprocessing: {df_final.shape}")
    print(f"Predictor columns for pipeline: {predictor_cols}")
    print(f"Original ID columns kept: {original_id_cols}")

    # --- 3. Handle Sparse Categories (on full data before fitting pipeline) ---
    print("\n===== STEP 3: Grouping Sparse Categorical Features (on Full Data) =====")
    potential_categorical_features = df_final[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Identified potential categorical features for sparsity check: {potential_categorical_features}")
    sparsity_threshold = getattr(config, 'SPARSITY_THRESHOLD', 0.01)
    print(f"Using sparsity threshold: {sparsity_threshold}")

    if sparsity_threshold > 0 and not df_final.empty:
        for col in potential_categorical_features:
            frequencies = df_final[col].value_counts(normalize=True)
            sparse_cats = frequencies[frequencies < sparsity_threshold].index.tolist()
            if sparse_cats:
                print(f"  Grouping sparse categories in '{col}': {sparse_cats}")
                other_category = "_OTHER_"
                if pd.api.types.is_categorical_dtype(df_final[col]):
                    if other_category not in df_final[col].cat.categories:
                        df_final[col] = df_final[col].cat.add_categories([other_category])
                    df_final[col] = df_final[col].replace(sparse_cats, other_category)
                else:
                    df_final[col] = df_final[col].replace(sparse_cats, other_category)
    else:
        print("Skipping sparse category grouping.")

    # --- 4. Define and Fit Preprocessing Pipeline (on full data) ---
    print("\n===== STEP 4: Defining and Fitting Preprocessing Pipeline (on Full Data) =====")
    numeric_features = df_final[predictor_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_final[predictor_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Define transformers and pipeline
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    full_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('variance_threshold', VarianceThreshold(threshold=0))])

    # Fit the pipeline on ALL data's predictor columns
    print("Fitting pipeline...")
    try:
        full_pipeline.fit(df_final[predictor_cols], df_final['target_state']) # Fit on all predictors
    except Exception as e:
        print(f"ERROR fitting pipeline: {e}")
        raise
    print("Pipeline fitted.")

    # Save the fitted pipeline
    try:
        with open(preprocessor_file, 'wb') as f: pickle.dump(full_pipeline, f)
        print(f"Preprocessing pipeline saved to: {preprocessor_file}")
    except Exception as e: print(f"ERROR saving preprocessor: {e}")

    # --- 5. Apply Pipeline to All Data ---
    print("\n===== STEP 5: Applying Pipeline to Full Data =====")
    X_full_baked_np = full_pipeline.transform(df_final[predictor_cols])

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
        final_feature_names_after_vt = [f"feature_{i}" for i in range(X_full_baked_np.shape[1])]

    # Create the full baked DataFrame
    full_baked_df = pd.DataFrame(X_full_baked_np, columns=final_feature_names_after_vt, index=df_final.index)
    # Add back base IDs and original IDs
    id_cols_to_add = base_id_cols + original_id_cols
    full_baked_df = pd.concat([df_final[id_cols_to_add], full_baked_df], axis=1)
    print(f"Full baked data shape: {full_baked_df.shape}")

    # --- 6. Save Full Baked Data ---
    print("\n===== STEP 6: Saving Full Baked Data =====")
    try:
        full_baked_df.to_parquet(full_baked_file, index=False)
        print(f"Full baked data saved to: {full_baked_file}")
    except Exception as e:
        print(f"ERROR saving full baked data: {e}")

    # --- 7. Sample Individuals (if specified) ---
    print("\n===== STEP 7: Sampling Individuals for Training Splits =====")
    all_person_ids = df_final['cpsidp'].unique()
    n_all_persons = len(all_person_ids)
    sampled_ids = all_person_ids # Default to all IDs

    if config.PREPROCESS_NUM_INDIVIDUALS is not None and config.PREPROCESS_NUM_INDIVIDUALS > 0:
        if config.PREPROCESS_NUM_INDIVIDUALS >= n_all_persons:
            print(f"Requested {config.PREPROCESS_NUM_INDIVIDUALS} individuals, have {n_all_persons}. Using all available.")
        else:
            print(f"Sampling {config.PREPROCESS_NUM_INDIVIDUALS} individuals from {n_all_persons} available...")
            sampled_ids = np.random.choice(all_person_ids, config.PREPROCESS_NUM_INDIVIDUALS, replace=False)
            print(f"Selected {len(sampled_ids)} individuals for training/val/test splits.")
    else:
        print("PREPROCESS_NUM_INDIVIDUALS not set or <= 0. Using all individuals for splits.")

    n_sampled_persons = len(sampled_ids)

    # --- 8. Split Sampled IDs into Train/Val/Test ---
    print("\n===== STEP 8: Splitting Sampled Individual IDs =====")
    if n_sampled_persons == 0:
        print("ERROR: No individuals available after potential sampling. Cannot create splits.")
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

    # --- 9. Filter Full Baked Data to Create Splits ---
    print("\n===== STEP 9: Creating Data Splits from Full Baked Data =====")
    train_data_baked = full_baked_df[full_baked_df['cpsidp'].isin(train_ids)].copy()
    val_data_baked = full_baked_df[full_baked_df['cpsidp'].isin(val_ids)].copy() if len(val_ids) > 0 else pd.DataFrame(columns=full_baked_df.columns)
    test_data_baked = full_baked_df[full_baked_df['cpsidp'].isin(test_ids)].copy() if len(test_ids) > 0 else pd.DataFrame(columns=full_baked_df.columns)

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

    print("Calculating n_classes from target_state column...")
    unique_target_values_inc_na = df_final['target_state'].unique()
    valid_target_values = unique_target_values_inc_na[~pd.isna(unique_target_values_inc_na)]
    n_classes = len(valid_target_values)
    print(f"Final calculated n_classes: {n_classes}")
    if len(df_final) > 0 and n_classes == 0: # Check if df_final is not empty
        print("WARNING: Target state column appears to be all NA values or the cleaned dataframe is empty.")
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
        'preprocessing_pipeline_path': str(preprocessor_file),
        'full_baked_data_path': str(full_baked_file) # Add path to the full data
    }

    try:
        with open(metadata_file, 'wb') as f: pickle.dump(metadata, f)
        print(f"Metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"ERROR saving metadata: {e}")

    print("\n--- Preprocessing Script Finished ---")

if __name__ == "__main__":
    preprocess_data()