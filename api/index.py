from flask import Flask, request, jsonify
from padelpy import from_smiles # Ensure this is installed and PaDEL-Descriptor is configured
import pandas as pd
from joblib import load
import os
import numpy as np # Added for regression
import sklearn # Referenced by joblib, good to have explicit import if version matters
from flask_cors import CORS

app = Flask(__name__)

# CORS Configuration (user's original)
CORS(app)

# --- Determine Base Directory ---
base_dir = os.path.dirname(__file__)

# --- Classification Model Loading ---
classifier_model_filename = 'pipeline_voting.joblib'
classifier_model_path = os.path.join(base_dir, classifier_model_filename)
clf_model = None # Initialize to None

try:
    if os.path.exists(classifier_model_path):
        classifier_pipeline = load(classifier_model_path)
        if hasattr(classifier_pipeline, 'named_steps') and 'model' in classifier_pipeline.named_steps:
            clf_model = classifier_pipeline.named_steps['model']
            print("Classification model (step 'model' from pipeline) loaded successfully.")
        else:
            clf_model = classifier_pipeline
            print("Classification model (direct load) loaded successfully.")
    else:
        print(f"Error: Classification model file not found at {classifier_model_path}")
except Exception as e:
    print(f"Error loading classification model from {classifier_model_path}: {str(e)}")

# --- Regression Models Loading ---
REGRESSION_MODEL_FOLDER_NAME = 'tree_reg_models'
REGRESSION_MODEL_DIR_PATH = os.path.join(base_dir, REGRESSION_MODEL_FOLDER_NAME)
NUM_REGRESSION_MODELS = 100
saved_regression_trees = []

if clf_model:
    try:
        print(f"Attempting to load regression models from: {REGRESSION_MODEL_DIR_PATH}")
        loaded_count = 0
        if not os.path.isdir(REGRESSION_MODEL_DIR_PATH):
            print(f"Warning: Regression model directory not found: {REGRESSION_MODEL_DIR_PATH}")
        else:
            for i in range(NUM_REGRESSION_MODELS):
                tree_model_file = os.path.join(REGRESSION_MODEL_DIR_PATH, f'tree_{i}.joblib')
                if os.path.exists(tree_model_file):
                    try:
                        tree = load(tree_model_file)
                        saved_regression_trees.append(tree)
                        loaded_count += 1
                    except Exception as load_exc:
                        print(f"Error loading regression model file {tree_model_file}: {str(load_exc)}")
            
            if loaded_count > 0:
                print(f"Successfully loaded {loaded_count} out of {NUM_REGRESSION_MODELS} expected regression tree models.")
            if loaded_count < NUM_REGRESSION_MODELS and loaded_count > 0 :
                print(f"Warning: Expected {NUM_REGRESSION_MODELS} regression models, but only found/loaded {loaded_count}.")
            elif loaded_count == 0 and os.path.isdir(REGRESSION_MODEL_DIR_PATH):
                print(f"Warning: No regression tree models were loaded from {REGRESSION_MODEL_DIR_PATH}. Regression functionality will be unavailable.")
    except Exception as e:
        print(f"An unexpected error occurred during the setup for loading regression models: {str(e)}")

# --- Define Descriptor Lists ---

# Descriptors for the CLASSIFICATION model
classification_required_descriptors = [
    'nN', 'nX', 'AATS2i', 'nBondsD', 'nBondsD2', 'C1SP2', 'C3SP2', 'SCH-5',
    'nHssNH', 'ndssC', 'nssNH', 'SdssC', 'SdS', 'mindO', 'mindS', 'minssS',
    'maxdssC', 'ETA_dAlpha_B', 'MDEN-23', 'n5Ring', 'nT5Ring', 'nHeteroRing',
    'n5HeteroRing', 'nT5HeteroRing', 'SRW5', 'SRW7', 'SRW9', 'WTPT-5'
]

# Descriptors for the REGRESSION models (EXACT ORDER from your model inspection)
regression_required_descriptors = [
    'AMR', 'ATS6s', 'AATS2i', 'AATS5i', 'ATSC3v', 'ATSC5v', 'ATSC2p', 'ATSC4i', 
    'MATS5s', 'GATS4c', 'GATS7c', 'GATS7e', 'BCUTp-1l', 'SwHBa', 'JGI3', 
    'JGI9', 'JGI10', 'TDB10e', 'PPSA-1', 'FPSA-3', 'Du'
]

print(f"Classification model will use {len(classification_required_descriptors)} descriptors.")
print(f"Regression models will use {len(regression_required_descriptors)} descriptors (verified order).")


@app.route("/api/predict", methods=["POST"])
def predict():
    if clf_model is None:
        return jsonify({"error": "Classification model is not available. Please check server configuration."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided. Please send a JSON payload."}), 400

    compounds_input = data.get("compound")
    percentage_str = data.get("percentage")

    if not isinstance(compounds_input, list):
        if isinstance(compounds_input, str):
            compounds_input = [compounds_input]
        else:
            return jsonify({"error": "'compound' field must be a list of SMILES strings or a single SMILES string."}), 400
    
    if not all(isinstance(s, str) for s in compounds_input):
        return jsonify({"error": "All items in the 'compound' list must be SMILES strings."}), 400

    percentage = 95.0 
    if percentage_str is not None:
        try:
            percentage = float(percentage_str)
            if not (0 < percentage < 100):
                return jsonify({"error": "Percentage for confidence interval must be a number strictly between 0 and 100."}), 400
        except ValueError:
            return jsonify({"error": f"Invalid 'percentage' value: '{percentage_str}'. Must be a number."}), 400
    
    all_classification_results = {}
    all_regression_results = {}
    all_descriptors_results = {}
    batch_processing_errors = []

    print(f"Starting batch processing for {len(compounds_input)} compounds.")

    for smiles_string in compounds_input:
        if not smiles_string:
            key_name = f"EMPTY_INPUT_{len(all_classification_results)}"
            all_classification_results[key_name] = "Error: Empty SMILES string provided in the list."
            batch_processing_errors.append({"input_smiles": "", "error": "Empty SMILES string provided."})
            continue

        try:
            print(f"Processing SMILES: {smiles_string[:30]}...")
            descriptor_dict = from_smiles(smiles_string, timeout=60)
            
            # --- Extract descriptor values for display ---
            descriptors_for_display = {}
            for key in classification_required_descriptors:
                value = descriptor_dict.get(key)
                if value is None or value == '':
                    descriptors_for_display[key] = 0.0
                else:
                    try:
                        descriptors_for_display[key] = float(value)
                    except ValueError:
                        descriptors_for_display[key] = 0.0
            all_descriptors_results[smiles_string] = descriptors_for_display
            
            # --- Prepare DataFrame for Classification ---
            clf_filtered_values = []
            for key in classification_required_descriptors:
                value = descriptor_dict.get(key)
                val_to_append = 0.0
                if value is None:
                    print(f"Warning (Classifier - SMILES: {smiles_string[:15]}): Descriptor '{key}' not found. Defaulting to 0.0.")
                else:
                    try:
                        val_to_append = float(value)
                    except ValueError:
                        print(f"Warning (Classifier - SMILES: {smiles_string[:15]}): Descriptor '{key}' non-numeric ('{value}'). Defaulting to 0.0.")
                clf_filtered_values.append(val_to_append)
            df_classification = pd.DataFrame([clf_filtered_values], columns=classification_required_descriptors)

            # --- Classification Prediction ---
            classification_prediction_array = clf_model.predict(df_classification)
            classification_result = str(classification_prediction_array[0])
            all_classification_results[smiles_string] = classification_result

            # --- Regression Prediction (if "Activator") ---
            if classification_result == "Activator":
                if not saved_regression_trees:
                    all_regression_results[smiles_string] = {"error": "Regression models are not loaded or failed to load."}
                else:
                    # --- Prepare DataFrame for Regression ---
                    reg_filtered_values = []
                    for key in regression_required_descriptors: # Use the CORRECT, ORDERED regression descriptor list
                        value = descriptor_dict.get(key)
                        val_to_append = 0.0
                        if value is None:
                            print(f"Warning (Regression - SMILES: {smiles_string[:15]}): Descriptor '{key}' not found for regression. Defaulting to 0.0.")
                        else:
                            try:
                                val_to_append = float(value)
                            except ValueError: # Handles cases like PaDELPy returning ''
                                print(f"Warning (Regression - SMILES: {smiles_string[:15]}): Descriptor '{key}' non-numeric ('{value}'). Defaulting to 0.0.")
                        reg_filtered_values.append(val_to_append)
                    df_regression = pd.DataFrame([reg_filtered_values], columns=regression_required_descriptors)
                    
                    individual_reg_predictions_for_smiles = []
                    has_regression_tree_error = False
                    for idx, tree_model in enumerate(saved_regression_trees):
                        try:
                            pred = tree_model.predict(df_regression) # Use df_regression
                            individual_reg_predictions_for_smiles.append(pred[0])
                        except Exception as tree_pred_e:
                            print(f"Error (SMILES: {smiles_string[:15]}): Predicting with regression tree model index {idx}: {str(tree_pred_e)}")
                            has_regression_tree_error = True
                    
                    if not individual_reg_predictions_for_smiles and has_regression_tree_error:
                        all_regression_results[smiles_string] = {"error": "Regression prediction failed for all attempted tree models for this compound."}
                    elif not individual_reg_predictions_for_smiles:
                        all_regression_results[smiles_string] = {"error": "No regression predictions could be made (e.g. no successful tree model predictions)."}
                    else:
                        all_reg_predictions_np = np.array(individual_reg_predictions_for_smiles)
                        alpha = (100.0 - percentage) / 2.0
                        
                        lower_bound = float(np.percentile(all_reg_predictions_np, alpha))
                        upper_bound = float(np.percentile(all_reg_predictions_np, 100.0 - alpha))
                        median_prediction = float(np.median(all_reg_predictions_np))

                        all_regression_results[smiles_string] = {
                            "regression_AC50_median": median_prediction,
                            "regression_AC50_lower_bound": lower_bound,
                            "regression_AC50_upper_bound": upper_bound,
                            "confidence_interval_percentage": percentage,
                            "num_regression_models_used": len(all_reg_predictions_np)
                        }
        
        except RuntimeError as re_padel:
            error_message = f"PaDELPy processing failed: {str(re_padel)}"
            print(f"Error for SMILES '{smiles_string}': {error_message}")
            all_classification_results[smiles_string] = "Error: PaDELPy processing failed."
            batch_processing_errors.append({"smiles": smiles_string, "error": error_message})
        except ValueError as ve_desc:
            error_message = f"Descriptor value error: {str(ve_desc)}"
            print(f"Error for SMILES '{smiles_string}': {error_message}")
            all_classification_results[smiles_string] = "Error: Invalid descriptor data."
            batch_processing_errors.append({"smiles": smiles_string, "error": error_message})
        except Exception as e_general:
            error_message = f"General processing error: {str(e_general)}"
            print(f"Error for SMILES '{smiles_string}': {error_message}")
            all_classification_results[smiles_string] = "Error: Processing failed."
            batch_processing_errors.append({"smiles": smiles_string, "error": error_message})

    final_response = {
        "classification_results": all_classification_results,
        "regression_results": all_regression_results,
        "descriptors_results": all_descriptors_results
    }
    if batch_processing_errors:
        final_response["batch_processing_errors"] = batch_processing_errors
    
    print("Batch processing complete. Returning results.")
    return jsonify(final_response)

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "8080")) 
    print(f"Flask app starting on host 0.0.0.0, port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)