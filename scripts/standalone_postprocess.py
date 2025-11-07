import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
CONFIDENCE_THRESHOLD = 0.990  # Absolute confidence threshold for motor detection
Y_REFINEMENT_WINDOW = 3  # Window size for Y-axis coordinate refinement

def refine_y_coordinate(pred, z, y, x, window_size):
    """Refine Y coordinate using probability distribution in neighborhood"""
    y_idx = y.item()
    y_start = max(0, y_idx - window_size)
    y_end = min(pred.shape[1], y_idx + window_size + 1)
    y_probs = pred[z, y_start:y_end, x].cpu().numpy()
    
    if len(y_probs) > 1:
        weights = y_probs / y_probs.sum()
        y_refined = sum((np.arange(len(y_probs)) + y_start) * weights)
        return y_refined
    return y_idx

def process_prediction(pred, shape, max_val):
    """Process a single prediction to get motor coordinates"""
    if max_val <= CONFIDENCE_THRESHOLD:
        return [-1, -1, -1]
    
    # Get coordinates from argmax
    idx = torch.argmax(pred)
    z, y, x = torch.unravel_index(idx, pred.shape)
    
    # Refine Y coordinate
    y_refined = refine_y_coordinate(pred, z, y, x, Y_REFINEMENT_WINDOW)
    
    # Convert to original scale
    coords = [
        (z.item() + 0.5) / pred.shape[0] * shape[0],  # Z coordinate
        (y_refined + 0.5) / pred.shape[1] * shape[1], # Refined Y coordinate
        (x.item() + 0.5) / pred.shape[2] * shape[2]   # X coordinate
    ]
    
    return coords

def process_predictions(prediction_dir):
    """Process all predictions in the directory"""
    predictions = []
    shapes = []
    tomo_ids = []
    max_vals = []
    
    # Check if directory exists
    if not os.path.exists(prediction_dir):
        print(f"Error: Prediction directory '{prediction_dir}' does not exist!")
        return pd.DataFrame()
        
    # Load predictions and metadata
    print(f"Checking directory: {prediction_dir}")
    files = sorted(os.listdir(prediction_dir))
    print(f"Found {len(files)} files")
    pred_files = [f for f in files if f.endswith('_pred.pt')]
    print(f"Found {len(pred_files)} prediction files")
    
    if not pred_files:
        print("No prediction files found! Please run inference first.")
        return pd.DataFrame()
    
    # Load predictions and metadata
    print("\nLoading predictions...")
    for fname in pred_files:
            tomo_id = fname.split('_pred.pt')[0]
            
            # Load prediction
            pred_path = os.path.join(prediction_dir, fname)
            pred = torch.load(pred_path)
            predictions.append(pred)
            
            # Load metadata
            meta_path = pred_path.replace('.pt', '.json')
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            shapes.append((meta['z_shape'], meta['y_shape'], meta['x_shape']))
            tomo_ids.append(tomo_id)
            max_vals.append(float(pred.max()))
    
    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.4f}")
    print("\nMax values for each prediction:")
    for tid, mval in zip(tomo_ids, max_vals):
        print(f"{tid}: {mval:.4f} {'(Below threshold)' if mval <= CONFIDENCE_THRESHOLD else '(Above threshold)'}")
    
    # Process each prediction
    print("\nProcessing predictions...")
    rows = []
    for pred, shape, tomo_id, max_val in zip(predictions, shapes, tomo_ids, max_vals):
        coords = process_prediction(pred, shape, max_val)
        
        rows.append({
            'tomo_id': tomo_id,
            'Motor_Axis_0': coords[0],
            'Motor_Axis_1': coords[1],
            'Motor_Axis_2': coords[2],
            'max_val': max_val
        })
    
    # Create final submission
    submission = pd.DataFrame(rows)
    return submission

def main():
    # Paths
    pred_dir = r"D:\Project\Major_Project\FinalPhase_01\predictions"  # Directory with model predictions
    output_path = "./submission.csv"  # Final submission file
    detailed_path = "./detailed_coords.csv"  # Detailed results with confidence values
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for predictions in: {os.path.abspath(pred_dir)}")
    
    # Process predictions
    print("Processing predictions...")
    submission = process_predictions(pred_dir)
    
    if submission.empty:
        print("\nError: No predictions to process!")
        print("Please run inference first using standalone_ensemble_inference.py")
        return
    
    # Save outputs
    submission[['tomo_id', 'Motor_Axis_0', 'Motor_Axis_1', 'Motor_Axis_2']].to_csv(
        output_path, index=False)
    submission.to_csv(detailed_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"Submission: {output_path}")
    print(f"Detailed results: {detailed_path}")

if __name__ == "__main__":
    main()
