import numpy as np
import json
import os
from models.training import HumanModelTrainer
from conformal_prediction import TrajectoryPredictor

def get_latest_dir():
    """Get path to most recent results directory"""
    results_dirs = sorted(os.listdir("results"))
    if not results_dirs:
        raise ValueError("No results directory found")
    return os.path.join("results", results_dirs[-1])

def load_data_and_model():
    """Load demonstrations and trained model"""
    latest_dir = get_latest_dir()
    data_dir = os.path.join(latest_dir, "data")
    
    # Load demonstrations
    demo_file = os.path.join(data_dir, "demonstrations.npy")
    demonstrations = np.load(demo_file, allow_pickle=True)
    
    # Load trained model
    model_file = os.path.join(data_dir, "training_results.json")
    trainer = HumanModelTrainer()
    trainer.load_model(model_file)
    
    return demonstrations, trainer

def compute_prediction_errors(predictor, demonstrations, horizons=[1,3,5,10]):
    """Compute prediction errors for different horizons"""
    errors = {h: [] for h in horizons}
    
    print("Computing prediction errors...")
    for demo_idx, demo in enumerate(demonstrations):
        if not demo.actions or not demo.states or not demo.robot_states:
            continue
            
        # For each timestep that allows longest horizon prediction
        max_horizon = max(horizons)
        n_steps = len(demo.states) - max_horizon
        
        for t in range(n_steps):
            current_state = demo.states[t]
            current_action = demo.actions[t]
            current_robot_state = demo.robot_states[t]
            
            # For each prediction horizon
            for h in horizons:
                # Get predicted trajectory
                predicted = predictor.predict(
                    current_state, 
                    current_action,
                    current_robot_state,
                    h
                )
                
                # Get actual trajectory
                actual = demo.states[t+1:t+h+1]
                
                # Compute position errors
                pos_errors = [
                    np.linalg.norm(p[:2] - a[:2])
                    for p, a in zip(predicted, actual)
                ]
                
                # Store maximum error
                errors[h].append(max(pos_errors))
                
        if (demo_idx + 1) % 10 == 0:
            print(f"Processed {demo_idx + 1} demonstrations")
            
    return errors

def compute_conformal_bounds(errors, significance_level=0.1):
    """Compute conformal prediction bounds"""
    bounds = {}
    for horizon, horizon_errors in errors.items():
        if horizon_errors:
            bound = np.quantile(horizon_errors, 1 - significance_level)
            bounds[str(horizon)] = float(bound)
    return bounds

def calibrate_and_save():
    """Perform offline calibration and save results"""
    print("Loading data and model...")
    demonstrations, trainer = load_data_and_model()
    
    # Initialize predictor with proper parameters
    predictor = TrajectoryPredictor(trainer, dt=0.1)  # Match dt with planner
    
    # Compute prediction errors
    horizons = [1, 3, 5, 10]  # Match horizons with planner
    errors = compute_prediction_errors(predictor, demonstrations, horizons)
    
    # Compute conformal bounds
    significance_level = 0.1  # 90% coverage
    bounds = compute_conformal_bounds(errors, significance_level)
    
    # Save results
    calibration_data = {
        'quantiles': bounds,
        'significance_level': significance_level,
        'horizons': horizons,
        'n_calibration_points': {
            str(h): len(errs) for h, errs in errors.items()
        }
    }
    
    # Save to latest results directory
    latest_dir = get_latest_dir()
    save_path = os.path.join(latest_dir, "data", "conformal_bounds.json")
    
    with open(save_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
        
    print(f"\nCalibration complete. Results saved to {save_path}")
    print("\nConformal bounds:")
    for h, bound in bounds.items():
        print(f"Horizon {h}: {bound:.3f}")

if __name__ == "__main__":
    calibrate_and_save()
