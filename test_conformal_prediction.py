import numpy as np
import matplotlib.pyplot as plt
from conformal_prediction import ConformalPredictor
import os

def load_latest_demonstrations():
    """Load demonstrations from most recent results directory"""
    results_dirs = sorted(os.listdir("results"))
    if not results_dirs:
        raise ValueError(
            "No results directory found. Please run in this order:\n"
            "1. python generate_scenarios.py\n"
            "2. python main.py\n"
            "3. python test_conformal_prediction.py"
        )
    
    latest_dir = os.path.join("results", results_dirs[-1])
    demo_file = os.path.join(latest_dir, "data", "demonstrations.npy")
    
    if not os.path.exists(demo_file):
        raise ValueError(
            f"No demonstrations found at {demo_file}\n"
            "Please run the scripts in this order:\n"
            "1. python generate_scenarios.py\n"
            "2. python main.py\n"
            "3. python test_conformal_prediction.py"
        )
        
    return np.load(demo_file, allow_pickle=True)

def test_conformal_prediction():
    """Test conformal prediction implementation"""
    # Load demonstrations and trained model
    print("Loading data...")
    demonstrations = load_latest_demonstrations()
    
    # Load trained model
    results_dirs = sorted(os.listdir("results"))
    latest_dir = os.path.join("results", results_dirs[-1])
    model_file = os.path.join(latest_dir, "data", "training_results.json")
    
    from models.training import HumanModelTrainer
    trainer = HumanModelTrainer()
    trainer.load_model(model_file)
    
    # Initialize predictor with human model
    predictor = ConformalPredictor(trainer, significance_level=0.1)
    
    # Calibrate
    predictor.calibrate(demonstrations)
    
    # Test prediction
    print("\nTesting prediction...")
    
    # Use first state-action pair from demonstrations
    demo = demonstrations[0]
    initial_state = demo.states[0]
    initial_action = demo.actions[0]
    initial_robot_state = demo.robot_states[0]  # Get robot state too
    
    # Make predictions for different horizons
    horizons = [1, 3, 5, 10]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for ax, horizon in zip(axes, horizons):
        predictor.visualize_prediction(
            initial_state,
            initial_action,
            initial_robot_state,  # Pass robot state
            horizon,
            ax
        )
        
    plt.tight_layout()
    
    # Save plot
    results_dirs = sorted(os.listdir("results"))
    latest_dir = os.path.join("results", results_dirs[-1])
    plots_dir = os.path.join(latest_dir, "plots")
    
    save_path = os.path.join(plots_dir, "conformal_prediction_test.png")
    plt.savefig(save_path)
    print(f"\nTest plot saved to {save_path}")

if __name__ == "__main__":
    test_conformal_prediction()