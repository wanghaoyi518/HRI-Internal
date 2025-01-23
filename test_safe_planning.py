import numpy as np
import matplotlib.pyplot as plt
from safe_planning import SafePlanner  # Import SafePlanner from proper module
from models.training import HumanModelTrainer
from typing import Dict, List, Optional  # Add type hints
import os
import json

def get_latest_dir():
    """Get path to most recent results directory"""
    results_dirs = sorted(os.listdir("results"))
    if not results_dirs:
        raise ValueError("No results directory found")
    return os.path.join("results", results_dirs[-1])

def load_latest_data():
    """Load demonstrations and trained model from most recent results"""
    latest_dir = get_latest_dir()  # Use get_latest_dir function
    
    # Load trained model
    model_file = os.path.join(latest_dir, "data", "training_results.json")
    if not os.path.exists(model_file):
        raise ValueError(f"No trained model found at {model_file}")
        
    trainer = HumanModelTrainer()
    trainer.load_model(model_file)
    
    return trainer

def test_intersection_scenario():
    """Test safe planning in intersection scenario"""
    print("Loading data...")
    trainer = load_latest_data()  # Only need trainer, not demonstrations
    
    # Get latest directory path and check for conformal bounds
    latest_dir = get_latest_dir()
    bounds_file = os.path.join(latest_dir, "data", "conformal_bounds.json")
    
    if not os.path.exists(bounds_file):
        raise ValueError(
            "Conformal bounds not found. Please run in this order:\n"
            "1. python generate_scenarios.py\n"
            "2. python train_model.py\n"
            "3. python calibrate_conformal.py\n"
            "4. python test_safe_planning.py"
        )
    
    # Initialize safe planner
    print("\nInitializing safe planner...")
    planner = SafePlanner(
        human_model=trainer,
        horizon=10,
        dt=0.1,
        safety_margin=1.2
    )
    
    # Load pre-computed conformal prediction bounds
    planner.load_calibrated_bounds(bounds_file)
    
    # Test scenario setup
    # Robot moving left to right, human moving bottom to top
    robot_state = np.array([0.0, 5.0, 0.0, 0.0])  # [x,y,θ,v]
    human_state = np.array([5.0, 0.0, np.pi/2, 0.5])  # Moving upward
    human_action = np.array([0.0, 0.5])  # [steering, acceleration]
    robot_goal = np.array([10.0, 5.0])
    
    print("\nGenerating safe plan...")
    # Generate plan
    plan = planner.plan(
        robot_state=robot_state,
        human_state=human_state,
        human_action=human_action,
        robot_goal=robot_goal,
        verbose=True
    )
    
    if not plan['success']:
        print("Warning: Optimization did not converge!")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot full scene
    planner.visualize_plan(
        robot_state=robot_state,
        human_state=human_state,
        human_action=human_action,
        robot_goal=robot_goal,
        ax=ax1
    )
    ax1.set_title('Safe Planning - Full Scene')
    
    # Plot detailed view of intersection
    planner.visualize_plan(
        robot_state=robot_state,
        human_state=human_state,
        human_action=human_action,
        robot_goal=robot_goal,
        ax=ax2
    )
    ax2.set_xlim(3, 7)  # Zoom in on intersection
    ax2.set_ylim(3, 7)
    ax2.set_title('Safe Planning - Intersection Detail')
    
    plt.tight_layout()
    
    # Save results
    results_dirs = sorted(os.listdir("results"))
    latest_dir = os.path.join("results", results_dirs[-1])
    plots_dir = os.path.join(latest_dir, "plots")
    
    save_path = os.path.join(plots_dir, "safe_planning_test.png")
    plt.savefig(save_path)
    print(f"\nTest plot saved to {save_path}")

def test_multiple_scenarios():
    """Test safe planning with different human behaviors"""
    print("Loading data...")
    trainer = load_latest_data()  # Only unpack trainer since that's all we need
    
    # Get latest directory path and check for conformal bounds
    latest_dir = get_latest_dir()
    bounds_file = os.path.join(latest_dir, "data", "conformal_bounds.json")
    
    if not os.path.exists(bounds_file):
        raise ValueError(
            "Conformal bounds not found. Please run calibrate_conformal.py first"
        )
    
    # Initialize safe planner
    print("\nInitializing safe planner...")
    planner = SafePlanner(
        human_model=trainer,
        horizon=10,
        dt=0.1,
        safety_margin=1.2
    )
    
    # Load pre-computed conformal prediction bounds
    planner.load_calibrated_bounds(bounds_file)
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'Attentive Human',
            'human_state': np.array([5.0, 0.0, np.pi/2, 0.3]),
            'human_action': np.array([0.0, 0.3])  # Cautious approach
        },
        {
            'name': 'Aggressive Human',
            'human_state': np.array([5.0, 0.0, np.pi/2, 0.8]),
            'human_action': np.array([0.0, 0.8])  # Fast approach
        },
        {
            'name': 'Delayed Human',
            'human_state': np.array([5.0, -2.0, np.pi/2, 0.0]),
            'human_action': np.array([0.0, 0.0])  # Starting from stop
        }
    ]
    
    # Common robot setup
    robot_state = np.array([0.0, 5.0, 0.0, 0.0])
    robot_goal = np.array([10.0, 5.0])
    
    # Create subplot for each scenario
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for ax, scenario in zip(axes, scenarios):
        print(f"\nTesting {scenario['name']}...")
        
        plan = planner.plan(
            robot_state=robot_state,
            human_state=scenario['human_state'],
            human_action=scenario['human_action'],
            robot_goal=robot_goal
        )
        
        planner.visualize_plan(
            robot_state=robot_state,
            human_state=scenario['human_state'],
            human_action=scenario['human_action'],
            robot_goal=robot_goal,
            ax=ax
        )
        ax.set_title(scenario['name'])
    
    plt.tight_layout()
    
    # Save results
    results_dirs = sorted(os.listdir("results"))
    latest_dir = os.path.join("results", results_dirs[-1])
    plots_dir = os.path.join(latest_dir, "plots")
    
    save_path = os.path.join(plots_dir, "safe_planning_scenarios.png")
    plt.savefig(save_path)
    print(f"\nScenarios plot saved to {save_path}")

if __name__ == "__main__":
    # Test basic intersection scenario
    test_intersection_scenario()
    
    # Test multiple scenarios
    test_multiple_scenarios()