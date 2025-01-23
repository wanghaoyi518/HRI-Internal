import os
import numpy as np
from models.training import HumanModelTrainer
from data.processor import DataProcessor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_training_losses(trainer, save_path):
    """Plot training losses over time"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(trainer.train_history['base_loss']) + 1)
    
    # Plot individual losses
    plt.plot(epochs, trainer.train_history['base_loss'], label='Base Loss', color='blue')
    plt.plot(epochs, trainer.train_history['attention_loss'], label='Attention Loss', color='red')
    plt.plot(epochs, trainer.train_history['style_loss'], label='Style Loss', color='green')
    plt.plot(epochs, trainer.train_history['total_loss'], label='Total Loss', color='black', linewidth=2)
    
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def evaluate_predictions(trainer, test_demos, plots_dir):
    """Evaluate model prediction accuracy"""
    print("\nEvaluating model predictions...")
    
    # Collect all state-action pairs
    states, actions, robot_states = [], [], []
    next_states = []  # Ground truth next states
    
    for demo in test_demos:
        for t in range(len(demo) - 1):  # Changed from demo.states to demo
            # Extract from processed demo format
            current_step = demo[t]
            next_step = demo[t + 1]
            
            states.append(current_step['state'])
            actions.append(current_step['action'])
            robot_states.append(current_step['robot_state'])
            next_states.append(next_step['state'])
            
    states = np.array(states)
    actions = np.array(actions)
    robot_states = np.array(robot_states)
    next_states = np.array(next_states)
    
    # Make predictions
    predicted_actions = []
    for s, a, r in zip(states, actions, robot_states):
        rewards = trainer.predict_reward_weights(s, a, r)
        # Normalize predictions using softmax for better scaling
        base_steering = np.tanh(rewards['base'])  # Use tanh to bound between -1 and 1
        attention_acc = np.tanh(rewards['attention'])
        
        pred_action = np.array([base_steering, attention_acc])
        predicted_actions.append(pred_action)
    predicted_actions = np.array(predicted_actions)

    # Compute metrics
    position_rmse = np.sqrt(np.mean(np.sum((next_states[:,:2] - states[:,:2])**2, axis=1)))
    velocity_rmse = np.sqrt(np.mean((next_states[:,3] - states[:,3])**2))
    
    print("\nPrediction Metrics:")
    print(f"Position RMSE: {position_rmse:.3f} units")
    print(f"Velocity RMSE: {velocity_rmse:.3f} units/s")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.scatter(states[:, 0], states[:, 1], c='blue', alpha=0.3, label='Current')
    plt.scatter(next_states[:, 0], next_states[:, 1], c='red', alpha=0.3, label='Next')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Position Transitions\nRMSE: {position_rmse:.3f}")
    plt.legend()
    
    plt.subplot(122)
    plt.scatter(states[:, 3], next_states[:, 3], alpha=0.3)
    plt.plot([0, np.max(states[:, 3])], [0, np.max(states[:, 3])], 'r--')
    plt.xlabel("Current Velocity")
    plt.ylabel("Next Velocity")
    plt.title(f"Velocity Transitions\nRMSE: {velocity_rmse:.3f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prediction_accuracy.png"))
    plt.close()

def main(results_dir: str = None):
    """
    Train human model using generated demonstrations
    Args:
        results_dir: Path to results directory containing demonstrations
    """
    if results_dir is None:
        # Use most recent results directory if none specified
        results_dirs = sorted(os.listdir("results"))
        if not results_dirs:
            raise ValueError("No results directory found. Run generate_scenarios.py first.")
        results_dir = os.path.join("results", results_dirs[-1])
    
    # Load demonstrations
    demo_file = os.path.join(results_dir, "data", "demonstrations.npy")  # Changed from demonstrations.json.npy
    if not os.path.exists(demo_file):
        raise ValueError(f"No demonstrations found at {demo_file}")
        
    demonstrations = np.load(demo_file, allow_pickle=True)
    
    # Process demonstrations
    print("\nProcessing demonstrations...")
    processor = DataProcessor()
    processed_demos = []
    for demo in demonstrations:
        processed_demo = processor.process_demonstration(demo)
        processed_demos.append(processed_demo)
    
    # Split demonstrations into train/test sets
    train_demos, test_demos = train_test_split(
        processed_demos, 
        test_size=0.2,
        random_state=42
    )
    
    # Train model on training set
    print(f"\nTraining human model on {len(train_demos)} demonstrations...")
    trainer = HumanModelTrainer()
    trainer.train(train_demos)
    
    # Evaluate on test set
    evaluate_predictions(trainer, test_demos, results_dir)
    
    # Save training results
    results_path = os.path.join(results_dir, "data", "training_results.json")
    trainer.save_results(results_path)
    
    print(f"\nTraining complete. Results saved to {results_path}")
    return trainer

if __name__ == "__main__":
    trained_model = main()
