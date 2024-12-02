# main.py

import os
import numpy as np
from models.human_state import HumanState
from models.reward import RewardFunction
from models.training import HumanModelTrainer
from data.collector import DataCollector
from data.processor import DataProcessor
from scenarios.scenario_generator import ScenarioGenerator, Environment
import json
from datetime import datetime
import matplotlib.pyplot as plt

def setup_results_directory():
    """Create results directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    plots_dir = os.path.join(results_dir, "plots")
    data_dir = os.path.join(results_dir, "data")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return results_dir, plots_dir, data_dir

def main():
    # Setup directories
    results_dir, plots_dir, data_dir = setup_results_directory()
    
    # Initialize components
    collector = DataCollector()
    processor = DataProcessor()
    trainer = HumanModelTrainer()
    scenario_gen = ScenarioGenerator(Environment(10.0, 10.0))
    
    # Generate scenarios
    n_scenarios = 100  # Changed to 100 scenarios
    scenarios = scenario_gen.generate_scenarios(n_scenarios)
    
    # Randomly select 10 scenarios to plot
    scenarios_to_plot = np.random.choice(n_scenarios, 10, replace=False)
    
    # Save scenario configurations
    with open(os.path.join(data_dir, 'scenarios.json'), 'w') as f:
        serializable_scenarios = []
        for scenario in scenarios:
            scenario_dict = {
                'robot': {
                    'start_pos': scenario['robot'].start_pos.tolist(),
                    'goal_pos': scenario['robot'].goal_pos.tolist(),
                    'initial_heading': float(scenario['robot'].initial_heading),
                    'initial_speed': float(scenario['robot'].initial_speed)
                },
                'human': {
                    'start_pos': scenario['human'].start_pos.tolist(),
                    'goal_pos': scenario['human'].goal_pos.tolist(),
                    'initial_heading': float(scenario['human'].initial_heading),
                    'initial_speed': float(scenario['human'].initial_speed)
                },
                'attention_profile': scenario['attention_profile'],
                'style_profile': scenario['style_profile']
            }
            serializable_scenarios.append(scenario_dict)
        json.dump(serializable_scenarios, f, indent=2)
    
    # Collect demonstrations
    demonstrations = []
    for i, scenario in enumerate(scenarios):
        if (i+1) % 10 == 0:
            print(f"\nCollecting demonstration for scenario {i+1}/{n_scenarios}")
        
        # Initialize states
        human_state = HumanState(np.concatenate([
            scenario['human'].start_pos,
            [scenario['human'].initial_heading],
            [scenario['human'].initial_speed]
        ]))
        
        robot_state = np.concatenate([
            scenario['robot'].start_pos,
            [scenario['robot'].initial_heading],
            [scenario['robot'].initial_speed]
        ])
        
        # Collect demonstration
        demo = collector.collect_demonstration(
            human_state=human_state,
            robot_state=robot_state,
            human_config=scenario['human'],
            robot_config=scenario['robot'],
            attention_profile=scenario['attention_profile'],
            style_profile=scenario['style_profile'],
            duration=30.0
        )
        
        demonstrations.append(demo)
    
    # Plot selected scenarios
    for i in scenarios_to_plot:
        scenario = scenarios[i]
        demo = demonstrations[i]
        collector.save_demonstration_plot(
            demo=demo,
            title=f"Scenario {i+1}\nAttention: {scenario['attention_profile']['value']:.2f}, Style: {scenario['style_profile']['value']:.2f}",
            save_path=os.path.join(plots_dir, f'trajectory_{i+1}.png')
        )
    
    print("number of scenarios collected in the dataset", n_scenarios)



    # Process and train
    print("\nProcessing demonstrations...")
    processed_demos = []
    for demo in demonstrations:
        processed_demo = processor.process_demonstration(demo)
        processed_demos.append(processed_demo)
    
    print("\nTraining human model...")
    trainer.train(processed_demos)
    
    # Save training results
    trainer.save_results(os.path.join(data_dir, 'training_results.json'))
    
    print(f"\nExperiment complete. Results saved in {results_dir}")
    return trainer, demonstrations

if __name__ == "__main__":
    trained_model, demonstrations = main()