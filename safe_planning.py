import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from models.human_state import HumanState
from train_model import HumanModelTrainer
from conformal_prediction import TrajectoryPredictor
from scipy.optimize import minimize
import time
import json
import os

class SafePlanner:
    def __init__(self, 
                 human_model,
                 horizon: int = 10,
                 dt: float = 0.1,
                 safety_margin: float = 1.2):
        """Initialize Safe Planner"""
        self.human_model = human_model
        self.horizon = horizon
        self.dt = dt
        self.safety_margin = safety_margin
        
        # Planning constraints
        self.max_speed = 2.0
        self.max_steering = 1.0
        self.max_acceleration = 1.0
        
        # Robot task weights
        self.w_goal = 1.0  # Goal reaching
        self.w_safety = 10.0  # Safety constraint
        self.w_control = 0.1  # Control smoothness
        
        # Initialize predictor and conformal bounds
        self.predictor = TrajectoryPredictor(human_model, dt=dt)
        self.conformal_bounds = None

    def load_calibrated_bounds(self, bounds_file: str):
        """Load pre-computed conformal prediction bounds"""
        with open(bounds_file, 'r') as f:
            bounds_data = json.load(f)
            self.conformal_bounds = bounds_data['quantiles']
        print(f"Loaded conformal bounds from {bounds_file}")

    def _get_prediction_set(self, human_state, human_action, robot_state, horizon):
        """Generate trajectory prediction with uncertainty regions"""
        # Get trajectory prediction
        trajectory = self.predictor.predict(
            human_state,
            human_action,
            robot_state,
            horizon
        )

        # Create uncertainty regions
        regions = []
        radius = float(self.conformal_bounds[str(horizon)]) * self.safety_margin

        # Generate circular regions for each predicted state
        for state in trajectory:
            angles = np.linspace(0, 2*np.pi, 32)
            circle = np.zeros((32, 2))
            circle[:, 0] = state[0] + radius * np.cos(angles)
            circle[:, 1] = state[1] + radius * np.sin(angles)
            regions.append(circle)

        return {
            'trajectory': trajectory,
            'regions': regions
        }

    def plan(self, robot_state: np.ndarray,
             human_state: np.ndarray,
             human_action: np.ndarray,
             robot_goal: np.ndarray,
             verbose: bool = False) -> Dict:
        """Generate safe plan using MPC optimization"""
        # Get human prediction with uncertainty
        prediction = self._get_prediction_set(
            human_state, 
            human_action,
            robot_state, 
            self.horizon
        )
        
        # Initial guess: straight line to goal
        initial_controls = np.zeros((self.horizon, 2))  # [steering, acceleration]
        
        # Simple proportional control for initial guess
        dx = robot_goal[0] - robot_state[0]
        dy = robot_goal[1] - robot_state[1]
        desired_theta = np.arctan2(dy, dx)
        theta_error = desired_theta - robot_state[2]
        
        # Normalize angle
        while theta_error > np.pi: theta_error -= 2*np.pi
        while theta_error < -np.pi: theta_error += 2*np.pi
        
        # Set constant steering and acceleration
        initial_controls[:, 0] = 0.5 * theta_error  # Proportional steering
        initial_controls[:, 1] = 0.5  # Constant acceleration
        
        # Setup optimization problem
        result = minimize(
            fun=lambda u: self._compute_cost(u, robot_state, robot_goal),
            x0=initial_controls.flatten(),
            method='SLSQP',
            bounds=self._get_control_bounds(),
            constraints=self._get_safety_constraints(
                robot_state, prediction['regions']
            ),
            options={'maxiter': 100}
        )
        
        if verbose:
            print(f"Optimization success: {result.success}")
            print(f"Final cost: {result.fun:.3f}")
        
        # Return plan
        opt_controls = result.x.reshape(-1, 2)
        opt_trajectory = self._roll_out_trajectory(robot_state, opt_controls)
        
        return {
            'success': result.success,
            'controls': opt_controls,
            'trajectory': opt_trajectory,
            'cost': result.fun,
            'human_prediction': prediction
        }

    def _compute_cost(self, controls: np.ndarray, robot_state: np.ndarray, goal: np.ndarray) -> float:
        """Compute optimization cost"""
        controls = controls.reshape(-1, 2)
        trajectory = self._roll_out_trajectory(robot_state, controls)
        
        # Distance to goal
        final_pos = trajectory[-1][:2]
        goal_cost = self.w_goal * np.linalg.norm(final_pos - goal)
        
        # Control smoothness
        control_cost = self.w_control * np.sum(controls**2)
        
        return goal_cost + control_cost

    def _get_control_bounds(self) -> List:
        """Get bounds for control inputs"""
        bounds = []
        for _ in range(self.horizon):
            bounds.append((-self.max_steering, self.max_steering))
            bounds.append((-self.max_acceleration, self.max_acceleration))
        return bounds

    def _get_safety_constraints(self, robot_state: np.ndarray, prediction_regions: List) -> List:
        """Generate safety constraints from prediction regions"""
        constraints = []
        
        def safety_constraint(controls, t, region):
            trajectory = self._roll_out_trajectory(robot_state, controls.reshape(-1, 2))
            robot_pos = trajectory[t][:2]
            distances = np.linalg.norm(region - robot_pos, axis=1)
            min_dist = np.min(distances)
            return min_dist - self.safety_margin
        
        for t, region in enumerate(prediction_regions):
            constraints.append({
                'type': 'ineq',
                'fun': lambda u, t=t, r=region: safety_constraint(u, t, r)
            })
            
        return constraints

    def _roll_out_trajectory(self, state: np.ndarray, controls: np.ndarray) -> List[np.ndarray]:
        """Roll out trajectory using controls"""
        trajectory = [state.copy()]
        current_state = state.copy()
        
        for control in controls:
            x, y, theta, v = current_state
            steering, acceleration = control
            
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
            dtheta = v * steering * self.dt
            dv = acceleration * self.dt
            
            new_state = np.array([
                x + dx,
                y + dy,
                theta + dtheta,
                np.clip(v + dv, 0, self.max_speed)
            ])
            
            trajectory.append(new_state)
            current_state = new_state
            
        return trajectory

    def visualize_plan(self,
                      robot_state: np.ndarray,
                      human_state: np.ndarray,
                      human_action: np.ndarray,
                      robot_goal: np.ndarray,
                      ax: Optional[plt.Axes] = None):
        """Visualize planned trajectory with predictions"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            
        # Get plan and predictions
        plan = self.plan(robot_state, human_state, human_action, robot_goal)
        prediction = plan['human_prediction']
        
        # Plot robot trajectory
        robot_traj = np.array([s[:2] for s in plan['trajectory']])
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], 
                'r-', label='Planned Robot Path')
        
        # Plot robot position
        ax.plot(robot_state[0], robot_state[1], 'rs', label='Robot')
        
        # Plot robot goal
        ax.plot(robot_goal[0], robot_goal[1], 'r*', label='Robot Goal')
        
        # Plot human trajectory prediction
        human_traj = np.array([s[:2] for s in prediction['trajectory']])
        ax.plot(human_traj[:, 0], human_traj[:, 1], 
                'b-', label='Predicted Human Path')
        
        # Plot uncertainty regions
        for region in prediction['regions']:
            ax.fill(region[:, 0], region[:, 1],
                   alpha=0.2, color='blue',
                   label='Human Position Uncertainty')
                   
        # Plot human position
        ax.plot(human_state[0], human_state[1], 'bs', label='Human')
        
        # Clean up plot
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

def get_latest_dir():
    results_dirs = sorted(os.listdir("results"))
    if not results_dirs:
        raise ValueError("No results directory found")
    return os.path.join("results", results_dirs[-1])

def load_model():
    latest_dir = get_latest_dir()
    model_file = os.path.join(latest_dir, "data", "training_results.json")
    
    if not os.path.exists(model_file):
        raise ValueError(f"No trained model found at {model_file}")
        
    trainer = HumanModelTrainer()
    trainer.load_model(model_file)
    return trainer

def test_safe_planning():
    """Test complete safe planning workflow"""
    print("Loading model and conformal bounds...")
    trainer = load_model()
    
    # Load pre-computed conformal bounds
    latest_dir = get_latest_dir()
    bounds_file = os.path.join(latest_dir, "data", "conformal_bounds.json")
    if not os.path.exists(bounds_file):
        raise ValueError("Run calibrate_conformal.py first")

    # Initialize planner with human model
    planner = SafePlanner(
        human_model=trainer,
        horizon=10,
        dt=0.1,
        safety_margin=1.2
    )
    
    # Load calibrated bounds
    planner.load_calibrated_bounds(bounds_file)
    
    # Generate test scenario
    print("\nGenerating test scenario...")
    
    # Robot approaching intersection from left
    robot_state = np.array([2.0, 5.0, 0.0, 0.8])  # [x,y,θ,v]
    robot_goal = np.array([8.0, 5.0])
    
    # Human approaching from bottom with varying behaviors
    scenarios = [
        {
            'name': 'Attentive-Conservative',
            'state': np.array([5.0, 2.0, np.pi/2, 0.3]),  # Slower speed
            'action': np.array([0.0, 0.3])  # Gentle acceleration
        },
        {
            'name': 'Distracted-Aggressive', 
            'state': np.array([5.0, 1.0, np.pi/2, 1.0]),  # Higher speed
            'action': np.array([0.0, 0.8])  # Stronger acceleration
        }
    ]
    
    # Test each scenario
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    for ax, scenario in zip(axes, scenarios):
        print(f"\nTesting {scenario['name']} scenario...")
        
        # Plan safe trajectory
        t_start = time.time()
        plan = planner.plan(
            robot_state=robot_state,
            human_state=scenario['state'],
            human_action=scenario['action'],
            robot_goal=robot_goal,
            verbose=True
        )
        t_plan = time.time() - t_start
        
        # Visualize results
        planner.visualize_plan(
            robot_state=robot_state,
            human_state=scenario['state'],
            human_action=scenario['action'],
            robot_goal=robot_goal,
            ax=ax
        )
        
        ax.set_title(f"{scenario['name']}\nPlanning time: {t_plan:.3f}s")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    
    plt.tight_layout()
    
    # Save results
    plots_dir = os.path.join(latest_dir, "plots")
    save_path = os.path.join(plots_dir, "safe_planning_results.png")
    plt.savefig(save_path)
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    test_safe_planning()