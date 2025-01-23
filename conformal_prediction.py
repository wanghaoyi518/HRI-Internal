import numpy as np
from typing import Dict, List, Tuple, Optional
from data.collector import Demonstration
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class TrajectoryPredictor:
    """Predicts future trajectories using trained human model"""
    
    def __init__(self, human_model, dt: float = 0.1):
        """
        Args:
            human_model: Trained HumanModelTrainer instance
            dt: Time step for prediction
        """
        self.human_model = human_model
        self.dt = dt
        
        # Pre-fit polynomial features with dummy data once
        dummy_features = np.zeros((1, 11))  # 11 features
        self.human_model.base_model.named_steps['poly'].fit(dummy_features)
        self.human_model.att_model.named_steps['poly'].fit(dummy_features)
        self.human_model.style_model.named_steps['poly'].fit(dummy_features)
        
        # Get transformed feature size for scaler initialization
        poly_features = self.human_model.base_model.named_steps['poly'].transform(dummy_features)
        
        # Initialize scalers with dummy data
        self.human_model.base_model.named_steps['scaler'].fit(poly_features)
        self.human_model.att_model.named_steps['scaler'].fit(poly_features)
        self.human_model.style_model.named_steps['scaler'].fit(poly_features)
        
    def predict(self, state: np.ndarray, 
                action: np.ndarray,
                robot_state: np.ndarray,
                horizon: int) -> List[np.ndarray]:
        """
        Predict trajectory using trained human model
        
        Args:
            state: Current human state [x, y, θ, v]
            action: Current human action [steering, acceleration]
            robot_state: Current robot state [x, y, θ, v]
            horizon: Number of steps to predict
            
        Returns:
            List of predicted states
        """
        trajectory = []
        current_state = state.copy()
        current_robot_state = robot_state.copy()
        current_action = action.copy()  # Keep initial action
        
        for _ in range(horizon):
            # Extract features exactly as human model does
            features = self._extract_features(current_state, current_action, current_robot_state)
            features_reshaped = features.reshape(1, -1)
            
            # Use human model's predict_reward_weights method
            rewards = self.human_model.predict_reward_weights(
                current_state,
                current_action,
                current_robot_state
            )
            # Use base reward weights to predict next action
            predicted_action = np.array([
                np.clip(rewards['base'], -1.0, 1.0),  # steering
                np.clip(rewards['attention'], -1.0, 1.0)  # acceleration
            ])
            
            current_action = predicted_action  # Update action for next step
            
            # Update state using original dynamics
            x, y, theta, v = current_state
            steering, acceleration = predicted_action
            
            # State update using same equations as collector
            dx = v * np.cos(theta)  # Position updates use raw velocity
            dy = v * np.sin(theta)
            dtheta = steering      # Direct steering control
            dv = acceleration * self.dt  # Only velocity needs dt
            
            # Update state
            new_state = np.array([
                x + dx * self.dt,  # Scale position changes by dt
                y + dy * self.dt,
                theta + dtheta,    # Heading changes directly
                np.clip(v + dv, 0, 2.0)  # Limit speed
            ])
            
            trajectory.append(new_state)
            current_state = new_state
            
            # Simple robot state propagation for next iteration
            # In practice, this would come from the robot's motion plan
            rx, ry, rtheta, rv = current_robot_state
            rdx = rv * np.cos(rtheta) * self.dt
            rdy = rv * np.sin(rtheta) * self.dt
            current_robot_state = np.array([rx + rdx, ry + rdy, rtheta, rv])
            
        return trajectory

    def _extract_features(self, state, action, robot_state):
        """Extract features in the same way as HumanModelTrainer"""
        return np.concatenate([
            state,              # [x, y, θ, v]
            action,            # [steering, acceleration]
            robot_state,       # [x, y, θ, v]
            [np.linalg.norm(state[:2] - robot_state[:2])]  # Distance
        ])

class ConformalPredictor:
    """
    Implements conformal prediction for safe trajectory forecasting
    Provides uncertainty bounds with statistical guarantees
    """
    
    def __init__(self, human_model,
                 significance_level: float = 0.1, 
                 prediction_horizons: List[int] = None):
        """
        Initialize Conformal Predictor
        
        Args:
            human_model: Trained HumanModelTrainer instance
            significance_level: Desired miscoverage level (default 0.1 = 90% coverage)
            prediction_horizons: List of timesteps to predict (default 1-10)
        """
        self.significance_level = significance_level
        self.prediction_horizons = prediction_horizons or list(range(1, 11))
        self.predictor = TrajectoryPredictor(human_model)
        
        # Storage for calibration
        self.calibration_scores = {h: [] for h in self.prediction_horizons}
        self.calibrated = False
    
    def calibrate(self, demonstrations: List[Demonstration]):
        """
        Calibrate predictor using demonstration dataset
        
        Args:
            demonstrations: List of demonstrations for calibration
        """
        print("Calibrating conformal predictor...")
        
        for demo_idx, demo in enumerate(demonstrations):
            # Skip demonstrations with no actions or states or robot_states
            if not demo.actions or not demo.states or not demo.robot_states:
                continue
                
            # For each timestep that allows for longest horizon prediction
            max_horizon = max(self.prediction_horizons)
            for t in range(len(demo.states) - max_horizon):
                current_state = demo.states[t]
                current_action = demo.actions[t]
                current_robot_state = demo.robot_states[t]
                
                # For each prediction horizon
                for h in self.prediction_horizons:
                    # Skip if we don't have enough future states
                    if t + h >= len(demo.states):
                        continue
                        
                    # Get predicted trajectory with all required arguments
                    predicted_trajectory = self.predictor.predict(
                        current_state,
                        current_action,
                        current_robot_state,
                        h
                    )
                    
                    # Get actual future states
                    actual_trajectory = demo.states[t+1:t+h+1]
                    
                    # Compute nonconformity score
                    score = self._compute_nonconformity(
                        predicted_trajectory,
                        actual_trajectory
                    )
                    
                    self.calibration_scores[h].append(score)
            
            if (demo_idx + 1) % 10 == 0:
                print(f"Processed {demo_idx + 1} demonstrations")
        
        # Compute quantiles for each horizon
        self.quantiles = {
            h: np.quantile(scores, 1 - self.significance_level)
            for h, scores in self.calibration_scores.items()
            if scores  # Only compute if we have scores
        }
        
        self.calibrated = True
        print("Calibration complete.")
        
    def predict(self, state: np.ndarray,
                action: np.ndarray,
                robot_state: np.ndarray,
                horizon: int) -> Dict:
        """
        Generate prediction with uncertainty regions
        
        Args:
            state: Current state [x, y, θ, v]
            action: Current action [steering, acceleration]
            robot_state: Current robot state [x, y, θ, v]
            horizon: Prediction horizon steps
            
        Returns:
            Dict containing predicted trajectory and uncertainty regions
        """
        if not self.calibrated:
            raise RuntimeError("Predictor must be calibrated before prediction")
        
        if horizon not in self.prediction_horizons:
            raise ValueError(f"Horizon {horizon} not in calibrated horizons")
            
        # Get predicted trajectory with robot state
        predicted_trajectory = self.predictor.predict(
            state, action, robot_state, horizon
        )
        
        # Get calibrated quantile for this horizon
        quantile = self.quantiles[horizon]
        
        # Generate prediction regions
        prediction_regions = []
        for predicted_state in predicted_trajectory:
            region = self._generate_prediction_region(
                predicted_state,
                quantile
            )
            prediction_regions.append(region)
        
        return {
            'trajectory': predicted_trajectory,
            'regions': prediction_regions,
            'quantile': quantile
        }
    
    def _compute_nonconformity(self, 
                             predicted_trajectory: List[np.ndarray],
                             actual_trajectory: List[np.ndarray]) -> float:
        """
        Compute nonconformity score between predicted and actual trajectories
        
        Args:
            predicted_trajectory: List of predicted states
            actual_trajectory: List of actual states
            
        Returns:
            Nonconformity score
        """
        # Ensure same length
        min_len = min(len(predicted_trajectory), len(actual_trajectory))
        predicted_trajectory = predicted_trajectory[:min_len]
        actual_trajectory = actual_trajectory[:min_len]
        
        # Compute position errors
        position_errors = [
            np.linalg.norm(pred[:2] - actual[:2])
            for pred, actual in zip(predicted_trajectory, actual_trajectory)
        ]
        
        # Return maximum error as nonconformity score
        return max(position_errors) if position_errors else 0.0
    
    def _generate_prediction_region(self,
                                  center_state: np.ndarray,
                                  radius: float,
                                  n_points: int = 32) -> np.ndarray:
        """
        Generate points forming prediction region boundary
        
        Args:
            center_state: Center state of prediction region
            radius: Radius based on calibrated quantile
            n_points: Number of boundary points
            
        Returns:
            Array of boundary points forming convex hull
        """
        angles = np.linspace(0, 2*np.pi, n_points)
        circle_points = np.zeros((n_points, 2))
        
        # Generate circle points for position components
        circle_points[:, 0] = center_state[0] + radius * np.cos(angles)
        circle_points[:, 1] = center_state[1] + radius * np.sin(angles)
        
        return circle_points
    
    def visualize_prediction(self, 
                           state: np.ndarray,
                           action: np.ndarray,
                           robot_state: np.ndarray,  # Added robot_state parameter
                           horizon: int,
                           ax: Optional[plt.Axes] = None):
        """
        Visualize prediction with uncertainty regions
        
        Args:
            state: Current state
            action: Current action
            robot_state: Current robot state
            horizon: Prediction horizon
            ax: Optional matplotlib axes for plotting
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        
        # Make prediction with robot state
        prediction = self.predict(state, action, robot_state, horizon)
        
        # Plot predicted trajectory
        trajectory = prediction['trajectory']
        trajectory_points = np.array([s[:2] for s in trajectory])
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                'b-', label='Predicted Trajectory')
        
        # Plot prediction regions
        for region in prediction['regions']:
            ax.fill(region[:, 0], region[:, 1], 
                   alpha=0.2, color='blue',
                   label='Prediction Region')
        
        # Plot current state
        ax.plot(state[0], state[1], 'ko', label='Current State')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Trajectory Prediction with {horizon} Step Horizon')
        ax.legend()
        ax.grid(True)
