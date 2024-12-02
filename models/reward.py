# models/reward.py

import numpy as np
from typing import Dict, List, Optional

class RewardFunction:
    """
    Enhanced reward function that depends on internal state
    R(s,a,φ) = θ(φ)ᵀψ(s,a)
    
    Features are split into:
    - Base features (core driving behavior)
    - Attention features (responsiveness and safety)
    - Style features (aggression and preferences)
    """
    def __init__(self):
        # Base feature weights (lane keeping, collision avoidance, speed, control, distance)
        self.theta_base = np.array([
            1.0,    # Lane keeping
            -50.0,  # Collision avoidance
            10.0,   # Speed matching
            10.0,   # Control smoothness
            -60.0   # Distance keeping
        ])
        
        # Attention feature weights
        self.theta_attention = np.array([
            15.0,   # Reaction time
            -80.0,  # Safety margin
            5.0,    # Prediction error
            10.0,   # Response time
            8.0,    # Path smoothness
            12.0    # Anticipatory behavior
        ])
        
        # Style feature weights
        self.theta_style = np.array([
            15.0,   # Acceleration profile
            -30.0,  # Desired gap
            8.0,    # Speed preference
            10.0,   # Lane change aggressiveness
            5.0     # Goal orientation
        ])
        
        # Safety parameters
        self.min_safe_distance = 0.5
        self.comfortable_distance = 2.0
        self.max_comfortable_speed = 2.0
        
    def compute_reward(self, state: np.ndarray, 
                      action: Optional[np.ndarray], 
                      robot_state: np.ndarray, 
                      internal_state: Dict[str, float]) -> float:
        """
        Compute total reward value given state and internal state
        
        Args:
            state: Human state [x, y, θ, v]
            action: Human action [steering, acceleration] or None
            robot_state: Robot state [x, y, θ, v]
            internal_state: Dict with 'attention' and 'style' values
            
        Returns:
            Combined reward value
        """
        if action is None:
            return 0.0
            
        try:
            # Compute basic reward components
            base_reward = self._compute_base_reward(state, action, robot_state)
            
            # Scale attention and style rewards by internal state
            attention = internal_state.get('attention', 0.5)
            style = internal_state.get('style', 0.5)
            
            attention_reward = attention * self._compute_attention_reward(
                state, action, robot_state
            )
            style_reward = style * self._compute_style_reward(
                state, action, robot_state
            )
            
            # Combine rewards with safety constraints
            distance_to_robot = np.linalg.norm(state[:2] - robot_state[:2])
            if distance_to_robot < self.min_safe_distance:
                return float('-inf')  # Strong safety constraint
                
            return float(base_reward + attention_reward + style_reward)
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

    def _compute_base_reward(self, state, action, robot_state):
        """Compute base reward component"""
        features = self._compute_base_features(state, action, robot_state)
        return np.dot(self.theta_base, features)

    def _compute_attention_reward(self, state, action, robot_state):
        """Compute attention-dependent reward component"""
        features = self._compute_attention_features(state, action, robot_state)
        return np.dot(self.theta_attention, features)

    def _compute_style_reward(self, state, action, robot_state):
        """Compute style-dependent reward component"""
        features = self._compute_style_features(state, action, robot_state)
        return np.dot(self.theta_style, features)
    
    def _compute_base_features(self, state, action, robot_state):
        """
        Compute basic driving features
        Returns: [lane_dev, collision_risk, speed_match, control, distance]
        """
        return np.array([
            self._lane_deviation(state),
            self._collision_risk(state, robot_state),
            self._speed_matching(state),
            self._control_effort(action),
            self._distance_keeping(state, robot_state)
        ])
    
    def _compute_attention_features(self, state, action, robot_state):
        """
        Compute attention-specific features
        Returns: [reaction, safety, prediction, response, smoothness, anticipation]
        """
        return np.array([
            self._reaction_feature(state, action, robot_state),
            self._safety_margin(state, robot_state),
            self._prediction_error(state, robot_state),
            self._response_time_feature(state, action),
            self._path_smoothness(state, action),
            self._anticipatory_behavior(state, robot_state)
        ])
    
    def _compute_style_features(self, state, action, robot_state):
        """
        Compute style-specific features
        Returns: [accel, gap, speed, lane_change, goal]
        """
        return np.array([
            self._acceleration_profile(state, action),
            self._desired_gap(state, robot_state),
            self._speed_preference(state),
            self._lane_change_aggressiveness(state, action),
            self._goal_orientation(state, robot_state)
        ])
    
    # Base feature computations
    def _lane_deviation(self, state):
        """Compute lane keeping reward"""
        return -np.square(state[0])  # Distance from lane center
    
    def _collision_risk(self, state, robot_state):
        """Compute collision avoidance reward"""
        dist = np.linalg.norm(state[:2] - robot_state[:2])
        return -np.exp(-dist/self.min_safe_distance)
    
    def _speed_matching(self, state):
        """Compute speed matching reward"""
        target_speed = 1.0
        return -np.square(state[3] - target_speed)
    
    def _control_effort(self, action):
        """Compute control effort penalty"""
        return -np.sum(np.square(action))
    
    def _distance_keeping(self, state, robot_state):
        """Compute distance keeping reward"""
        dist = np.linalg.norm(state[:2] - robot_state[:2])
        return -np.square(dist - self.comfortable_distance)
    
    # Attention feature computations
    def _reaction_feature(self, state, action, robot_state):
        """Compute reaction to robot feature"""
        dist = np.linalg.norm(state[:2] - robot_state[:2])
        vel_towards_robot = np.dot(
            state[:2] - robot_state[:2],
            state[3] * np.array([np.cos(state[2]), np.sin(state[2])])
        )
        return -vel_towards_robot if dist < self.comfortable_distance else 0.0
    
    def _safety_margin(self, state, robot_state):
        """Compute safety margin feature"""
        dist = max(self.min_safe_distance, np.linalg.norm(state[:2] - robot_state[:2]))
        return -1.0/dist
    
    def _response_time_feature(self, state, action):
        """Measure responsiveness to changing conditions"""
        # Penalize delayed acceleration changes
        return -np.abs(action[1])
    
    def _path_smoothness(self, state, action):
        """Measure smoothness of trajectory"""
        # Penalize sharp steering
        return -np.square(action[0])
    
    def _prediction_error(self, state, robot_state):
        """Compute prediction error feature"""
        # Simple prediction based on current velocities
        rel_vel = state[3] - robot_state[3]
        return -np.abs(rel_vel)
    
    def _anticipatory_behavior(self, state, robot_state):
        """Measure anticipatory behavior"""
        rel_pos = state[:2] - robot_state[:2]
        rel_vel = state[3] * np.array([np.cos(state[2]), np.sin(state[2])])
        ttc = np.linalg.norm(rel_pos) / (np.linalg.norm(rel_vel) + 1e-6)
        return -1.0 / (ttc + 1e-6)
    
    # Style feature computations
    def _acceleration_profile(self, state, action):
        """Compute acceleration profile feature"""
        return -np.square(action[1])  # Penalize high accelerations
    
    def _desired_gap(self, state, robot_state):
        """Compute desired gap feature"""
        dist = np.linalg.norm(state[:2] - robot_state[:2])
        return -np.square(dist - self.comfortable_distance)
    
    def _speed_preference(self, state):
        """Compute speed preference feature"""
        return -np.square(state[3] - self.max_comfortable_speed)
    
    def _lane_change_aggressiveness(self, state, action):
        """Compute lane change aggressiveness"""
        return -np.abs(action[0])  # Penalize aggressive steering
    
    def _goal_orientation(self, state, robot_state):
        """Compute goal orientation feature"""
        heading_diff = np.abs(state[2] - np.arctan2(
            robot_state[1] - state[1],
            robot_state[0] - state[0]
        ))
        return -np.square(heading_diff)