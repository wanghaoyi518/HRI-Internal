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
        # Base reward weights [lane, boundary, road]
        self.theta_base = np.array([
            1.0,    # θ₀: Lane following
            -50.0,  # θ₁: Boundary avoidance
            10.0    # θ₂: Road following
        ])

        # Style reward weights [control, speed]
        self.theta_style = np.array([
            100.0,  # θ₃: Control smoothness
            10.0    # θ₄: Speed maintenance
        ])

        # Car avoidance weights based on attention
        self.theta_car_avoidance = {
            'attentive': -70.0,      # Strong avoidance
            'distracted': -20.0,     # Weak avoidance
            'very_distracted': 0.0   # No avoidance
        }

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
            # Get attention level
            attention = internal_state.get('attention', 0.5)
            
            # Compute base features
            r = self.theta_base[0] * self._lane_following(state)       # Stay in lanes
            r += self.theta_base[1] * self._boundary_avoidance(state)  # Avoid boundaries
            r += self.theta_base[2] * self._road_following(state)      # Stay on road
            
            # Compute style features
            r += self.theta_style[0] * self._control_smoothness(action) # Smooth control
            r += self.theta_style[1] * self._speed_maintenance(state)   # Maintain speed
            
            # Add car avoidance based on attention
            car_weight = self._get_car_avoidance_weight(attention)
            r += car_weight * self._car_avoidance(state, robot_state)   # Avoid cars
            
            return float(r)
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

    def _get_car_avoidance_weight(self, attention: float) -> float:
        """Get car avoidance weight based on attention level"""
        if attention > 0.7:
            return self.theta_car_avoidance['attentive']
        elif attention > 0.3:
            return self.theta_car_avoidance['distracted']
        else:
            return self.theta_car_avoidance['very_distracted']

    # Feature computation methods
    def _lane_following(self, state):
        """Lane following feature"""
        return -np.square(state[1] - 5.0)  # Lane center at y=5

    def _boundary_avoidance(self, state):
        """Boundary avoidance feature"""
        y = state[1]  # Extract y coordinate from state
        return -(np.exp(-0.5 * y**2) + np.exp(-0.5 * (y-10.0)**2))  # Boundaries at y=0,10

    def _road_following(self, state):
        """Road following feature"""
        return np.exp(-0.5 * ((state[0]-5.0)**2 + (state[1]-5.0)**2) / 100.0)

    def _control_smoothness(self, action):
        """Control smoothness feature"""
        return -np.sum(np.square(action))  # Penalize large actions

    def _speed_maintenance(self, state):
        """Speed maintenance feature"""
        return -np.square(state[3] - 1.0)  # Target speed = 1.0

    def _car_avoidance(self, state, robot_state):
        """Car avoidance feature"""
        dist = np.linalg.norm(state[:2] - robot_state[:2])
        return np.exp(-0.5 * dist**2)