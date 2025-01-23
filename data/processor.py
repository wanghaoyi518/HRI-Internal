# data/processor.py

import numpy as np
from typing import Dict, List
from models.reward import RewardFunction
from data.collector import Demonstration

class DataProcessor:
    def __init__(self):
        self.reward_function = RewardFunction()
        
    def process_demonstration(self, demo: Demonstration) -> List[Dict]:
        """Process raw demonstration into timestep dictionaries"""
        processed_data = []
        
        for t in range(len(demo.states)):
            state = demo.states[t]
            action = demo.actions[t]
            robot_state = demo.robot_states[t]
            internal_state = demo.internal_states[t]
            
            # Skip if no action available
            if action is None:
                continue
                
            # Compute base rewards
            base_reward = (
                self.reward_function.theta_base[0] * self.reward_function._lane_following(state) +
                self.reward_function.theta_base[1] * self.reward_function._boundary_avoidance(state) +
                self.reward_function.theta_base[2] * self.reward_function._road_following(state)
            )
            
            # Compute style rewards
            style_reward = (
                self.reward_function.theta_style[0] * self.reward_function._control_smoothness(action) +
                self.reward_function.theta_style[1] * self.reward_function._speed_maintenance(state)
            )
            
            # Compute attention-based car avoidance reward
            attention = internal_state.get('attention', 0.5)
            car_weight = self.reward_function._get_car_avoidance_weight(attention)
            attention_reward = car_weight * self.reward_function._car_avoidance(state, robot_state)
            
            processed_data.append({
                'state': state,
                'action': action,
                'robot_state': robot_state,
                'internal_state': internal_state,
                'base_reward': base_reward,
                'attention_reward': attention_reward,
                'style_reward': style_reward
            })
            
        return processed_data