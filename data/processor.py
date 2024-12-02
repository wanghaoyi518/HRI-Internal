# data/processor.py

import numpy as np
from typing import Dict, List
from models.reward import RewardFunction
from data.collector import Demonstration

class DataProcessor:
    def __init__(self):
        self.reward_function = RewardFunction()
        
    def process_demonstration(self, demo: Demonstration) -> List[Dict]:
        """
        Process raw demonstration into timestep dictionaries
        """
        processed_data = []
        
        for t in range(len(demo.states)):
            state = demo.states[t]
            action = demo.actions[t]
            robot_state = demo.robot_states[t]
            internal_state = demo.internal_states[t]
            
            # Skip if no action available
            if action is None:
                continue
                
            # Compute reward components
            base_reward = self.reward_function._compute_base_reward(
                state, action, robot_state
            )
            attention_reward = self.reward_function._compute_attention_reward(
                state, action, robot_state
            )
            style_reward = self.reward_function._compute_style_reward(
                state, action, robot_state
            )
            
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