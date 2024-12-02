# scenarios/scenario_generator.py

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Environment:
    """Environment specifications"""
    width: float = 10.0  # 10m width
    height: float = 10.0  # 10m height

@dataclass
class AgentConfig:
    """Agent (human/robot) configuration"""
    start_pos: np.ndarray  # [x, y]
    goal_pos: np.ndarray   # [x, y]
    initial_heading: float # θ
    initial_speed: float   # v

class ScenarioGenerator:
    """Generates random interaction scenarios"""
    def __init__(self, env: Environment = Environment()):
        self.env = env
        self.robot_goal = np.array([10.0, 10.0])  # Fixed robot goal
        
    def generate_random_scenario(self) -> Dict:
        """Generate a random scenario"""
        # Fixed robot start
        robot_config = AgentConfig(
            start_pos=np.array([0.0, 0.0]),
            goal_pos=self.robot_goal,
            initial_heading=np.pi/4,  # 45 degrees towards goal
            initial_speed=0.0
        )
        
        # Random human configuration
        human_start = np.random.uniform(
            low=[0.0, 0.0],
            high=[self.env.width, self.env.height],
            size=2
        )
        human_goal = np.random.uniform(
            low=[0.0, 0.0],
            high=[self.env.width, self.env.height],
            size=2
        )
        
        # Initial heading towards goal
        direction = human_goal - human_start
        initial_heading = np.arctan2(direction[1], direction[0])
        
        human_config = AgentConfig(
            start_pos=human_start,
            goal_pos=human_goal,
            initial_heading=initial_heading,
            initial_speed=0.0
        )
        
        # Random internal states
        attention = np.random.uniform(0.2, 0.9)
        style = np.random.uniform(0.2, 0.9)
        
        return {
            'robot': robot_config,
            'human': human_config,
            'attention_profile': {
                'type': 'constant',
                'value': attention
            },
            'style_profile': {
                'type': 'constant',
                'value': style
            }
        }

    def generate_scenarios(self, n: int) -> List[Dict]:
        """Generate n random scenarios"""
        return [self.generate_random_scenario() for _ in range(n)]