# # scenarios/scenario_generator.py

# import numpy as np
# from typing import Dict, List, Tuple
# from dataclasses import dataclass

# @dataclass
# class Environment:
#     """Environment specifications"""
#     width: float = 10.0  # 10m width
#     height: float = 10.0  # 10m height

# @dataclass
# class AgentConfig:
#     """Agent (human/robot) configuration"""
#     start_pos: np.ndarray  # [x, y]
#     goal_pos: np.ndarray   # [x, y]
#     initial_heading: float # θ
#     initial_speed: float   # v

# class ScenarioGenerator:
#     """Generates random interaction scenarios"""
#     def __init__(self, env: Environment = Environment()):
#         self.env = env
#         self.robot_goal = np.array([10.0, 10.0])  # Fixed robot goal
        
#     def generate_random_scenario(self) -> Dict:
#         """Generate a random scenario"""
#         # Fixed robot start
#         robot_config = AgentConfig(
#             start_pos=np.array([0.0, 0.0]),
#             goal_pos=self.robot_goal,
#             initial_heading=np.pi/4,  # 45 degrees towards goal
#             initial_speed=0.0
#         )
        
#         # Random human configuration
#         human_start = np.random.uniform(
#             low=[0.0, 0.0],
#             high=[self.env.width, self.env.height],
#             size=2
#         )
#         human_goal = np.random.uniform(
#             low=[0.0, 0.0],
#             high=[self.env.width, self.env.height],
#             size=2
#         )
        
#         # Initial heading towards goal
#         direction = human_goal - human_start
#         initial_heading = np.arctan2(direction[1], direction[0])
        
#         human_config = AgentConfig(
#             start_pos=human_start,
#             goal_pos=human_goal,
#             initial_heading=initial_heading,
#             initial_speed=0.0
#         )
        
#         # Random internal states
#         attention = np.random.uniform(0.2, 0.9)
#         style = np.random.uniform(0.2, 0.9)
        
#         return {
#             'robot': robot_config,
#             'human': human_config,
#             'attention_profile': {
#                 'type': 'constant',
#                 'value': attention
#             },
#             'style_profile': {
#                 'type': 'constant',
#                 'value': style
#             }
#         }

#     def generate_scenarios(self, n: int) -> List[Dict]:
#         """Generate n random scenarios"""
#         return [self.generate_random_scenario() for _ in range(n)]




















# class definition
# scenario: crossing

import numpy as np
from typing import Dict, List

class AgentConfig:
    def __init__(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                 initial_heading: float = 0.0, initial_speed: float = 0.0):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.initial_heading = initial_heading
        self.initial_speed = initial_speed

class Environment:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

class ScenarioGenerator:
    def __init__(self, env: Environment):
        self.env = env
        
    def generate_scenarios(self, n_scenarios: int) -> List[Dict]:
        scenarios = []
        
        # First add the 4 corner cases
        corner_cases = [
            {'attention': 1.0, 'style': 0.0},
            {'attention': 1.0, 'style': 1.0},
            {'attention': 0.0, 'style': 0.0},
            {'attention': 0.0, 'style': 1.0}
        ]
        
        for case in corner_cases:
            # Fixed positions for both agents
            robot_start = np.array([0.0, 5.0])
            robot_goal = np.array([10.0, 5.0])
            human_start = np.array([5.0, 0.0])
            human_goal = np.array([5.0, 10.0])
            
            # Calculate initial headings (facing goal)
            robot_heading = np.arctan2(robot_goal[1] - robot_start[1],
                                     robot_goal[0] - robot_start[0])
            human_heading = np.arctan2(human_goal[1] - human_start[1],
                                     human_goal[0] - human_start[0])
            
            # Create agent configurations
            robot_config = AgentConfig(
                start_pos=robot_start,
                goal_pos=robot_goal,
                initial_heading=robot_heading,
                initial_speed=0.0
            )
            
            human_config = AgentConfig(
                start_pos=human_start,
                goal_pos=human_goal,
                initial_heading=human_heading,
                initial_speed=0.0
            )
            
            # Use corner case values
            attention_profile = {
                'type': 'constant',
                'value': case['attention']
            }
            
            style_profile = {
                'type': 'constant',
                'value': case['style']
            }
            
            scenario = {
                'robot': robot_config,
                'human': human_config,
                'attention_profile': attention_profile,
                'style_profile': style_profile
            }
            
            scenarios.append(scenario)
        
        # Generate remaining random scenarios
        for _ in range(n_scenarios - 4):
            # Fixed positions for both agents
            robot_start = np.array([0.0, 5.0])
            robot_goal = np.array([10.0, 5.0])
            human_start = np.array([5.0, 0.0])
            human_goal = np.array([5.0, 10.0])
            
            # Calculate initial headings (facing goal)
            robot_heading = np.arctan2(robot_goal[1] - robot_start[1],
                                     robot_goal[0] - robot_start[0])
            human_heading = np.arctan2(human_goal[1] - human_start[1],
                                     human_goal[0] - human_start[0])
            
            # Create agent configurations
            robot_config = AgentConfig(
                start_pos=robot_start,
                goal_pos=robot_goal,
                initial_heading=robot_heading,
                initial_speed=0.0
            )
            
            human_config = AgentConfig(
                start_pos=human_start,
                goal_pos=human_goal,
                initial_heading=human_heading,
                initial_speed=0.0
            )
            
            # Randomize internal states
            attention_profile = {
                'type': 'constant',
                'value': np.random.uniform(0.5, 1.0)
            }
            
            style_profile = {
                'type': 'constant',
                'value': np.random.uniform(0.3, 0.7)
            }
            
            scenario = {
                'robot': robot_config,
                'human': human_config,
                'attention_profile': attention_profile,
                'style_profile': style_profile
            }
            
            scenarios.append(scenario)
        
        return scenarios