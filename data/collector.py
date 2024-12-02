# data/collector.py

import numpy as np
from typing import Dict, List, Optional
from models.human_state import HumanState
from models.reward import RewardFunction
from scenarios.scenario_generator import AgentConfig

class Demonstration:
    """Single demonstration data containing all trajectory information"""
    def __init__(self):
        self.states = []           # List of [x, y, θ, v] 
        self.actions = []          # List of [steering, acceleration]
        self.robot_states = []     # List of [x, y, θ, v]
        self.robot_actions = []    # List of [steering, acceleration]
        self.internal_states = []  # List of {'attention': float, 'style': float}
        self.timestamps = []       # List of time points
        self.rewards = []          # Computed rewards for each timestep
        self.human_goal = None
        self.robot_goal = None
        self.scenario_info = None

    def add_timestep(self, state: np.ndarray,
                    action: Optional[np.ndarray],
                    robot_state: np.ndarray,
                    robot_action: Optional[np.ndarray],
                    internal_state: Dict[str, float],
                    timestamp: float,
                    reward: Optional[float] = None):
        """Add data for a single timestep"""
        self.states.append(state)
        self.actions.append(action)
        self.robot_states.append(robot_state)
        self.robot_actions.append(robot_action)
        self.internal_states.append(internal_state)
        self.timestamps.append(timestamp)
        self.rewards.append(reward)

    def get_timestep(self, t: int) -> Dict:
        """Get all data for timestep t"""
        return {
            'state': self.states[t],
            'action': self.actions[t],
            'robot_state': self.robot_states[t],
            'robot_action': self.robot_actions[t],
            'internal_state': self.internal_states[t],
            'timestamp': self.timestamps[t],
            'reward': self.rewards[t]
        }

class DataCollector:
    """Collects and manages demonstrations"""
    def __init__(self, dt: float = 0.1):  # Using 0.1s timestep for smoother control
        self.demonstrations = []
        self.dt = dt
        self.reward_function = RewardFunction()

    def collect_demonstration(self, 
                            human_state: HumanState,
                            robot_state: np.ndarray,
                            human_config: AgentConfig,
                            robot_config: AgentConfig,
                            attention_profile: Dict,
                            style_profile: Dict,
                            duration: float = 30.0) -> Demonstration:
        """Collect complete demonstration"""
        demo = Demonstration()
        demo.human_goal = human_config.goal_pos
        demo.robot_goal = robot_config.goal_pos
        demo.scenario_info = {
            'human_start': human_config.start_pos,
            'human_goal': human_config.goal_pos,
            'robot_start': robot_config.start_pos,
            'robot_goal': robot_config.goal_pos,
            'attention_profile': attention_profile,
            'style_profile': style_profile
        }

        t = 0
        while t < duration:
            # Check robot goal first
            robot_dist = np.linalg.norm(robot_state[:2] - robot_config.goal_pos)
            robot_speed = abs(robot_state[3])
            if robot_dist < 0.3 and robot_speed < 0.01:
                break  # Stop simulation if robot has reached goal and stopped
                
            internal_state = {
                'attention': self._get_attention(t, attention_profile),
                'style': self._get_style(t, style_profile)
            }
            
            human_state.internal_state.set_state(
                internal_state['attention'],
                internal_state['style']
            )

            human_action = self._get_human_action(
                human_state.physical_state,
                robot_state,
                internal_state,
                human_config.goal_pos
            )

            robot_action = self._get_robot_action(
                robot_state,
                human_state.physical_state,
                robot_config.goal_pos
            )

            reward = self.reward_function.compute_reward(
                human_state.physical_state,
                human_action,
                robot_state,
                internal_state
            )

            demo.add_timestep(
                state=human_state.physical_state.copy(),
                action=human_action,
                robot_state=robot_state.copy(),
                robot_action=robot_action,
                internal_state=internal_state.copy(),
                timestamp=t,
                reward=reward
            )

            human_state = self._simulate_step(human_state, human_action)
            robot_state = self._simulate_robot_step(robot_state, robot_action)
            t += self.dt

        self.demonstrations.append(demo)
        return demo

    def _get_robot_action(self, robot_state, human_state, goal):
        """Goal-directed robot action with hard stopping behavior"""
        if robot_state is None or human_state is None or goal is None:
            return np.array([0.0, 0.0])
            
        try:
            dist_to_goal = np.linalg.norm(goal - robot_state[:2])
            current_speed = robot_state[3]
            
            # Very close to goal - force full stop
            if dist_to_goal < 0.3:
                return np.array([0.0, -3.0])  # Strong braking force
                
            # Approaching goal - aggressive speed reduction
            if dist_to_goal < 1.0:
                # Quadratic speed reduction for smoother deceleration
                desired_speed = max(0.01, 0.2 * dist_to_goal * dist_to_goal)
                
                to_goal = goal - robot_state[:2]
                desired_heading = np.arctan2(to_goal[1], to_goal[0])
                heading_error = self._normalize_angle(desired_heading - robot_state[2])
                
                # Stronger steering when close to goal
                steering = 3.0 * heading_error
                
                # Further reduce speed if not aligned with goal
                if abs(heading_error) > 0.1:
                    desired_speed *= 0.3
                    
                speed_error = desired_speed - current_speed
                return np.array([steering, np.clip(speed_error, -2.0, 1.0)])
            
            # Normal navigation
            to_goal = goal - robot_state[:2]
            desired_heading = np.arctan2(to_goal[1], to_goal[0])
            heading_error = self._normalize_angle(desired_heading - robot_state[2])
            
            steering = 2.0 * heading_error
            desired_speed = min(1.0, dist_to_goal)
            
            # Safety around human
            dist_to_human = np.linalg.norm(robot_state[:2] - human_state[:2])
            if dist_to_human < 2.0:
                desired_speed *= (dist_to_human / 2.0)
            
            speed_error = desired_speed - current_speed
            return np.array([steering, np.clip(speed_error, -1.0, 1.0)])
            
        except:
            return np.array([0.0, 0.0])

    def _get_human_action(self, human_state, robot_state, internal_state, goal):
        """Goal-directed human action with improved stopping behavior"""
        if human_state is None or robot_state is None or internal_state is None or goal is None:
            return np.array([0.0, 0.0])
            
        try:
            dist_to_goal = np.linalg.norm(goal - human_state[:2])
            current_speed = human_state[3]
            
            # Very close to goal - stop
            if dist_to_goal < 0.3:
                return np.array([0.0, -2.0 * current_speed])
            
            # Approaching goal - reduce speed
            if dist_to_goal < 1.0:
                desired_speed = max(0.05, 0.3 * dist_to_goal * dist_to_goal)
                desired_speed *= (0.3 + 0.7 * internal_state['style'])
                
                to_goal = goal - human_state[:2]
                desired_heading = np.arctan2(to_goal[1], to_goal[0])
                heading_error = self._normalize_angle(desired_heading - human_state[2])
                
                steering = 3.0 * heading_error * (0.5 + 0.5 * internal_state['attention'])
                
                if abs(heading_error) > 0.1:
                    desired_speed *= 0.3
                
                speed_error = desired_speed - current_speed
                return np.array([steering, np.clip(speed_error, -2.0, 1.0)])
            
            # Normal navigation
            to_goal = goal - human_state[:2]
            desired_heading = np.arctan2(to_goal[1], to_goal[0])
            heading_error = self._normalize_angle(desired_heading - human_state[2])
            
            steering = 2.0 * heading_error * (0.5 + 0.5 * internal_state['attention'])
            desired_speed = min(1.0, dist_to_goal) * (0.5 + 0.5 * internal_state['style'])
            
            # Safety check
            dist_to_robot = np.linalg.norm(human_state[:2] - robot_state[:2])
            safe_dist = 2.0 / internal_state['attention']
            
            if dist_to_robot < safe_dist:
                desired_speed *= (dist_to_robot / safe_dist)
                
            speed_error = desired_speed - current_speed
            return np.array([steering, np.clip(speed_error, -1.0, 1.0)])
            
        except:
            return np.array([0.0, 0.0])

    def _get_attention(self, t: float, profile: Dict) -> float:
        """Get attention level at time t"""
        if profile['type'] == 'constant':
            return profile['value']
        elif profile['type'] == 'step':
            for i in range(len(profile['times'])-1):
                if profile['times'][i] <= t < profile['times'][i+1]:
                    return profile['values'][i]
            return profile['values'][-1]
        elif profile['type'] == 'function':
            return profile['f'](t)
        else:
            raise ValueError(f"Unknown profile type: {profile['type']}")

    def _get_style(self, t: float, profile: Dict) -> float:
        """Get style at time t"""
        return self._get_attention(t, profile)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _simulate_step(self, human_state: HumanState, action: np.ndarray) -> HumanState:
        """Simulate one timestep for human"""
        x, y, theta, v = human_state.physical_state
        steering, acceleration = action
        
        dx = v * np.cos(theta) * self.dt
        dy = v * np.sin(theta) * self.dt
        dtheta = v * steering * self.dt
        dv = acceleration * self.dt
        
        new_state = np.array([
            x + dx,
            y + dy,
            self._normalize_angle(theta + dtheta),
            np.clip(v + dv, 0, 2.0)
        ])
        
        human_state.physical_state = new_state
        return human_state

    def _simulate_robot_step(self, robot_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simulate one timestep for robot"""
        x, y, theta, v = robot_state
        steering, acceleration = action
        
        dx = v * np.cos(theta) * self.dt
        dy = v * np.sin(theta) * self.dt
        dtheta = v * steering * self.dt
        dv = acceleration * self.dt
        
        return np.array([
            x + dx,
            y + dy,
            self._normalize_angle(theta + dtheta),
            np.clip(v + dv, 0, 2.0)
        ])

    def save_demonstration_plot(self, demo: Demonstration, title: str = None, save_path: str = None):
        """Save visualization of demonstration"""
        from utils.visualizer import TrajectoryVisualizer
        visualizer = TrajectoryVisualizer()
        visualizer.plot_trajectory(demo, title, save_path)
        
        if save_path:
            metrics_path = save_path.replace('.png', '_metrics.png')
            visualizer.plot_metrics(demo, metrics_path)