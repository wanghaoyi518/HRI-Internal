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
            # Add early termination condition
            human_dist = np.linalg.norm(human_state.physical_state[:2] - human_config.goal_pos)
            human_speed = abs(human_state.physical_state[3])
            robot_dist = np.linalg.norm(robot_state[:2] - robot_config.goal_pos)
            robot_speed = abs(robot_state[3])
            
            # Stop if both agents are at their goals and nearly stopped
            if (human_dist < 0.3 and human_speed < 0.1 and 
                robot_dist < 0.3 and robot_speed < 0.1):
                break

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
        if robot_state is None or human_state is None or goal is None:
            return np.array([0.0, 0.0])
            
        try:
            dist_to_goal = np.linalg.norm(goal - robot_state[:2])
            current_speed = robot_state[3]
            
            # Very close to goal - force full stop
            if dist_to_goal < 0.5:
                return np.array([0.0, -4.0])  # Strong braking force
                
            # Compute relative position and velocity
            rel_pos = robot_state[:2] - human_state[:2]
            rel_dist = np.linalg.norm(rel_pos)
            
            robot_vel = current_speed * np.array([np.cos(robot_state[2]), np.sin(robot_state[2])])
            human_vel = human_state[3] * np.array([np.cos(human_state[2]), np.sin(human_state[2])])
            rel_vel = robot_vel - human_vel
            rel_speed = np.linalg.norm(rel_vel)
            
            # Check if approaching
            approaching = np.dot(rel_pos, rel_vel) < 0
            
            # Approaching goal - aggressive speed reduction
            if dist_to_goal < 2.0:
                desired_speed = max(0.01, 0.1 * dist_to_goal * dist_to_goal)
                
                to_goal = goal - robot_state[:2]
                desired_heading = np.arctan2(to_goal[1], to_goal[0])
                heading_error = self._normalize_angle(desired_heading - robot_state[2])
                
                steering = 3.0 * heading_error
                
                if abs(heading_error) > 0.1:
                    desired_speed *= 0.2
                    
                speed_error = desired_speed - current_speed
                return np.array([steering, np.clip(speed_error, -2.0, 1.0)])
            
            # Normal navigation
            to_goal = goal - robot_state[:2]
            desired_heading = np.arctan2(to_goal[1], to_goal[0])
            heading_error = self._normalize_angle(desired_heading - robot_state[2])
            
            steering = 2.0 * heading_error
            desired_speed = min(1.0, dist_to_goal)
            
            # Enhanced safety logic
            if approaching and rel_dist < 4.0:
                # Time to collision
                ttc = rel_dist / (rel_speed + 1e-6)
                
                # Progressive braking based on distance and TTC
                brake_factor = min(1.0, (rel_dist / 4.0) ** 2) * min(1.0, (ttc / 2.0) ** 2)
                desired_speed *= brake_factor
                
                # Stronger braking for close distances
                if rel_dist < 2.0:
                    desired_speed *= 0.5
                    if rel_dist < 1.0:
                        desired_speed *= 0.3
            
            speed_error = desired_speed - current_speed
            return np.array([steering, np.clip(speed_error, -1.0, 1.0)])
            
        except:
            return np.array([0.0, 0.0])

    def _get_human_action(self, human_state, robot_state, internal_state, goal):
        if human_state is None or robot_state is None or internal_state is None or goal is None:
            return np.array([0.0, 0.0])
        
        try:
            # Extract states
            dist_to_goal = np.linalg.norm(goal - human_state[:2])
            current_speed = human_state[3]
            attention = internal_state['attention']
            style = internal_state['style']
            
            # Very close to goal - force stop regardless of style
            if dist_to_goal < 0.3:
                return np.array([0.0, -4.0 * current_speed])  # Strong brake
            
            # Near goal - reduce speed and improve precision
            if dist_to_goal < 1.0:
                desired_speed = 0.2 * dist_to_goal  # Linear speed reduction
                
                to_goal = goal - human_state[:2]
                desired_heading = np.arctan2(to_goal[1], to_goal[0])
                heading_error = self._normalize_angle(desired_heading - human_state[2])
                
                # More precise steering near goal
                steering = 2.0 * heading_error
                
                # Stop if headed wrong direction near goal
                if abs(heading_error) > 0.5:
                    desired_speed = 0.0
                
                return np.array([
                    np.clip(steering, -1.0, 1.0),
                    np.clip(-current_speed, -2.0, 1.0)  # Focus on stopping
                ])
            
            # Normal navigation - rest of the existing logic
            # Base parameters
            max_speed = 1.0 + style
            attention_factor = 0.3 + 0.7 * attention
            
            # Compute relative position and velocity
            rel_pos = human_state[:2] - robot_state[:2]
            rel_dist = np.linalg.norm(rel_pos)
            
            human_vel = current_speed * np.array([np.cos(human_state[2]), np.sin(human_state[2])])
            robot_vel = robot_state[3] * np.array([np.cos(robot_state[2]), np.sin(robot_state[2])])
            rel_vel = human_vel - robot_vel
            rel_speed = np.linalg.norm(rel_vel)
            
            # Check if approaching
            approaching = np.dot(rel_pos, rel_vel) < 0
            
            # Very close to goal - stop
            if dist_to_goal < 0.3:
                return np.array([0.0, -2.0 * current_speed * attention_factor])
            
            # Approaching goal
            if dist_to_goal < 1.0:
                desired_speed = max(0.05, 0.3 * dist_to_goal * dist_to_goal)
                desired_speed *= (0.5 + 0.5 * style)
                
                to_goal = goal - human_state[:2]
                desired_heading = np.arctan2(to_goal[1], to_goal[0])
                heading_error = self._normalize_angle(desired_heading - human_state[2])
                
                steering = 3.0 * heading_error * attention_factor
                
                if abs(heading_error) > 0.1:
                    desired_speed *= 0.5
                
                speed_error = desired_speed - current_speed
                return np.array([steering, np.clip(speed_error, -2.0, 1.0) * attention_factor])
            
            # Normal navigation
            to_goal = goal - human_state[:2]
            desired_heading = np.arctan2(to_goal[1], to_goal[0])
            heading_error = self._normalize_angle(desired_heading - human_state[2])
            
            steering = 2.0 * heading_error * attention_factor * (0.5 + 0.5 * style)
            desired_speed = min(max_speed, dist_to_goal) * (0.7 + 0.3 * style)
            
            # Enhanced safety logic based on attention
            if approaching and rel_dist < 4.0 * (2.0 - attention):
                # Time to collision
                ttc = rel_dist / (rel_speed + 1e-6)
                
                # Progressive braking based on distance, TTC and attention
                brake_dist = 4.0 * (2.0 - attention)
                brake_factor = min(1.0, (rel_dist / brake_dist) ** 2) * min(1.0, (ttc / 2.0) ** 2)
                desired_speed *= brake_factor * attention_factor
                
                # Extra caution for low attention
                if attention < 0.5:
                    desired_speed *= 0.7
                
                # Stronger braking for close distances
                if rel_dist < 2.0:
                    desired_speed *= 0.5 * attention_factor
                    if rel_dist < 1.0:
                        desired_speed *= 0.3
            
            speed_error = desired_speed - current_speed
            acceleration = np.clip(speed_error, -1.0, 1.0) * attention_factor
            
            return np.array([np.clip(steering, -1.0, 1.0), acceleration])
            
        except Exception as e:
            print(f"Error in human action computation: {e}")
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
        
        dx = v * np.cos(theta)  # No dt needed since it's built into the dynamics
        dy = v * np.sin(theta)
        dtheta = steering  # Direct heading control
        dv = acceleration * self.dt  # Only velocity change needs dt
        
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
        
        dx = v * np.cos(theta)  # No dt needed since it's built into the dynamics
        dy = v * np.sin(theta)
        dtheta = steering  # Direct heading control
        dv = acceleration * self.dt  # Only velocity change needs dt
        
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
        
        # Save static plot and animation
        if save_path:
            # Save PNG
            visualizer.plot_trajectory(demo, title, save_path)
            
            # Save GIF
            gif_path = save_path.replace('.png', '.gif')
            visualizer.create_animation(demo, title, gif_path)
            
            # Save metrics
            metrics_path = save_path.replace('.png', '_metrics.png')
            visualizer.plot_metrics(demo, metrics_path)
        else:
            visualizer.plot_trajectory(demo, title)