# utils/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from data.collector import Demonstration
import matplotlib.patches as patches

class TrajectoryVisualizer:
    """Visualizes human-robot interactions from demonstrations"""
    
    def __init__(self):
        self.colors = {
            'human': '#0066CC',  # Darker blue
            'robot': '#CC0000',  # Darker red
            'human_dots': '#66B3FF',  # Lighter blue
            'robot_dots': '#FF6666',  # Lighter red
            'human_goal': '#003366',  # Navy blue
            'robot_goal': '#660000',  # Dark red
            'boundary': 'gray'
        }
        
    def plot_trajectory(self, demo: Demonstration, title: str = None, save_path: str = None):
        """Plot complete trajectory visualization"""
        # Create single plot with appropriate size
        fig = plt.figure(figsize=(10, 8))
        ax_traj = plt.gca()
        
        # Trajectory plot
        self._plot_trajectory_main(demo, ax_traj)
        
        # Overall title
        if title:
            plt.title(title, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_trajectory_main(self, demo: Demonstration, ax):
        """Plot main trajectory visualization"""
        print("=== Starting visualization ===")
        print("Demo data:", len(demo.states) if demo else "No demo data")
        # Draw environment boundary (10m x 10m)
        boundary = patches.Rectangle((0, 0), 10, 10, fill=False, 
                                color=self.colors['boundary'], 
                                linestyle='--', alpha=0.5)
        ax.add_patch(boundary)
        
        # Extract trajectories
        human_traj = np.array(demo.states)
        robot_traj = np.array(demo.robot_states)
        times = np.array(demo.timestamps)
        
        # Plot continuous trajectories with thicker lines
        ax.plot(human_traj[:, 0], human_traj[:, 1], 
                color=self.colors['human'], label='Human', alpha=0.7, linewidth=2)
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], 
                color=self.colors['robot'], label='Robot', alpha=0.7, linewidth=2)
        
        # Calculate indices for each second
        dt = times[1] - times[0]  # Get timestep
        steps_per_second = int(round(1.0/dt))
        
        print("len(times): ", len(times))
        print("steps_per_second: ", steps_per_second)
        
        # Plot second markers with enhanced visibility
        for t in range(0, len(times), steps_per_second):  # Changed to every 1 seconds
            if t < len(human_traj):
                # Plot human position at this second with two circles
                # ax.scatter(human_traj[t, 0], human_traj[t, 1],
                #         color=self.colors['human'], s=300,  # Larger outer circle
                #         marker='o', alpha=0.3, zorder=5)
                ax.scatter(human_traj[t, 0], human_traj[t, 1],
                        color=self.colors['human'], s=100,  # Smaller inner circle
                        marker='o', alpha=1.0, zorder=6)  # Increased opacity
                # Add time label with background for better visibility
                # ax.text(human_traj[t, 0], human_traj[t, 1], f'{int(times[t])}s',
                #     color=self.colors['human'], fontsize=10, fontweight='bold',
                #     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                #     verticalalignment='bottom', horizontalalignment='right')
                
                # Plot robot position at this second with two circles
                # ax.scatter(robot_traj[t, 0], robot_traj[t, 1],
                #         color=self.colors['robot'], s=300,  # Larger outer circle
                #         marker='o', alpha=0.3, zorder=5)
                ax.scatter(robot_traj[t, 0], robot_traj[t, 1],
                        color=self.colors['robot'], s=100,  # Smaller inner circle
                        marker='o', alpha=1.0, zorder=6)  # Increased opacity
                # # Add time label with background
                # ax.text(robot_traj[t, 0], robot_traj[t, 1], f'{int(times[t])}s',
                #     color=self.colors['robot'], fontsize=10, fontweight='bold',
                #     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                #     verticalalignment='top', horizontalalignment='left')
        
        # Plot start positions with bigger markers
        ax.scatter(human_traj[0, 0], human_traj[0, 1], 
                color=self.colors['human'], s=400, marker='*', 
                label='Human Start', zorder=7, edgecolor='white', linewidth=2)
        ax.scatter(robot_traj[0, 0], robot_traj[0, 1], 
                color=self.colors['robot'], s=400, marker='*', 
                label='Robot Start', zorder=7, edgecolor='white', linewidth=2)
        
        # Plot goals with standard star markers
        if demo.scenario_info:
            human_goal = demo.scenario_info['human_goal']
            robot_goal = demo.scenario_info['robot_goal']
            
            ax.scatter(human_goal[0], human_goal[1], 
                    color=self.colors['human_goal'], s=200, marker='*',
                    label='Human Goal', zorder=5)
            ax.scatter(robot_goal[0], robot_goal[1], 
                    color=self.colors['robot_goal'], s=200, marker='*',
                    label='Robot Goal', zorder=5)
        
        # Plot orientations at regular intervals
        step = max(1, len(human_traj) // 10)  # Show 10 orientations along trajectory
        for i in range(0, len(human_traj), step):
            self._plot_orientation(ax, human_traj[i, 0], human_traj[i, 1], 
                                human_traj[i, 2], self.colors['human'])
            self._plot_orientation(ax, robot_traj[i, 0], robot_traj[i, 1], 
                                robot_traj[i, 2], self.colors['robot'])
        
        # Set plot properties
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                ncol=3, fancybox=True, shadow=True)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Trajectory Visualization')
        
    # def _plot_internal_states(self, demo: Demonstration, ax):
    #     """Plot evolution of internal states"""
    #     times = np.array(demo.timestamps)
    #     attention = [state['attention'] for state in demo.internal_states]
    #     style = [state['style'] for state in demo.internal_states]
        
    #     ax.plot(times, attention, 'g-', label='Attention', linewidth=2)
    #     ax.plot(times, style, 'm-', label='Style', linewidth=2)
        
    #     # Add measurement points
    #     ax.scatter(times, attention, color='green', alpha=0.3, s=20)
    #     ax.scatter(times, style, color='magenta', alpha=0.3, s=20)
        
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(loc='upper right')
    #     ax.set_title('Internal State Evolution')
    #     ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('State Value')
    #     ax.set_ylim(-0.1, 1.1)
        
    def _plot_orientation(self, ax, x, y, theta, color):
        """Plot orientation arrow"""
        length = 0.3  # Arrow length
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, 
                fc=color, ec=color, alpha=0.5)
        
    def plot_metrics(self, demo: Demonstration, save_path: str = None):
        """Plot additional metrics for the demonstration"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Speed profiles
        ax = axes[0, 0]
        human_speeds = [state[3] for state in demo.states]
        robot_speeds = [state[3] for state in demo.robot_states]
        times = demo.timestamps
        
        ax.plot(times, human_speeds, color=self.colors['human'], label='Human')
        ax.plot(times, robot_speeds, color=self.colors['robot'], label='Robot')
        ax.set_title('Speed Profiles')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.grid(True)
        ax.legend()
        
        # Distance between agents
        ax = axes[0, 1]
        distances = [np.linalg.norm(np.array(h[:2]) - np.array(r[:2])) 
                    for h, r in zip(demo.states, demo.robot_states)]
        ax.plot(times, distances, 'k-', label='Inter-agent Distance')
        ax.set_title('Distance Between Agents')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.grid(True)
        ax.legend()
        
        # Distance to goals
        ax = axes[1, 0]
        if demo.scenario_info:
            human_goal = demo.scenario_info['human_goal']
            robot_goal = demo.scenario_info['robot_goal']
            
            dist_to_human_goal = [np.linalg.norm(np.array(s[:2]) - human_goal) 
                                for s in demo.states]
            dist_to_robot_goal = [np.linalg.norm(np.array(s[:2]) - robot_goal) 
                                for s in demo.robot_states]
            
            ax.plot(times, dist_to_human_goal, 
                   color=self.colors['human'], label='Human to Goal')
            ax.plot(times, dist_to_robot_goal, 
                   color=self.colors['robot'], label='Robot to Goal')
            ax.set_title('Distance to Goals')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance (m)')
            ax.grid(True)
            ax.legend()
        
        # Actions
        ax = axes[1, 1]
        if demo.actions:
            human_actions = np.array(demo.actions)
            robot_actions = np.array(demo.robot_actions)
            
            ax.plot(times, human_actions[:, 0], 
                   color=self.colors['human'], linestyle='-', label='Human Steering')
            ax.plot(times, human_actions[:, 1], 
                   color=self.colors['human'], linestyle='--', label='Human Acceleration')
            ax.plot(times, robot_actions[:, 0], 
                   color=self.colors['robot'], linestyle='-', label='Robot Steering')
            ax.plot(times, robot_actions[:, 1], 
                   color=self.colors['robot'], linestyle='--', label='Robot Acceleration')
            ax.set_title('Control Actions')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Action Value')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()