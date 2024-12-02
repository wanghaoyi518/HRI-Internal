# File: models/state_estimation.py

import numpy as np
from typing import Dict, List, Optional
import theano as th
import theano.tensor as tt
from models.human_state import HumanDriver

class StateEstimator:
    """
    Handles estimation and updating of human internal state
    Uses Bayesian inference to maintain belief over internal states
    """
    def __init__(self, human_model: HumanDriver):
        self.human_model = human_model
        
        # Discretization for belief representation
        self.n_att_bins = 10
        self.n_style_bins = 10
        
        # Initialize uniform belief
        self.belief = self._initialize_belief()
        
        # Compile theano functions
        self._compile_update_functions()
        
    def _initialize_belief(self) -> np.ndarray:
        """Initialize uniform belief distribution"""
        belief = np.ones((self.n_att_bins, self.n_style_bins))
        return belief / belief.sum()
        
    def _compile_update_functions(self):
        """Compile theano functions for belief updates"""
        # Symbolic variables
        observation = tt.vector('observation')
        belief = tt.matrix('belief')
        
        # Likelihood computation
        likelihood = self._compute_likelihood_function(observation)
        
        # State transition
        transition = self._compute_transition_function(belief)
        
        # Belief update
        posterior = transition * likelihood
        posterior = posterior / posterior.sum()
        
        # Compile functions
        self.compute_likelihood = th.function([observation], likelihood)
        self.compute_transition = th.function([belief], transition)
        self.update_belief = th.function([belief, observation], posterior)
        
    def _compute_likelihood_function(self, observation):
        """
        Compute likelihood of observation given internal state
        P(o|φ) based on reward model
        """
        state = observation[:4]  # Physical state
        action = observation[4:6]  # Human action
        robot_state = observation[6:10]  # Robot state
        
        # Get expected action from human model
        expected_action = self.human_model.compute_reward(state, action, robot_state)
        
        # Compute likelihood using Gaussian around expected action
        variance = 0.1
        likelihood = tt.exp(-tt.sum(tt.square(action - expected_action))/(2*variance))
        
        return likelihood
        
    def _compute_transition_function(self, belief):
        """
        Compute state transition probabilities
        P(φt+1|φt) using transition model
        """
        # Simple Gaussian transition model
        transition_matrix = tt.zeros_like(belief)
        
        for i in range(self.n_att_bins):
            for j in range(self.n_style_bins):
                att = i / (self.n_att_bins-1)
                style = j / (self.n_style_bins-1)
                
                next_state = self.human_model.internal_state.predict_next_state()
                transition_matrix = tt.set_subtensor(
                    transition_matrix[i,j],
                    self._gaussian_transition(next_state['attentiveness'], next_state['driving_style'])
                )
                
        return transition_matrix
        
    def update(self, observation: np.ndarray):
        """
        Update belief over internal state given new observation
        """
        # Compute likelihood
        likelihood = self.compute_likelihood(observation)
        
        # Predict next state (transition)
        predicted_belief = self.compute_transition(self.belief)
        
        # Update belief
        self.belief = self.update_belief(predicted_belief, observation)
        
        # Update human model's internal state estimate with MAP
        self._update_map_estimate()
        
    def _update_map_estimate(self):
        """Update MAP estimate of internal state"""
        # Find MAP state
        i, j = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        
        # Convert to continuous values
        att = i / (self.n_att_bins-1)
        style = j / (self.n_style_bins-1)
        
        # Update human model
        self.human_model.internal_state.set_state(att, style)