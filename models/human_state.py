# models/human_state.py

import numpy as np
import theano as th
import theano.tensor as tt
from typing import Dict, List, Tuple
import utils.helpers as utils

class InternalState:
    """
    Represents the human's internal state φ = [φatt, φstyle]
    φatt: attention level (0 to 1)
    φstyle: driving style (0 to 1, conservative to aggressive)
    """
    def __init__(self, att: float = 0.5, style: float = 0.5):
        # Internal state components
        self.phi_att = utils.scalar()
        self.phi_style = utils.scalar()
        
        # Set initial values
        self.phi_att.set_value(self.validate_value(att))
        self.phi_style.set_value(self.validate_value(style))
        
    def validate_value(self, val: float) -> float:
        """Ensure value is in [0,1]"""
        return np.clip(val, 0.0, 1.0)
    
    def get_state(self) -> Dict[str, float]:
        """Get current internal state values"""
        return {
            'attention': self.phi_att.get_value(),
            'style': self.phi_style.get_value()
        }
        
    def set_state(self, att: float, style: float):
        """Set internal state values"""
        self.phi_att.set_value(self.validate_value(att))
        self.phi_style.set_value(self.validate_value(style))

class HumanState:
    """
    Complete human state representation including both
    physical state and internal state
    """
    def __init__(self, x0: List[float]):
        # Physical state: [x, y, θ, v]
        self.physical_state = np.array(x0)
        # Internal state
        self.internal_state = InternalState()
        
    def update_physical(self, new_state: List[float]):
        """Update physical state"""
        self.physical_state = np.array(new_state)
        
    def get_full_state(self) -> Dict:
        """Get complete state"""
        return {
            'physical': self.physical_state,
            'internal': self.internal_state.get_state()
        }