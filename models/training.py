# models/training.py

import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from data.collector import Demonstration
import json

class HumanModelTrainer:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2)
        self.base_model = LinearRegression()
        self.att_model = LinearRegression()
        self.style_model = LinearRegression()
        
        # For storing training results
        self.train_history = {
            'base_loss': [],
            'attention_loss': [],
            'style_loss': []
        }
        self.model_params = {}

    def prepare_training_data(self, processed_demos: List[List[Dict]]):
        """
        Prepare training data from processed demonstrations
        
        Args:
            processed_demos: List of processed demonstrations
        Returns:
            X: Feature matrix
            y_base: Base reward values
            y_att: Attention-dependent values
            y_style: Style-dependent values
        """
        X = []  # States and actions
        y_base = []  # Base reward values
        y_att = []   # Attention-dependent values
        y_style = [] # Style-dependent values
        
        for demo in processed_demos:
            for timestep in demo:
                if timestep['action'] is not None:  # Only use timesteps with actions
                    # Extract features
                    features = self._extract_features(
                        timestep['state'],
                        timestep['action'],
                        timestep['robot_state']
                    )
                    X.append(features)
                    
                    # Extract reward components
                    y_base.append(timestep['base_reward'])
                    y_att.append(timestep['attention_reward'])
                    y_style.append(timestep['style_reward'])
        
        return np.array(X), np.array(y_base), np.array(y_att), np.array(y_style)
    
    def _extract_features(self, state, action, robot_state):
        """Extract features for training"""
        features = np.concatenate([
            state,
            action,
            robot_state,
            [np.linalg.norm(state[:2] - robot_state[:2])]  # Distance feature
        ])
        return features
        
    def train(self, processed_demos: List[List[Dict]], verbose: bool = True):
        """
        Train the human model using processed demonstrations
        
        Args:
            processed_demos: List of processed demonstrations
            verbose: Whether to print training progress
        """
        # Prepare training data
        X, y_base, y_att, y_style = self.prepare_training_data(processed_demos)
        
        if len(X) == 0:
            raise ValueError("No valid training data found in demonstrations")
            
        # Transform features
        X_poly = self.poly.fit_transform(X)
        
        if verbose:
            print(f"Training on {len(X)} samples...")
        
        # Train base reward model
        self.base_model.fit(X_poly, y_base)
        base_loss = np.mean((self.base_model.predict(X_poly) - y_base) ** 2)
        self.train_history['base_loss'].append(base_loss)
        
        # Train attention model
        self.att_model.fit(X_poly, y_att)
        att_loss = np.mean((self.att_model.predict(X_poly) - y_att) ** 2)
        self.train_history['attention_loss'].append(att_loss)
        
        # Train style model
        self.style_model.fit(X_poly, y_style)
        style_loss = np.mean((self.style_model.predict(X_poly) - y_style) ** 2)
        self.train_history['style_loss'].append(style_loss)
        
        if verbose:
            print(f"Training complete:")
            print(f"  Base loss: {base_loss:.4f}")
            print(f"  Attention loss: {att_loss:.4f}")
            print(f"  Style loss: {style_loss:.4f}")

    def predict_reward_weights(self, 
                             state: np.ndarray,
                             action: np.ndarray,
                             robot_state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict reward weights for a given state-action pair
        
        Returns:
            Dictionary containing base, attention, and style weights
        """
        # Extract and transform features
        features = self._extract_features(state, action, robot_state)
        features_poly = self.poly.transform(features.reshape(1, -1))
        
        # Predict weights
        base_weights = self.base_model.predict(features_poly)
        att_weights = self.att_model.predict(features_poly)
        style_weights = self.style_model.predict(features_poly)
        
        return {
            'base': base_weights[0],
            'attention': att_weights[0],
            'style': style_weights[0]
        }

    def save_results(self, save_path: str):
        """Save training results and model parameters"""
        results = {
            'training_history': self.train_history,
            'model_parameters': {
                'base_model': {
                    'coefficients': self.base_model.coef_.tolist(),
                    'intercept': float(self.base_model.intercept_)
                },
                'attention_model': {
                    'coefficients': self.att_model.coef_.tolist(),
                    'intercept': float(self.att_model.intercept_)
                },
                'style_model': {
                    'coefficients': self.style_model.coef_.tolist(),
                    'intercept': float(self.style_model.intercept_)
                }
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)