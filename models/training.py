# models/training.py

import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from data.collector import Demonstration
import json

class HumanModelTrainer:
    def __init__(self):
        # Create pipelines with feature scaling
        self.base_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        self.att_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        self.style_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # For storing training results
        self.train_history = {
            'base_loss': [],
            'attention_loss': [],
            'style_loss': [],
            'total_loss': []
        }
        self.model_params = {}
        self.attention_scale = 1.0  # Changed from 0.01 to 1.0 to disable scaling

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
        """Train the human model using processed demonstrations"""
        # Prepare training data
        X, y_base, y_att, y_style = self.prepare_training_data(processed_demos)
        
        if len(X) == 0:
            raise ValueError("No valid training data found in demonstrations")
        
        if verbose:
            print(f"Training on {len(X)} samples...")
            print("Feature matrix shape:", X.shape)
            print("Target ranges:")
            print(f"  Base: [{y_base.min():.2f}, {y_base.max():.2f}]")
            print(f"  Attention: [{y_att.min():.2f}, {y_att.max():.2f}]")
            print(f"  Style: [{y_style.min():.2f}, {y_style.max():.2f}]")
        
        # Single fit for each model
        self.base_model.fit(X, y_base)
        base_loss = np.mean((self.base_model.predict(X) - y_base) ** 2)
        
        self.att_model.fit(X, y_att)
        att_loss = np.mean((self.att_model.predict(X) - y_att) ** 2)  # Removed scaling
        
        self.style_model.fit(X, y_style)
        style_loss = np.mean((self.style_model.predict(X) - y_style) ** 2)
        
        # Store final losses without scaling
        self.train_history['base_loss'].append(base_loss)
        self.train_history['attention_loss'].append(att_loss)  # No scaling
        self.train_history['style_loss'].append(style_loss)
        self.train_history['total_loss'].append(base_loss + att_loss + style_loss)  # Equal weights
        
        if verbose:
            print(f"\nTraining complete:")
            print(f"  Base loss: {base_loss:.4f}")
            print(f"  Attention loss: {att_loss:.4f}")
            print(f"  Style loss: {style_loss:.4f}")

    def predict_reward_weights(self, state: np.ndarray, action: np.ndarray, 
                             robot_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict reward weights for a given state-action pair"""
        features = self._extract_features(state, action, robot_state)
        features = features.reshape(1, -1)
        
        return {
            'base': self.base_model.predict(features)[0],
            'attention': self.att_model.predict(features)[0],
            'style': self.style_model.predict(features)[0]
        }

    def save_results(self, save_path: str):
        """Save training results and model parameters"""
        results = {
            'training_history': self.train_history,
            'model_parameters': {
                'base_model': {
                    'coefficients': self.base_model.named_steps['regressor'].coef_.tolist(),
                    'intercept': float(self.base_model.named_steps['regressor'].intercept_)
                },
                'attention_model': {
                    'coefficients': self.att_model.named_steps['regressor'].coef_.tolist(),
                    'intercept': float(self.att_model.named_steps['regressor'].intercept_)
                },
                'style_model': {
                    'coefficients': self.style_model.named_steps['regressor'].coef_.tolist(),
                    'intercept': float(self.style_model.named_steps['regressor'].intercept_)
                }
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

    def load_model(self, model_path: str):
        """Load trained model parameters from file"""
        with open(model_path, 'r') as f:
            saved_results = json.load(f)
            
        model_params = saved_results['model_parameters']
        
        # Create dummy data for fitting transformers
        dummy_features = np.zeros((1, 11))  # 11 features as defined in _extract_features
        
        # Fit polynomial features with dummy data
        self.base_model.named_steps['poly'].fit(dummy_features)
        self.att_model.named_steps['poly'].fit(dummy_features)
        self.style_model.named_steps['poly'].fit(dummy_features)
        
        # Load base model parameters
        base_params = model_params['base_model']
        self.base_model.named_steps['regressor'].coef_ = np.array(base_params['coefficients'])
        self.base_model.named_steps['regressor'].intercept_ = base_params['intercept']
        
        # Load attention model parameters
        att_params = model_params['attention_model']
        self.att_model.named_steps['regressor'].coef_ = np.array(att_params['coefficients'])
        self.att_model.named_steps['regressor'].intercept_ = att_params['intercept']
        
        # Load style model parameters
        style_params = model_params['style_model']
        self.style_model.named_steps['regressor'].coef_ = np.array(style_params['coefficients'])
        self.style_model.named_steps['regressor'].intercept_ = style_params['intercept']
        
        # Load training history if available
        if 'training_history' in saved_results:
            self.train_history = saved_results['training_history']
            
        print(f"Model loaded successfully from {model_path}")