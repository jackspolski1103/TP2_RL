"""
Entorno con recompensa constante.

Este entorno siempre devuelve la misma observación (0) y recompensa constante (+1).
Útil para verificar que la función de valor aprende correctamente un valor constante.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ConstantRewardEnv(gym.Env):
    """
    Entorno de recompensa constante para testing de DQN.
    
    Características:
    - Acción única (solo hay una acción posible)
    - Observación constante: siempre 0
    - Duración: 1 paso por episodio
    - Recompensa: siempre +1
    
    Este entorno sirve para verificar si la función de valor aprende
    correctamente un valor constante en un caso trivial.
    """
    
    def __init__(self):
        """
        Inicializa el entorno de recompensa constante.
        """
        super().__init__()
        
        # Espacio de observación: valor discreto constante (0)
        self.observation_space = spaces.Discrete(1)
        
        # Espacio de acción: una sola acción posible
        self.action_space = spaces.Discrete(1)
        
        # Estado interno
        self.current_step = 0
    
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Args:
            seed: Semilla para reproducibilidad (no usada en este entorno)
            options: Opciones adicionales (no usadas)
        
        Returns:
            observation: Observación inicial (siempre 0)
            info: Información adicional (vacía)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        observation = 0  # Observación constante
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar (ignorada, solo hay una acción)
            
        Returns:
            observation: Nueva observación (siempre 0)
            reward: Recompensa obtenida (siempre +1)
            terminated: Si el episodio terminó (siempre True después de 1 paso)
            truncated: Si el episodio fue truncado (siempre False)
            info: Información adicional
        """
        self.current_step += 1
        
        # Observación constante
        observation = 0
        
        # Recompensa constante
        reward = 1.0
        
        # El episodio termina después de 1 paso
        terminated = True
        truncated = False
        
        info = {
            'step': self.current_step,
            'reward_type': 'constant'
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Renderiza el entorno mostrando el estado actual."""
        print(f"ConstantRewardEnv - Paso: {self.current_step}, Obs: 0, Reward: +1")
