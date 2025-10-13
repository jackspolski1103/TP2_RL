"""
Entorno de dos pasos con recompensa retrasada.

Entorno simple con exactamente dos pasos donde la recompensa solo
se obtiene al final. Útil para testear el mecanismo de descuento temporal.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TwoStepDelayedRewardEnv(gym.Env):
    """
    Entorno de dos pasos con recompensa retrasada para testing de DQN.
    
    Características:
    - Acción única (solo hay una acción posible)
    - Observación determinista:
      * Paso 1: observación = 0
      * Paso 2: observación = 1
    - Duración: exactamente 2 pasos por episodio
    - Recompensa: 0 en el primer paso, +1 en el segundo paso (final)
    
    Este entorno sirve para testear el mecanismo de descuento temporal
    y verificar que el agente puede aprender valores con recompensas retrasadas.
    """
    
    def __init__(self):
        """
        Inicializa el entorno de dos pasos con recompensa retrasada.
        """
        super().__init__()
        
        # Espacio de observación: valores discretos 0 o 1 (paso 1 o paso 2)
        self.observation_space = spaces.Discrete(2)
        
        # Espacio de acción: una sola acción posible
        self.action_space = spaces.Discrete(1)
        
        # Estado interno
        self.current_step = 0
        self.current_state = 0  # 0 = primer paso, 1 = segundo paso
    
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Args:
            seed: Semilla para reproducibilidad (no usada en este entorno)
            options: Opciones adicionales (no usadas)
        
        Returns:
            observation: Observación inicial (siempre 0 para el primer paso)
            info: Información adicional
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_state = 0  # Empezar en el primer paso
        
        observation = self.current_state
        info = {
            'step': self.current_step,
            'state': self.current_state
        }
        
        return observation, info
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar (ignorada, solo hay una acción)
            
        Returns:
            observation: Nueva observación (0 en paso 1, 1 en paso 2)
            reward: Recompensa obtenida (0 en paso 1, +1 en paso 2)
            terminated: Si el episodio terminó (True después del paso 2)
            truncated: Si el episodio fue truncado (siempre False)
            info: Información adicional
        """
        self.current_step += 1
        
        if self.current_step == 1:
            # Primer paso: sin recompensa, transición al segundo estado
            self.current_state = 1
            observation = self.current_state
            reward = 0.0
            terminated = False
            truncated = False
            
        elif self.current_step == 2:
            # Segundo paso: recompensa +1, episodio termina
            observation = self.current_state  # Mantener estado 1
            reward = 1.0
            terminated = True
            truncated = False
            
        else:
            # No debería llegar aquí, pero por seguridad
            observation = self.current_state
            reward = 0.0
            terminated = True
            truncated = False
        
        info = {
            'step': self.current_step,
            'state': self.current_state,
            'reward': reward,
            'is_final_step': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Renderiza el entorno mostrando el estado actual."""
        step_name = "Primer paso" if self.current_state == 0 else "Segundo paso"
        print(f"TwoStepDelayedRewardEnv - {step_name} (Estado: {self.current_state}, "
              f"Paso: {self.current_step})")
    
    def get_optimal_q_values(self, discount_factor=0.99):
        """
        Calcula los valores Q óptimos para este entorno.
        
        Args:
            discount_factor: Factor de descuento (gamma)
            
        Returns:
            q_values: Diccionario con valores Q óptimos para cada estado
        """
        # En este entorno, solo hay una acción, así que Q(s,a) = V(s)
        # V(estado_1) = 0 + γ * V(estado_2) = γ * 1 = γ
        # V(estado_2) = 1 (recompensa inmediata)
        
        q_values = {
            0: discount_factor * 1.0,  # Estado 0: valor descontado del estado final
            1: 1.0                     # Estado 1: recompensa inmediata
        }
        
        return q_values
    
    def get_expected_return(self, discount_factor=0.99):
        """
        Calcula el retorno esperado desde el inicio del episodio.
        
        Args:
            discount_factor: Factor de descuento (gamma)
            
        Returns:
            expected_return: Retorno esperado total
        """
        # Retorno = r_1 + γ * r_2 = 0 + γ * 1 = γ
        return discount_factor
