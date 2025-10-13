"""
Entorno con observaciones aleatorias y recompensa binaria.

Este entorno devuelve observaciones aleatorias (+1 o -1) y la recompensa
es igual a la observación. Útil para probar si la red puede aprender
a predecir un valor dependiente de la entrada.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RandomObsBinaryRewardEnv(gym.Env):
    """
    Entorno con observaciones aleatorias y recompensa binaria para testing de DQN.
    
    Características:
    - Acción única (solo hay una acción posible)
    - Observaciones aleatorias: +1 o -1 (equiprobables)
    - Duración: 1 paso por episodio
    - Recompensa: igual a la observación (+1 o -1)
    
    Este entorno sirve para ver si la red puede aprender a predecir
    un valor dependiente de la entrada cuando hay variabilidad.
    """
    
    def __init__(self, seed=None):
        """
        Inicializa el entorno con observaciones aleatorias.
        
        Args:
            seed: Semilla para reproducibilidad
        """
        super().__init__()
        
        # Espacio de observación: valores discretos 0 o 1 (que mapean a -1 o +1)
        self.observation_space = spaces.Discrete(2)
        
        # Espacio de acción: una sola acción posible
        self.action_space = spaces.Discrete(1)
        
        # Estado interno
        self.current_step = 0
        self.current_obs = None
        
        # Configurar generador de números aleatorios
        self.np_random = None
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed=None):
        """
        Configura la semilla para reproducibilidad.
        
        Args:
            seed: Semilla para el generador de números aleatorios
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales (no usadas)
        
        Returns:
            observation: Observación inicial aleatoria (0 o 1)
            info: Información adicional
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        
        # Generar observación aleatoria: 0 o 1 (que representa -1 o +1)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(None)
        
        self.current_obs = self.np_random.integers(0, 2)  # 0 o 1
        
        info = {
            'obs_value': self._obs_to_value(self.current_obs)
        }
        
        return self.current_obs, info
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar (ignorada, solo hay una acción)
            
        Returns:
            observation: Nueva observación aleatoria (0 o 1)
            reward: Recompensa obtenida (igual a la observación: -1 o +1)
            terminated: Si el episodio terminó (siempre True después de 1 paso)
            truncated: Si el episodio fue truncado (siempre False)
            info: Información adicional
        """
        self.current_step += 1
        
        # La recompensa es igual al valor de la observación actual
        reward = self._obs_to_value(self.current_obs)
        
        # Generar nueva observación aleatoria para el siguiente episodio
        # (aunque el episodio termina, esto es para consistencia)
        next_obs = self.np_random.integers(0, 2)
        
        # El episodio termina después de 1 paso
        terminated = True
        truncated = False
        
        info = {
            'step': self.current_step,
            'obs_value': self._obs_to_value(self.current_obs),
            'reward': reward,
            'next_obs_value': self._obs_to_value(next_obs)
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def _obs_to_value(self, obs):
        """
        Convierte la observación discreta (0 o 1) a valor real (-1 o +1).
        
        Args:
            obs: Observación discreta (0 o 1)
            
        Returns:
            value: Valor real (-1 o +1)
        """
        return 2 * obs - 1  # 0 -> -1, 1 -> +1
    
    def render(self):
        """Renderiza el entorno mostrando el estado actual."""
        obs_value = self._obs_to_value(self.current_obs) if self.current_obs is not None else "None"
        print(f"RandomObsBinaryRewardEnv - Paso: {self.current_step}, "
              f"Obs: {self.current_obs} (valor: {obs_value}), Reward: {obs_value}")
