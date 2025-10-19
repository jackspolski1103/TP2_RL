"""
Wrapper de Gymnasium para entornos MinAtar.

Este módulo proporciona un wrapper que convierte los entornos de MinAtar
en entornos compatibles con Gymnasium, permitiendo su uso con agentes DQN.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict
import minatar


class MinAtarWrapper(gym.Env):
    """
    Wrapper de Gymnasium para entornos MinAtar.
    
    Convierte la API de MinAtar a la API estándar de Gymnasium,
    permitiendo el uso con agentes DQN existentes.
    """
    
    def __init__(self, game_name: str, **kwargs):
        """
        Inicializa el wrapper de MinAtar.
        
        Args:
            game_name: Nombre del juego de MinAtar (ej: 'breakout')
            **kwargs: Argumentos adicionales para el entorno
        """
        super().__init__()
        
        # Crear entorno MinAtar
        self.env = minatar.Environment(game_name)
        
        # Configurar espacios
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=self.env.state_shape(), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.env.num_actions())
        
        # Estado actual
        self.current_state = None
        self.terminated = False
        self.truncated = False
        
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia el entorno.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
            
        Returns:
            observation: Estado inicial
            info: Información adicional
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Reiniciar entorno MinAtar
        self.env.reset()
        self.terminated = False
        self.truncated = False
        
        # Obtener estado inicial usando state()
        self.current_state = self.env.state().astype(np.float32)
        
        info = {}
        return self.current_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            observation: Nuevo estado
            reward: Recompensa obtenida
            terminated: Si el episodio terminó
            truncated: Si el episodio fue truncado
            info: Información adicional
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Episodio ya terminado. Llama a reset() primero.")
        
        # Ejecutar acción en MinAtar
        result = self.env.act(action)
        if len(result) == 2:
            reward, terminated = result
        else:
            reward, terminated = result[1], result[2] if len(result) > 2 else False
        
        # Actualizar estado obteniendo el estado actual después de la acción
        self.current_state = self.env.state().astype(np.float32)
        self.terminated = terminated
        self.truncated = False  # MinAtar no tiene truncamiento
        
        # El estado ya está actualizado
        observation = self.current_state
        
        info = {}
        return observation, reward, terminated, self.truncated, info
    
    def render(self, mode: str = 'human') -> Any:
        """
        Renderiza el entorno (no implementado para MinAtar).
        
        Args:
            mode: Modo de renderizado
            
        Returns:
            None
        """
        if mode == 'human':
            # MinAtar no tiene renderizado visual directo
            pass
        return None
    
    def close(self):
        """Cierra el entorno."""
        if hasattr(self, 'env'):
            # MinAtar no tiene método close, pero podemos limpiar referencias
            self.env = None
    
    def seed(self, seed: int = None):
        """Establece la semilla (compatibilidad con versiones anteriores)."""
        if seed is not None:
            np.random.seed(seed)


def make_minatar_env(game_name: str, **kwargs) -> MinAtarWrapper:
    """
    Función de conveniencia para crear entornos MinAtar.
    
    Args:
        game_name: Nombre del juego de MinAtar
        **kwargs: Argumentos adicionales
        
    Returns:
        MinAtarWrapper: Entorno envuelto
    """
    return MinAtarWrapper(game_name, **kwargs)


# Registrar entornos MinAtar con Gymnasium
def register_minatar_envs():
    """Registra entornos MinAtar con Gymnasium."""
    
    # Lista de juegos disponibles en MinAtar
    minatar_games = ['breakout', 'asterix', 'freeway', 'seaquest', 'space_invaders']
    
    for game in minatar_games:
        env_id = f"MinAtar/{game.title()}-v0"
        
        def make_env(game_name=game):
            return MinAtarWrapper(game_name)
        
        gym.register(
            id=env_id,
            entry_point=make_env,
            max_episode_steps=1000,  # Límite razonable
        )


# Registrar automáticamente los entornos
register_minatar_envs()
