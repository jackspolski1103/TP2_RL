"""
Implementación del Replay Buffer para DQN.

Este módulo contiene la implementación del buffer de experiencias
que almacena transiciones (s, a, r, s', done) para entrenamiento estable.
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import Optional, Tuple, Dict, Any
import torch


# Definir estructura de experiencia
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Buffer circular para almacenar experiencias de entrenamiento.
    
    Implementa un buffer de tamaño fijo que almacena experiencias
    y permite muestreo aleatorio para entrenar el agente DQN.
    """
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Inicializa el replay buffer.
        
        Args:
            capacity: Capacidad máxima del buffer
            seed: Semilla para reproducibilidad
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Configurar semilla para reproducibilidad
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Añade una experiencia al buffer.
        
        Args:
            state: Estado actual como numpy array
            action: Acción ejecutada (entero)
            reward: Recompensa recibida (float)
            next_state: Siguiente estado como numpy array
            done: Si el episodio terminó (bool)
        """
        # Convertir a numpy arrays si no lo son ya
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
            
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Muestrea un batch aleatorio de experiencias.
        
        Args:
            batch_size: Tamaño del batch
            device: Dispositivo donde colocar los tensores
            
        Returns:
            Tupla de tensores (states, actions, rewards, next_states, dones) o None si no hay suficientes experiencias
        """
        # Verificar que hay suficientes experiencias
        if len(self.buffer) < batch_size:
            return None
        
        # Muestrear experiencias aleatoriamente
        experiences = random.sample(self.buffer, batch_size)
        
        # Separar componentes
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Convertir a tensores PyTorch
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def __len__(self) -> int:
        """Retorna el número actual de experiencias en el buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Verifica si el buffer tiene suficientes experiencias para entrenar.
        
        Args:
            min_size: Tamaño mínimo requerido
            
        Returns:
            ready: True si está listo para entrenar
        """
        return len(self.buffer) >= min_size
    
    def clear(self) -> None:
        """Limpia todas las experiencias del buffer."""
        self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del buffer.
        
        Returns:
            stats: Diccionario con estadísticas del buffer
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_reward': 0.0
            }
        
        # Calcular estadísticas
        rewards = [exp.reward for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay Buffer con muestreo prioritario.
    
    Extiende el ReplayBuffer básico para implementar muestreo
    basado en la magnitud del error TD (Prioritized Experience Replay).
    """
    
    def __init__(self, capacity, batch_size=32, alpha=0.6, beta=0.4, 
                 beta_increment=0.001, seed=None):
        """
        Inicializa el buffer prioritario.
        
        Args:
            capacity: Capacidad máxima del buffer
            batch_size: Tamaño del batch para muestreo
            alpha: Parámetro de priorización (0 = uniforme, 1 = totalmente prioritario)
            beta: Parámetro de corrección de sesgo
            beta_increment: Incremento de beta por paso
            seed: Semilla para reproducibilidad
        """
        super().__init__(capacity, batch_size, seed)
        
        # TODO: Implementar parámetros de priorización
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """
        Añade una experiencia con prioridad al buffer.
        
        Args:
            state: Estado actual
            action: Acción ejecutada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
            priority: Prioridad de la experiencia (usa max_priority si es None)
        """
        # TODO: Implementar adición con prioridad
        super().push(state, action, reward, next_state, done)
        
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority)
    
    def sample(self, batch_size=None):
        """
        Muestrea un batch basado en prioridades.
        
        Args:
            batch_size: Tamaño del batch
            
        Returns:
            batch: Batch de experiencias con pesos de importancia
        """
        # TODO: Implementar muestreo prioritario
        pass
    
    def update_priorities(self, indices, priorities):
        """
        Actualiza las prioridades de experiencias específicas.
        
        Args:
            indices: Índices de las experiencias a actualizar
            priorities: Nuevas prioridades
        """
        # TODO: Implementar actualización de prioridades
        pass
