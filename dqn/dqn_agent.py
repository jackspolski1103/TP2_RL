"""
Implementación del agente Deep Q-Network (DQN).

Este módulo contiene la implementación del algoritmo DQN usando
redes neuronales para aproximar la función de valor Q, siguiendo
estrictamente el Algoritmo 2 del paper original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Optional, Tuple, Union
from pathlib import Path

from utils.replay_buffer import ReplayBuffer


class MLPNetwork(nn.Module):
    """
    Red neuronal MLP mejorada para CartPole y entornos de baja dimensión.
    
    Arquitectura: Linear(input, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, output)
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Inicializa la red MLP.
        
        Args:
            input_size: Tamaño de la observación de entrada
            output_size: Número de acciones (tamaño de salida)
        """
        super(MLPNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # Más neuronas
            nn.ReLU(),
            nn.Linear(128, 128),  # Capa adicional
            nn.ReLU(),
            nn.Linear(128, 64),  # Reducción gradual
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red MLP.
        
        Args:
            x: Tensor de entrada (batch_size, input_size)
            
        Returns:
            q_values: Valores Q para cada acción (batch_size, output_size)
        """
        return self.network(x)


class CNNNetwork(nn.Module):
    """
    Red neuronal CNN para Breakout y entornos visuales.
    
    Arquitectura basada en el paper original de DQN para Atari.
    Entrada: 84x84x4 (4 frames apilados en escala de grises)
    """
    
    def __init__(self, input_channels: int, output_size: int):
        """
        Inicializa la red CNN.
        
        Args:
            input_channels: Número de canales de entrada (típicamente 4 para frame stacking)
            output_size: Número de acciones
        """
        super(CNNNetwork, self).__init__()
        
        # Capas convolucionales según el paper de DQN
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcular el tamaño de la salida de las convoluciones
        # Para entrada 84x84: conv1 -> 20x20, conv2 -> 9x9, conv3 -> 7x7
        conv_output_size = 64 * 7 * 7  # 3136
        
        # Capas totalmente conectadas
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red CNN.
        
        Args:
            x: Tensor de entrada (batch_size, channels, height, width)
            
        Returns:
            q_values: Valores Q para cada acción (batch_size, output_size)
        """
        # Normalizar entrada a [0, 1]
        x = x / 255.0
        
        # Capas convolucionales con ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Aplanar para capas FC
        x = x.view(x.size(0), -1)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class DQNAgent:
    """
    Agente que implementa el algoritmo Deep Q-Network.
    
    Implementa estrictamente el Algoritmo 2 del paper original:
    - Red online y target con hard updates
    - Replay buffer circular
    - Política epsilon-greedy con decaimiento exponencial
    - MSE loss para entrenamiento
    """
    
    def __init__(self, 
                 state_size: Union[int, Tuple[int, ...]], 
                 action_size: int,
                 model_type: str = "mlp",
                 learning_rate: float = 1e-3,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.001,
                 batch_size: int = 32,
                 buffer_capacity: int = 50000,
                 target_update_freq: int = 1000,
                 start_learning: int = 1000,
                 seed: Optional[int] = None):
        """
        Inicializa el agente DQN.
        
        Args:
            state_size: Dimensión del espacio de estados (int para MLP, tuple para CNN)
            action_size: Número de acciones posibles
            model_type: Tipo de modelo ("mlp" o "cnn")
            learning_rate: Tasa de aprendizaje para Adam
            discount_factor: Factor de descuento (gamma)
            epsilon_start: Probabilidad inicial de exploración
            epsilon_min: Valor mínimo de epsilon
            epsilon_decay: Tasa de decaimiento exponencial de epsilon
            batch_size: Tamaño del batch para entrenamiento
            buffer_capacity: Capacidad del replay buffer
            target_update_freq: Frecuencia de actualización de target network (en steps)
            start_learning: Número de steps antes de empezar a entrenar
            seed: Semilla para reproducibilidad
        """
        # Configurar semillas
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Parámetros del agente
        self.state_size = state_size
        self.action_size = action_size
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.start_learning = start_learning
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN usando dispositivo: {self.device}")
        
        # Inicializar redes neuronales
        if model_type == "mlp":
            if not isinstance(state_size, int):
                raise ValueError("Para MLP, state_size debe ser un entero")
            self.q_network = MLPNetwork(state_size, action_size).to(self.device)
            self.target_network = MLPNetwork(state_size, action_size).to(self.device)
        elif model_type == "cnn":
            if not isinstance(state_size, tuple) or len(state_size) != 3:
                raise ValueError("Para CNN, state_size debe ser una tupla (channels, height, width)")
            channels, height, width = state_size
            self.q_network = CNNNetwork(channels, action_size).to(self.device)
            self.target_network = CNNNetwork(channels, action_size).to(self.device)
        else:
            raise ValueError(f"model_type debe ser 'mlp' o 'cnn', recibido: {model_type}")
        
        # Copiar pesos iniciales a target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, seed=seed)
        
        # Contadores y estadísticas
        self.steps = 0
        self.training_stats = {
            'episodes': 0,
            'losses': [],
            'epsilon_history': []
        }
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual del entorno
            epsilon: Valor de epsilon a usar (usa self.epsilon si es None)
        
        Returns:
            action: Acción seleccionada (entero)
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            # Exploración: acción aleatoria
            return random.randint(0, self.action_size - 1)
        else:
            # Explotación: acción greedy según Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def push_transition(self, state: np.ndarray, action: int, reward: float, 
                       next_state: np.ndarray, done: bool) -> None:
        """
        Almacena una transición en el replay buffer.
        
        Args:
            state: Estado anterior
            action: Acción ejecutada
            reward: Recompensa recibida
            next_state: Nuevo estado
            done: Si el episodio terminó
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, global_step: int) -> Optional[float]:
        """
        Realiza un paso de entrenamiento si hay suficientes experiencias.
        
        Args:
            global_step: Paso global actual (para logging y target updates)
        
        Returns:
            loss: Pérdida del entrenamiento (None si no se entrenó)
        """
        # Verificar si es momento de entrenar
        if global_step < self.start_learning:
            return None
            
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Muestrear batch del replay buffer
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        # Calcular Q-values actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calcular Q-values target usando target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Calcular loss (MSE como en el paper original)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar target network si es necesario
        if global_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Guardar estadísticas
        self.training_stats['losses'].append(loss.item())
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Actualiza la target network copiando pesos de la main network (hard update)."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self, episode: int) -> None:
        """
        Decae el valor de epsilon usando decaimiento exponencial.
        
        Args:
            episode: Número de episodio actual
        """
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                      np.exp(-self.epsilon_decay * episode)
        self.training_stats['epsilon_history'].append(self.epsilon)
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar estado completo
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'steps': self.steps,
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo pre-entrenado.
        
        Args:
            filepath: Ruta del archivo del modelo
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restaurar redes
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaurar parámetros
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.training_stats = checkpoint['training_stats']
