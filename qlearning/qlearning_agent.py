"""
Implementación del agente Q-Learning tradicional para FrozenLake-v1.

Este módulo contiene la implementación del algoritmo Q-Learning
usando una tabla de valores Q para entornos con espacios discretos,
específicamente optimizado para FrozenLake-v1 (4x4).
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


class QLearningAgent:
    """
    Agente que implementa el algoritmo Q-Learning tradicional.
    
    Utiliza una tabla de valores Q para aprender la función de valor
    acción-estado en entornos con espacios de estados y acciones discretos.
    """
    
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            n_actions: Número de acciones posibles
            learning_rate: Tasa de aprendizaje (alpha)
            discount_factor: Factor de descuento (gamma)
            epsilon: Probabilidad inicial de exploración
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
        """
        # TODO: Inicializar parámetros del agente
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # TODO: Inicializar tabla Q
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'epsilon_history': []
        }
    
    def get_action(self, state, training=True):
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual del entorno
            training: Si está en modo entrenamiento (usa epsilon-greedy)
                     o evaluación (usa política greedy)
        
        Returns:
            action: Acción seleccionada
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(0, self.n_actions)
        else:
            # Explotación: acción greedy (mejor valor Q)
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la tabla Q usando la ecuación de Q-Learning.
        
        Args:
            state: Estado anterior
            action: Acción ejecutada
            reward: Recompensa recibida
            next_state: Nuevo estado
            done: Si el episodio terminó
            
        Returns:
            td_error: Error temporal (diferencia entre target y valor actual)
        """
        # Valor Q actual
        current_q = self.q_table[state][action]
        
        # Calcular el target
        if done:
            # Si el episodio terminó, no hay estado siguiente
            target = reward
        else:
            # Q-Learning: r + γ * max_a'(Q(s',a'))
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Calcular error temporal (TD error)
        td_error = target - current_q
        
        # Actualizar Q(s,a) usando la regla de Q-Learning
        self.q_table[state][action] = current_q + self.learning_rate * td_error
        
        return abs(td_error)  # Retornar valor absoluto del error
    
    def decay_epsilon(self, episode, decay_rate=0.001):
        """
        Decae el valor de epsilon usando decaimiento exponencial.
        
        Args:
            episode: Número de episodio actual
            decay_rate: Tasa de decaimiento
        """
        # Decaimiento exponencial: ε = min_epsilon + (max_epsilon - min_epsilon) * exp(-decay_rate * episode)
        max_epsilon = 1.0
        self.epsilon = self.epsilon_min + (max_epsilon - self.epsilon_min) * np.exp(-decay_rate * episode)
    
    def save_model(self, filepath):
        """
        Guarda la tabla Q en un archivo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        # Convertir defaultdict a dict normal para serialización
        q_table_dict = dict(self.q_table)
        
        # Crear directorio si no existe
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar usando pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'n_actions': self.n_actions,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'training_stats': self.training_stats
            }, f)
    
    def load_model(self, filepath):
        """
        Carga una tabla Q desde un archivo.
        
        Args:
            filepath: Ruta del archivo del modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restaurar parámetros
        self.n_actions = data['n_actions']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.training_stats = data['training_stats']
        
        # Restaurar tabla Q como defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for state, values in data['q_table'].items():
            self.q_table[state] = values
    
    def get_q_value(self, state, action):
        """
        Obtiene el valor Q para un par estado-acción.
        
        Args:
            state: Estado
            action: Acción
            
        Returns:
            q_value: Valor Q correspondiente
        """
        return self.q_table[state][action]
    
    def get_state_values(self, state):
        """
        Obtiene todos los valores Q para un estado.
        
        Args:
            state: Estado
            
        Returns:
            values: Array con valores Q para todas las acciones
        """
        return self.q_table[state].copy()


def train_qlearning(n_episodes=5000, alpha=0.8, gamma=0.99, 
                   epsilon_min=0.1, decay_rate=0.001, 
                   eval_interval=100, verbose=True):
    """
    Entrena un agente Q-Learning en FrozenLake-v1 (4x4).
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        alpha: Tasa de aprendizaje
        gamma: Factor de descuento
        epsilon_min: Valor mínimo de epsilon
        decay_rate: Tasa de decaimiento de epsilon
        eval_interval: Intervalo para calcular recompensa promedio
        verbose: Si mostrar progreso
        
    Returns:
        q_table: Tabla Q entrenada (como dict)
        rewards: Lista de recompensas por episodio
        avg_rewards: Lista de recompensas promedio cada eval_interval episodios
        losses: Lista de pérdidas promedio cada eval_interval episodios
    """
    # Inicializar entorno FrozenLake-v1 (4x4) - DETERMINÍSTICO
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    
    # Obtener dimensiones del entorno
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    if verbose:
        print(f"Entorno FrozenLake-v1 (4x4)")
        print(f"Estados: {n_states}, Acciones: {n_actions}")
        print(f"Entrenando por {n_episodes} episodios...")
    
    # Inicializar agente
    agent = QLearningAgent(
        n_actions=n_actions,
        learning_rate=alpha,
        discount_factor=gamma,
        epsilon=1.0,  # Epsilon inicial
        epsilon_min=epsilon_min
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    episode_rewards_buffer = []
    losses = []
    episode_losses_buffer = []
    
    # Loop de entrenamiento
    for episode in tqdm(range(n_episodes), desc="Entrenando Q-Learning", disable=not verbose):
        # Reiniciar entorno
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        step_count = 0
        done = False
        
        while not done:
            # Seleccionar acción usando epsilon-greedy
            action = agent.get_action(state, training=True)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar tabla Q y obtener error temporal
            td_error = agent.update(state, action, reward, next_state, done)
            total_loss += td_error
            step_count += 1
            
            # Actualizar estado y recompensa total
            state = next_state
            total_reward += reward
        
        # Guardar recompensa y pérdida del episodio
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular pérdida promedio por paso
        avg_episode_loss = total_loss / max(step_count, 1)
        episode_losses_buffer.append(avg_episode_loss)
        
        # Decaer epsilon
        agent.decay_epsilon(episode, decay_rate)
        
        # Calcular recompensa y pérdida promedio cada eval_interval episodios
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards_buffer)
            avg_loss = np.mean(episode_losses_buffer)
            avg_rewards.append(avg_reward)
            losses.append(avg_loss)
            episode_rewards_buffer = []  # Resetear buffer
            episode_losses_buffer = []  # Resetear buffer
            
            if verbose and (episode + 1) % (eval_interval * 5) == 0:
                print(f"Episodio {episode + 1}/{n_episodes}, "
                      f"Recompensa promedio: {avg_reward:.3f}, "
                      f"Pérdida promedio: {avg_loss:.3f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Convertir tabla Q a dict normal para retornar
    q_table_dict = dict(agent.q_table)
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Recompensa promedio final: {np.mean(rewards[-eval_interval:]):.3f}")
        print(f"Pérdida promedio final: {losses[-1]:.3f}")
        print(f"Epsilon final: {agent.epsilon:.3f}")
    
    return q_table_dict, rewards, avg_rewards, losses


def evaluate_qlearning(q_table, n_eval_episodes=100, epsilon=0.0, verbose=True):
    """
    Evalúa un agente Q-Learning entrenado en FrozenLake-v1.
    
    Args:
        q_table: Tabla Q entrenada (dict)
        n_eval_episodes: Número de episodios de evaluación
        epsilon: Epsilon para evaluación (0.0 = política greedy)
        verbose: Si mostrar progreso
        
    Returns:
        success_rate: Porcentaje de éxito (promedio de recompensas)
    """
    # Inicializar entorno - DETERMINÍSTICO
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    n_actions = env.action_space.n
    
    # Crear agente con tabla Q cargada
    agent = QLearningAgent(n_actions=n_actions, epsilon=epsilon)
    
    # Cargar tabla Q
    agent.q_table = defaultdict(lambda: np.zeros(n_actions))
    for state, values in q_table.items():
        agent.q_table[state] = values
    
    # Evaluar
    total_rewards = []
    
    for episode in tqdm(range(n_eval_episodes), desc="Evaluando", disable=not verbose):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Usar política (epsilon-greedy o greedy)
            action = agent.get_action(state, training=True if epsilon > 0 else False)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    env.close()
    
    # Calcular tasa de éxito
    success_rate = np.mean(total_rewards)
    
    if verbose:
        print(f"Evaluación completada sobre {n_eval_episodes} episodios")
        print(f"Tasa de éxito: {success_rate:.3f} ({success_rate*100:.1f}%)")
    
    return success_rate


def plot_training_curves(avg_rewards, eval_interval=100, save_path="plots/qlearning_rewards.png"):
    """
    Grafica las curvas de entrenamiento del Q-Learning.
    
    Args:
        avg_rewards: Lista de recompensas promedio
        eval_interval: Intervalo de evaluación usado
        save_path: Ruta donde guardar el gráfico
    """
    # Crear directorio si no existe
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Crear figura
    plt.figure(figsize=(12, 6))
    
    # Calcular episodios correspondientes
    episodes = np.arange(eval_interval, len(avg_rewards) * eval_interval + 1, eval_interval)
    
    # Graficar recompensa promedio
    plt.plot(episodes, avg_rewards, 'b-', linewidth=2, label='Recompensa Promedio')
    
    # Agregar línea de tendencia suavizada
    if len(avg_rewards) > 10:
        # Usar ventana móvil para suavizar
        window_size = max(3, len(avg_rewards) // 10)
        smoothed = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        plt.plot(smoothed_episodes, smoothed, 'r-', linewidth=3, alpha=0.8, 
                label=f'Tendencia (ventana {window_size})')
    
    # Configurar gráfico
    plt.xlabel('Episodios', fontsize=12)
    plt.ylabel('Recompensa Promedio', fontsize=12)
    plt.title('Evolución del Entrenamiento Q-Learning en FrozenLake-v1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Agregar información adicional
    final_performance = avg_rewards[-1] if avg_rewards else 0
    plt.text(0.02, 0.98, f'Rendimiento Final: {final_performance:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {save_path}")
    
    # Mostrar figura
    plt.show()


if __name__ == "__main__":
    """Ejecución directa del script."""
    print("=== Entrenamiento Q-Learning en FrozenLake-v1 ===\n")
    
    # Entrenar agente
    q_table, rewards, avg_rewards = train_qlearning()
    
    # Evaluar agente
    print("\n=== Evaluación del Agente Entrenado ===")
    success_rate = evaluate_qlearning(q_table)
    print(f"Success rate (greedy policy): {success_rate*100:.2f}%")
    
    # Generar gráficos
    print("\n=== Generando Gráficos ===")
    plot_training_curves(avg_rewards)
    
    print("\n¡Entrenamiento y evaluación completados!")
