"""
Script de entrenamiento para DQN en el entorno CartPole-v1.

Este script entrena un agente DQN en el entorno CartPole-v1 de Gymnasium,
un problema clásico de control donde el objetivo es mantener un poste
equilibrado en un carrito. Implementa el Algoritmo 2 del paper DQN.
"""

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter

from dqn_agent import DQNAgent


def train_cartpole(episodes: int = 1000,
                  gamma: float = 0.99,
                  lr: float = 5e-4,  # Reducir LR para más estabilidad
                  buffer_capacity: int = 100000,  # Buffer más grande
                  epsilon_start: float = 1.0,  # Empezar con más exploración
                  epsilon_min: float = 0.01,  # Epsilon mínimo más bajo
                  epsilon_decay: float = 0.0005,  # Decaimiento más lento
                  batch_size: int = 64,  # Batch más grande para estabilidad
                  target_update_freq: int = 1000,  # Updates menos frecuentes
                  start_learning: int = 2000,  # Más experiencias antes de entrenar
                  seed: int = 42,
                  save_model: bool = True,
                  log_tensorboard: bool = True) -> Tuple[DQNAgent, List[float], List[float]]:
    """
    Entrena un agente DQN en CartPole-v1.
    
    Args:
        episodes: Número de episodios de entrenamiento
        gamma: Factor de descuento
        lr: Tasa de aprendizaje
        buffer_capacity: Capacidad del replay buffer
        epsilon_start: Epsilon inicial
        epsilon_min: Epsilon mínimo
        epsilon_decay: Tasa de decaimiento de epsilon
        batch_size: Tamaño del batch
        target_update_freq: Frecuencia de actualización de target network
        start_learning: Pasos antes de empezar a entrenar
        seed: Semilla para reproducibilidad
        save_model: Si guardar el mejor modelo
        log_tensorboard: Si usar TensorBoard para logging
        
    Returns:
        agent: Agente entrenado
        episode_rewards: Lista de recompensas por episodio
        avg_rewards: Lista de recompensas promedio (ventana de 100)
    """
    # Configurar semillas
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Crear entorno
    env = gym.make('CartPole-v1')
    env.action_space.seed(seed)
    
    # Obtener dimensiones
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"=== Entrenamiento DQN en CartPole-v1 ===")
    print(f"Espacio de estados: {state_size}")
    print(f"Espacio de acciones: {action_size}")
    print(f"Episodios: {episodes}")
    print(f"Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Inicializar agente DQN
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        model_type="mlp",
        learning_rate=lr,
        discount_factor=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        seed=seed
    )
    
    # Configurar TensorBoard
    writer = None
    if log_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/cartpole/{timestamp}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs en: {log_dir}")
    
    # Variables para tracking
    episode_rewards = []
    avg_rewards = []
    losses = []
    global_step = 0
    best_avg_reward = -np.inf
    solved_episode = None
    
    # Criterio de "resuelto": promedio de 475+ en 100 episodios consecutivos (cerca del óptimo)
    SOLVED_THRESHOLD = 475.0  # Más ambicioso, cerca del máximo teórico de 500
    SOLVED_WINDOW = 100
    
    # Loop principal de entrenamiento
    for episode in tqdm(range(episodes), desc="Entrenando DQN"):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_losses = []
        
        while True:
            # Seleccionar acción
            action = agent.select_action(state)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Almacenar transición
            agent.push_transition(state, action, reward, next_state, done)
            
            # Entrenar si es posible
            loss = agent.train_step(global_step)
            if loss is not None:
                episode_losses.append(loss)
                losses.append(loss)
            
            # Actualizar estado y contadores
            state = next_state
            episode_reward += reward
            global_step += 1
            
            if done:
                break
        
        # Decaer epsilon
        agent.decay_epsilon(episode)
        
        # Guardar recompensa del episodio
        episode_rewards.append(episode_reward)
        
        # Calcular recompensa promedio (ventana de 100)
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
            
            # Verificar si se resolvió el entorno
            if avg_reward >= SOLVED_THRESHOLD and solved_episode is None:
                solved_episode = episode
                print(f"\n🎉 Entorno resuelto en episodio {episode}! Promedio: {avg_reward:.2f}")
            
            # Guardar mejor modelo
            if save_model and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                model_path = "checkpoints/dqn_cartpole.pt"
                agent.save_model(model_path)
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        # Logging a TensorBoard
        if writer is not None:
            writer.add_scalar('train/episode_reward', episode_reward, episode)
            writer.add_scalar('train/avg_reward_100', avg_rewards[-1], episode)
            writer.add_scalar('train/epsilon', agent.epsilon, episode)
            
            if episode_losses:
                avg_episode_loss = np.mean(episode_losses)
                writer.add_scalar('train/loss', avg_episode_loss, episode)
        
        # Logging periódico
        if (episode + 1) % 100 == 0:
            avg_reward = avg_rewards[-1]
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episodio {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    # Cerrar TensorBoard
    if writer is not None:
        writer.close()
    
    env.close()
    
    # Resumen final
    final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"\n=== Entrenamiento Completado ===")
    print(f"Recompensa promedio final: {final_avg_reward:.2f}")
    print(f"Mejor recompensa promedio: {best_avg_reward:.2f}")
    if solved_episode is not None:
        print(f"Entorno resuelto en episodio: {solved_episode}")
    else:
        print("Entorno no resuelto en el tiempo dado")
    
    if save_model:
        print(f"Mejor modelo guardado en: checkpoints/dqn_cartpole.pt")
    
    return agent, episode_rewards, avg_rewards


def evaluate_cartpole(model_path: str, 
                     episodes: int = 100, 
                     render: bool = False,
                     seed: int = 42) -> float:
    """
    Evalúa un agente DQN entrenado en CartPole-v1.
    
    Args:
        model_path: Ruta del modelo entrenado
        episodes: Número de episodios de evaluación
        render: Si renderizar la evaluación
        seed: Semilla para reproducibilidad
        
    Returns:
        avg_score: Puntuación promedio
    """
    # Crear entorno
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    env.action_space.seed(seed)
    
    # Obtener dimensiones
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Crear agente y cargar modelo
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        model_type="mlp",
        seed=seed
    )
    
    try:
        agent.load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {model_path}")
        return 0.0
    
    # Evaluar
    episode_rewards = []
    
    print(f"Evaluando agente por {episodes} episodios...")
    
    for episode in tqdm(range(episodes), desc="Evaluando"):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        
        while True:
            # Usar política greedy (epsilon = 0)
            action = agent.select_action(state, epsilon=0.0)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    env.close()
    
    # Calcular estadísticas
    avg_score = np.mean(episode_rewards)
    std_score = np.std(episode_rewards)
    min_score = np.min(episode_rewards)
    max_score = np.max(episode_rewards)
    
    print(f"\n=== Resultados de Evaluación ===")
    print(f"Episodios evaluados: {episodes}")
    print(f"Puntuación promedio: {avg_score:.2f} ± {std_score:.2f}")
    print(f"Puntuación mínima: {min_score:.2f}")
    print(f"Puntuación máxima: {max_score:.2f}")
    print(f"Éxito (>= 475): {np.sum(np.array(episode_rewards) >= 475)}/{episodes} ({100*np.sum(np.array(episode_rewards) >= 475)/episodes:.1f}%)")
    print(f"Bueno (>= 195): {np.sum(np.array(episode_rewards) >= 195)}/{episodes} ({100*np.sum(np.array(episode_rewards) >= 195)/episodes:.1f}%)")
    
    return avg_score


def main():
    """Función principal para entrenar y evaluar en CartPole."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar DQN en CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10000, help="Número de episodios")
    parser.add_argument("--eval", action="store_true", help="Evaluar modelo existente")
    parser.add_argument("--model", type=str, default="checkpoints/dqn_cartpole.pt", 
                       help="Ruta del modelo")
    parser.add_argument("--render", action="store_true", help="Renderizar entorno")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento")
    parser.add_argument("--epsilon-decay", type=float, default=0.001, help="Tasa de decaimiento de epsilon")
    parser.add_argument("--no-tensorboard", action="store_true", help="Desactivar TensorBoard")
    parser.add_argument("--no-save", action="store_true", help="No guardar modelo")
    
    args = parser.parse_args()
    
    if args.eval:
        print("=== Modo Evaluación ===")
        avg_score = evaluate_cartpole(
            model_path=args.model,
            episodes=100,
            render=args.render,
            seed=args.seed
        )
        print(f"\nPuntuación promedio final: {avg_score:.2f}")
    else:
        print("=== Modo Entrenamiento ===")
        agent, episode_rewards, avg_rewards = train_cartpole(
            episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
            save_model=not args.no_save,
            log_tensorboard=not args.no_tensorboard
        )
        
        final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        print(f"\nEntrenamiento completado. Recompensa promedio final: {final_avg:.2f}")
        
        if not args.no_tensorboard:
            print("\nPara ver los logs de TensorBoard:")
            print("tensorboard --logdir runs/cartpole")


if __name__ == "__main__":
    main()
