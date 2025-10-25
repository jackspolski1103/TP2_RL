"""
Script de entrenamiento para DQN en el entorno Breakout-v5.

Este script entrena un agente DQN en el entorno Breakout de Atari,
usando preprocesamiento de imágenes, frame stacking y una red CNN.
Implementa el Algoritmo 2 del paper DQN para entornos visuales.
"""

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Deque
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dqn_agent import DQNAgent
from minatar_wrapper import MinAtarWrapper


class FrameStack:
    """
    Wrapper para apilar frames consecutivos.
    
    Mantiene un buffer de los últimos N frames y los devuelve
    como un tensor apilado para capturar información temporal.
    """
    
    def __init__(self, num_frames: int = 4):
        """
        Inicializa el frame stack.
        
        Args:
            num_frames: Número de frames a apilar
        """
        self.num_frames = num_frames
        self.frames: Deque[np.ndarray] = deque(maxlen=num_frames)
    
    def reset(self, frame: np.ndarray) -> np.ndarray:
        """
        Reinicia el stack con el frame inicial.
        
        Args:
            frame: Frame inicial preprocesado (84x84)
            
        Returns:
            stacked_frames: Stack de frames (4x84x84)
        """
        # Llenar con el mismo frame inicial
        for _ in range(self.num_frames):
            self.frames.append(frame)
        
        return self.get_state()
    
    def step(self, frame: np.ndarray) -> np.ndarray:
        """
        Añade un nuevo frame al stack.
        
        Args:
            frame: Nuevo frame preprocesado (84x84)
            
        Returns:
            stacked_frames: Stack actualizado de frames (4x84x84)
        """
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Obtiene el estado actual como stack de frames.
        
        Returns:
            stacked_frames: Array de shape (16, 10, 10) para MinAtar CNN
        """
        # Reorganizar de (4, 10, 10, 4) a (16, 10, 10)
        frames = np.array(list(self.frames))  # Shape: (4, 10, 10, 4)
        # Reorganizar: (4, 10, 10, 4) -> (4*4, 10, 10) = (16, 10, 10)
        stacked = frames.transpose(0, 3, 1, 2)  # (4, 4, 10, 10)
        stacked = stacked.reshape(16, 10, 10)  # (16, 10, 10)
        return stacked


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocesa un frame de MinAtar para entrenamiento.
    
    Args:
        frame: Frame original de MinAtar (10x10x4)
        
    Returns:
        processed_frame: Frame preprocesado (10x10x4) ya normalizado
    """
    # MinAtar ya viene preprocesado y normalizado
    # Solo aseguramos que sea float32
    return frame.astype(np.float32)


def train_breakout(episodes: int = 1000,
                  gamma: float = 0.99,
                  lr: float = 5e-3,  # Aumentado significativamente para aprendizaje más rápido
                  buffer_capacity: int = 10000,
                  epsilon_start: float = 1.0,
                  epsilon_min: float = 0.001,  # Reducido para más explotación
                  epsilon_decay: float = 0.005,  # Decaimiento mucho más rápido
                  batch_size: int = 128,  # Aumentado para mejor estabilidad y aprendizaje
                  target_update_freq: int = 500,  # Actualización más frecuente
                  start_learning: int = 500,  # Empezar a entrenar aún antes
                  seed: int = 42,
                  save_model: bool = True,
                  log_tensorboard: bool = True) -> Tuple[DQNAgent, List[float], List[float]]:
    """
    Entrena un agente DQN en Breakout-v5.
    
    Args:
        episodes: Número de episodios de entrenamiento
        gamma: Factor de descuento
        lr: Tasa de aprendizaje (más baja para Atari)
        buffer_capacity: Capacidad del replay buffer (más grande para Atari)
        epsilon_start: Epsilon inicial
        epsilon_min: Epsilon mínimo
        epsilon_decay: Tasa de decaimiento de epsilon (más lenta)
        batch_size: Tamaño del batch
        target_update_freq: Frecuencia de actualización de target network
        start_learning: Pasos antes de empezar a entrenar (más alto para Atari)
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
    
    # Crear entorno Breakout usando MinAtar
    env = MinAtarWrapper('breakout')
    env.action_space.seed(seed)
    
    # Obtener dimensiones
    action_size = env.action_space.n
    # Para MinAtar: cada frame es 10x10x4, y apilamos 4 frames
    # El formato para CNN debe ser (channels, height, width)
    # Necesitamos reorganizar: (4, 10, 10, 4) -> (4*4, 10, 10) = (16, 10, 10)
    state_size = (16, 10, 10)  # 16 canales (4 frames * 4 canales por frame)
    
    print(f"=== Entrenamiento DQN en MinAtar Breakout ===")
    print(f"Espacio de estados: {state_size}")
    print(f"Espacio de acciones: {action_size}")
    print(f"Episodios: {episodes}")
    
    # Verificación detallada de GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔥 CUDA versión: {torch.version.cuda}")
    else:
        print("⚠️  GPU no disponible, usando CPU")
    
    print("NOTA: MinAtar Breakout es más rápido que Atari Breakout!")
    
    # Inicializar agente DQN con CNN
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        model_type="cnn",
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
        log_dir = f"runs/breakout/{timestamp}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs en: {log_dir}")
    
    # Variables para tracking
    episode_rewards = []
    avg_rewards = []
    losses = []
    global_step = 0
    best_avg_reward = -np.inf
    solved_episode = None
    
    # Criterio de "resuelto": terminar nivel sin perder 3 vidas consecutivas
    # Para simplificar, usaremos un threshold de puntuación
    SOLVED_THRESHOLD = 30.0  # Breakout es más difícil
    SOLVED_WINDOW = 100
    
    # Inicializar frame stack
    frame_stack = FrameStack(num_frames=4)
    
    # Loop principal de entrenamiento
    for episode in tqdm(range(episodes), desc="Entrenando DQN"):
        # Reiniciar entorno
        raw_state, _ = env.reset(seed=seed + episode)
        
        # Preprocesar frame inicial y crear stack
        processed_frame = preprocess_frame(raw_state)
        state = frame_stack.reset(processed_frame)
        
        episode_reward = 0
        episode_losses = []
        lives = 5  # Breakout empieza con 5 vidas
        
        while True:
            # Seleccionar acción
            action = agent.select_action(state)
            
            # Ejecutar acción
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Preprocesar nuevo frame
            next_processed_frame = preprocess_frame(next_raw_state)
            next_state = frame_stack.step(next_processed_frame)
            
            # Modificar recompensa para Breakout (clipping)
            clipped_reward = np.clip(reward, -1, 1)
            
            # Detectar pérdida de vida como episodio terminado (para aprendizaje)
            life_lost = False
            if 'lives' in info:
                if info['lives'] < lives:
                    life_lost = True
                    lives = info['lives']
            
            # Almacenar transición (usar life_lost como done para aprendizaje)
            agent.push_transition(state, action, clipped_reward, next_state, life_lost or done)
            
            # Entrenar si es posible
            loss = agent.train_step(global_step)
            if loss is not None:
                episode_losses.append(loss)
                losses.append(loss)
            
            # Actualizar estado y contadores
            state = next_state
            episode_reward += reward  # Usar recompensa original para tracking
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
                model_path = "checkpoints/dqn_breakout.pt"
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
        if (episode + 1) % 50 == 0:  # Más frecuente para Breakout
            avg_reward = avg_rewards[-1]
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episodio {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Steps: {global_step}")
    
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
        print(f"Mejor modelo guardado en: checkpoints/dqn_breakout.pt")
    
    return agent, episode_rewards, avg_rewards


def evaluate_breakout(model_path: str, 
                     episodes: int = 10, 
                     render: bool = False,
                     seed: int = 42) -> float:
    """
    Evalúa un agente DQN entrenado en MinAtar Breakout.
    
    Args:
        model_path: Ruta del modelo entrenado
        episodes: Número de episodios de evaluación
        render: Si renderizar la evaluación
        seed: Semilla para reproducibilidad
        
    Returns:
        avg_score: Puntuación promedio
    """
    # Crear entorno MinAtar
    env = MinAtarWrapper('breakout')
    env.action_space.seed(seed)
    
    # Obtener dimensiones
    action_size = env.action_space.n
    state_size = (16, 10, 10)  # 16 canales (4 frames * 4 canales por frame)
    
    # Crear agente y cargar modelo
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        model_type="cnn",
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
    frame_stack = FrameStack(num_frames=4)
    
    print(f"Evaluando agente por {episodes} episodios...")
    
    for episode in tqdm(range(episodes), desc="Evaluando"):
        raw_state, _ = env.reset(seed=seed + episode)
        processed_frame = preprocess_frame(raw_state)
        state = frame_stack.reset(processed_frame)
        episode_reward = 0
        
        while True:
            # Usar política greedy (epsilon = 0)
            action = agent.select_action(state, epsilon=0.0)
            
            # Ejecutar acción
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Preprocesar y actualizar estado
            next_processed_frame = preprocess_frame(next_raw_state)
            state = frame_stack.step(next_processed_frame)
            
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
    
    return avg_score


def main():
    """Función principal para entrenar y evaluar en Breakout."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar DQN en Breakout-v5")
    
    # Parámetros básicos
    parser.add_argument("--episodes", type=int, default=10000, help="Número de episodios")
    parser.add_argument("--eval", action="store_true", help="Evaluar modelo existente")
    parser.add_argument("--model", type=str, default="checkpoints/dqn_breakout.pt", 
                       help="Ruta del modelo")
    parser.add_argument("--render", action="store_true", help="Renderizar entorno")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    
    # Parámetros de aprendizaje
    parser.add_argument("--lr", type=float, default=5e-3, help="Tasa de aprendizaje")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Epsilon inicial")
    parser.add_argument("--epsilon-min", type=float, default=0.001, help="Epsilon mínimo")
    parser.add_argument("--epsilon-decay", type=float, default=0.005, help="Tasa de decaimiento de epsilon")
    
    # Parámetros del buffer y entrenamiento
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Capacidad del replay buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño del batch")
    parser.add_argument("--target-update-freq", type=int, default=500, help="Frecuencia de actualización de target network")
    parser.add_argument("--start-learning", type=int, default=500, help="Pasos antes de empezar a entrenar")
    
    # Parámetros de logging y guardado
    parser.add_argument("--no-tensorboard", action="store_true", help="Desactivar TensorBoard")
    parser.add_argument("--no-save", action="store_true", help="No guardar modelo")
    
    args = parser.parse_args()
    
    if args.eval:
        print("=== Modo Evaluación ===")
        avg_score = evaluate_breakout(
            model_path=args.model,
            episodes=10,
            render=args.render,
            seed=args.seed
        )
        print(f"\nPuntuación promedio final: {avg_score:.2f}")
    else:
        print("=== Modo Entrenamiento ===")
        print("NOTA: Breakout puede tardar varias horas en entrenar!")
        agent, episode_rewards, avg_rewards = train_breakout(
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            buffer_capacity=args.buffer_capacity,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            start_learning=args.start_learning,
            seed=args.seed,
            save_model=not args.no_save,
            log_tensorboard=not args.no_tensorboard
        )
        
        final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        print(f"\nEntrenamiento completado. Recompensa promedio final: {final_avg:.2f}")
        
        if not args.no_tensorboard:
            print("\nPara ver los logs de TensorBoard:")
            print("tensorboard --logdir runs/breakout")


if __name__ == "__main__":
    main()
