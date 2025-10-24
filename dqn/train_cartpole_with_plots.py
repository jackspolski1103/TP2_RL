#!/usr/bin/env python3
"""
Script de entrenamiento para DQN en CartPole-v1 con gráficos similares a REINFORCE.

Este script entrena un agente DQN y genera los mismos gráficos que REINFORCE:
- Recompensa vs Episodios
- Recompensa vs Wall Time
- Loss vs Episodios
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
import time

from dqn_agent import DQNAgent


def train_dqn_with_plots(episodes: int = 10000,
                        gamma: float = 0.99,
                        lr: float = 5e-4,
                        buffer_capacity: int = 1000,
                        epsilon_start: float = 1.0,
                        epsilon_min: float = 0.01,
                        epsilon_decay: float = 0.0005,
                        batch_size: int = 64,
                        target_update_freq: int = 1000,
                        start_learning: int = 2000,
                        seed: int = 42,
                        save_model: bool = True,
                        log_tensorboard: bool = True) -> Tuple[DQNAgent, List[float], List[float], List[float], List[float]]:
    """
    Entrena un agente DQN en CartPole-v1 y retorna datos para gráficos.
    
    Returns:
        agent: Agente DQN entrenado
        episode_rewards: Lista de recompensas por episodio
        avg_rewards: Lista de recompensas promedio
        losses: Lista de losses
        wall_times: Lista de tiempos de wall clock
    """
    # Configurar semillas
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Crear entorno
    env = gym.make("CartPole-v1")
    
    # Crear agente
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=lr,
        discount_factor=gamma,
        buffer_capacity=buffer_capacity,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        start_learning=start_learning
    )
    
    # Configurar logging
    if log_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/cartpole/dqn_{timestamp}"
        writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard logs: {log_dir}")
    else:
        writer = None
    
    # Listas para tracking
    episode_rewards = []
    avg_rewards = []
    losses = []
    wall_times = []
    
    # Medir tiempo de wall clock
    start_time = time.time()
    
    print(f"🚀 Entrenando DQN en CartPole-v1 por {episodes} episodios...")
    print(f"📊 Configuración:")
    print(f"   Learning Rate: {lr}")
    print(f"   Gamma: {gamma}")
    print(f"   Epsilon Decay: {epsilon_decay}")
    print(f"   Batch Size: {batch_size}")
    print()
    
    # Loop de entrenamiento
    global_step = 0
    for episode in tqdm(range(episodes), desc="Entrenando DQN"):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        
        while True:
            # Seleccionar acción
            action = agent.select_action(state)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar experiencia
            agent.push_transition(state, action, reward, next_state, done)
            
            # Entrenar agente
            loss = agent.train_step(global_step)
            if loss is not None:
                episode_losses.append(loss)
            
            # Actualizar estado y recompensa
            state = next_state
            total_reward += reward
            global_step += 1
            
            if done:
                break
        
        # Decaer epsilon
        agent.decay_epsilon(episode)
        
        # Guardar datos del episodio
        episode_rewards.append(total_reward)
        
        # Calcular promedio móvil
        if len(episode_rewards) >= 10:
            avg_rewards.append(np.mean(episode_rewards[-10:]))
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        # Guardar loss promedio del episodio
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0.0)
        
        # Guardar wall time
        wall_times.append(time.time() - start_time)
        
        # Logging a TensorBoard
        if writer is not None:
            writer.add_scalar("CartPole/Episode_Reward", total_reward, episode)
            writer.add_scalar("CartPole/Avg_Reward_10", avg_rewards[-1], episode)
            if episode_losses:
                writer.add_scalar("CartPole/Loss", np.mean(episode_losses), episode)
            writer.add_scalar("CartPole/Epsilon", agent.epsilon, episode)
        
        # Logging cada 50 episodios
        if (episode + 1) % 50 == 0:
            print(f"Episodio {episode + 1}/{episodes} | "
                  f"Recompensa: {total_reward:.1f} | "
                  f"Promedio (10): {avg_rewards[-1]:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Cerrar entorno y writer
    env.close()
    if writer is not None:
        writer.close()
    
    # Guardar modelo si se solicita
    if save_model:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        model_path = checkpoint_dir / "dqn_cartpole.pt"
        torch.save(agent.q_network.state_dict(), model_path)
        print(f"💾 Modelo guardado en: {model_path}")
    
    return agent, episode_rewards, avg_rewards, losses, wall_times


def create_dqn_plots(episode_rewards: List[float], 
                    avg_rewards: List[float], 
                    losses: List[float], 
                    wall_times: List[float]):
    """
    Crea los mismos gráficos que REINFORCE para DQN.
    """
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Gráfico de recompensa vs episodios
    plt.figure(figsize=(12, 8))
    
    episodes = list(range(1, len(episode_rewards) + 1))
    plt.plot(episodes, episode_rewards, alpha=0.3, color='lightblue', label='Recompensa por Episodio')
    
    # Promedio móvil de 10 episodios
    if len(episode_rewards) >= 10:
        moving_avg_10 = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - 9)
            moving_avg_10.append(np.mean(episode_rewards[start_idx:i+1]))
        plt.plot(episodes, moving_avg_10, color='blue', linewidth=2, label='Promedio Móvil (10 episodios)')
    
    # Promedio móvil de 100 episodios
    if len(episode_rewards) >= 100:
        moving_avg_100 = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - 99)
            moving_avg_100.append(np.mean(episode_rewards[start_idx:i+1]))
        plt.plot(episodes, moving_avg_100, color='red', linewidth=2, label='Promedio Móvil (100 episodios)')
    
    # Línea de referencia
    plt.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Threshold "Resuelto" (195)')
    
    plt.xlabel('Episodios', fontsize=12)
    plt.ylabel('Recompensa', fontsize=12)
    plt.title('DQN en CartPole-v1 - Recompensa vs Episodios', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plt.savefig(plots_dir / "dqn_cartpole_episodes.png", dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: plots/dqn_cartpole_episodes.png")
    plt.show()
    
    # 2. Gráfico de recompensa vs wall time
    plt.figure(figsize=(12, 8))
    
    # Convertir wall times a minutos
    wall_times_minutes = np.array(wall_times) / 60
    
    # Recompensas por episodio vs tiempo
    plt.plot(wall_times_minutes, episode_rewards, alpha=0.3, color='lightblue', label='Recompensa por Episodio')
    
    # Promedio móvil de 10 episodios vs tiempo
    if len(episode_rewards) >= 10:
        plt.plot(wall_times_minutes, moving_avg_10, color='blue', linewidth=2, label='Promedio Móvil (10 episodios)')
    
    # Promedio móvil de 100 episodios vs tiempo
    if len(episode_rewards) >= 100:
        plt.plot(wall_times_minutes, moving_avg_100, color='red', linewidth=2, label='Promedio Móvil (100 episodios)')
    
    # Línea de referencia
    plt.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Threshold "Resuelto" (195)')
    
    plt.xlabel('Tiempo de Entrenamiento (minutos)', fontsize=12)
    plt.ylabel('Recompensa', fontsize=12)
    plt.title('DQN en CartPole-v1 - Recompensa vs Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plt.savefig(plots_dir / "dqn_cartpole_walltime.png", dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: plots/dqn_cartpole_walltime.png")
    plt.show()
    
    # 3. Gráfico de loss vs episodios
    if len(losses) > 0:
        plt.figure(figsize=(12, 6))
        
        # Filtrar losses no nulos
        valid_losses = [loss for loss in losses if loss > 0]
        valid_episodes = [i for i, loss in enumerate(losses) if loss > 0]
        
        if valid_losses:
            plt.plot(valid_episodes, valid_losses, color='red', linewidth=2, label='DQN Loss')
            plt.xlabel('Episodios', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('DQN en CartPole-v1 - Loss vs Episodios', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Guardar gráfico
            plt.savefig(plots_dir / "dqn_cartpole_loss.png", dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado en: plots/dqn_cartpole_loss.png")
            plt.show()


def main():
    """Función principal."""
    print("🎮 Entrenando DQN en CartPole-v1")
    print("="*50)
    
    # Entrenar DQN
    agent, episode_rewards, avg_rewards, losses, wall_times = train_dqn_with_plots()
    
    # Calcular estadísticas finales
    final_avg = np.mean(episode_rewards[-10:])
    final_avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else final_avg
    
    print(f"\n📈 RESULTADOS FINALES:")
    print(f"   Últimos 10 episodios promedio: {final_avg:.3f}")
    print(f"   Últimos 100 episodios promedio: {final_avg_100:.3f}")
    print(f"   Tiempo total de entrenamiento: {wall_times[-1]:.1f} segundos")
    print(f"   Threshold CartPole (195): {'✅ ALCANZADO' if final_avg >= 195 else '❌ NO ALCANZADO'}")
    
    # Crear gráficos
    print(f"\n📊 Generando gráficos...")
    create_dqn_plots(episode_rewards, avg_rewards, losses, wall_times)
    
    print(f"\n🎉 Entrenamiento DQN completado!")


if __name__ == "__main__":
    main()
