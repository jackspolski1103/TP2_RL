"""
Entrenamiento de DQN en entornos personalizados.

Este script entrena agentes DQN en los tres entornos personalizados:
1. ConstantRewardEnv: Recompensa constante
2. RandomObsBinaryRewardEnv: Observaciones aleatorias con recompensa binaria
3. TwoStepDelayedRewardEnv: Recompensa retrasada en dos pasos

Cada entorno tiene características específicas que permiten probar diferentes
aspectos del algoritmo DQN.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

# Importar entornos personalizados
from envs import ConstantRewardEnv, RandomObsBinaryRewardEnv, TwoStepDelayedRewardEnv
from dqn_agent import DQNAgent


def state_to_one_hot(state: int, num_states: int) -> np.ndarray:
    """
    Convierte un estado discreto a one-hot encoding.
    
    Args:
        state: Estado discreto
        num_states: Número total de estados posibles
        
    Returns:
        one_hot: Vector one-hot del estado
    """
    one_hot = np.zeros(num_states)
    one_hot[state] = 1.0
    return one_hot


def train_constant_env(n_episodes: int = 1000, learning_rate: float = 1e-3, 
                      discount_factor: float = 0.99, epsilon_start: float = 1.0,
                      epsilon_min: float = 0.01, epsilon_decay: float = 0.001,
                      batch_size: int = 32, buffer_capacity: int = 10000,
                      target_update_freq: int = 100, start_learning: int = 100,
                      verbose: bool = True) -> Tuple[DQNAgent, List[float], List[float], List[float]]:
    """
    Entrena DQN en ConstantRewardEnv.
    
    Este entorno siempre devuelve recompensa +1, por lo que el agente debería
    aprender que el valor Q es constante e igual a 1.
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        learning_rate: Tasa de aprendizaje
        discount_factor: Factor de descuento
        epsilon_start: Epsilon inicial
        epsilon_min: Epsilon mínimo
        epsilon_decay: Tasa de decaimiento de epsilon
        batch_size: Tamaño del batch
        buffer_capacity: Capacidad del replay buffer
        target_update_freq: Frecuencia de actualización de target network
        start_learning: Episodios antes de empezar a entrenar
        verbose: Si mostrar progreso
        
    Returns:
        agent: Agente DQN entrenado
        rewards: Recompensas por episodio
        avg_rewards: Recompensas promedio
        losses: Pérdidas por episodio
    """
    if verbose:
        print("🎯 Entrenando DQN en ConstantRewardEnv")
        print("=" * 50)
    
    # Crear entorno
    env = ConstantRewardEnv()
    
    # Crear agente DQN
    agent = DQNAgent(
        state_size=1,  # Un solo estado (one-hot encoded)
        action_size=1,  # Una sola acción
        model_type="mlp",
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        seed=42
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    losses = []
    episode_rewards_buffer = []
    
    # Entrenamiento
    for episode in tqdm(range(n_episodes), desc="ConstantRewardEnv", disable=not verbose):
        state, _ = env.reset()
        # Convertir estado a one-hot
        state_one_hot = state_to_one_hot(state, 1)
        total_reward = 0
        episode_losses = []
        done = False
        step = 0
        
        while not done:
            # Seleccionar acción
            action = agent.select_action(state_one_hot)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convertir siguiente estado a one-hot
            next_state_one_hot = state_to_one_hot(next_state, 1)
            
            # Almacenar transición
            agent.push_transition(state_one_hot, action, reward, next_state_one_hot, done)
            
            # Entrenar
            loss = agent.train_step(episode)
            if loss is not None:
                episode_losses.append(loss)
            
            state_one_hot = next_state_one_hot
            total_reward += reward
            step += 1
        
        # Guardar datos del episodio
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular promedio móvil
        if len(episode_rewards_buffer) >= 10:
            avg_rewards.append(np.mean(episode_rewards_buffer[-10:]))
        else:
            avg_rewards.append(np.mean(episode_rewards_buffer))
        
        # Guardar loss promedio del episodio
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0.0)
        
        # Decaer epsilon
        agent.decay_epsilon(episode)
        
        # Logging cada 100 episodios
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}/{n_episodes}, "
                  f"Recompensa: {total_reward:.3f}, "
                  f"Promedio (10): {avg_rewards[-1]:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Recompensa promedio final: {np.mean(rewards[-100:]):.3f}")
        print(f"Epsilon final: {agent.epsilon:.3f}")
    
    return agent, rewards, avg_rewards, losses


def train_random_obs_env(n_episodes: int = 1000, learning_rate: float = 1e-3,
                        discount_factor: float = 0.99, epsilon_start: float = 1.0,
                        epsilon_min: float = 0.01, epsilon_decay: float = 0.001,
                        batch_size: int = 32, buffer_capacity: int = 10000,
                        target_update_freq: int = 100, start_learning: int = 100,
                        verbose: bool = True) -> Tuple[DQNAgent, List[float], List[float], List[float]]:
    """
    Entrena DQN en RandomObsBinaryRewardEnv.
    
    Este entorno devuelve observaciones aleatorias (+1 o -1) y la recompensa
    es igual a la observación. El agente debería aprender a predecir el valor
    correcto basado en la observación.
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        learning_rate: Tasa de aprendizaje
        discount_factor: Factor de descuento
        epsilon_start: Epsilon inicial
        epsilon_min: Epsilon mínimo
        epsilon_decay: Tasa de decaimiento de epsilon
        batch_size: Tamaño del batch
        buffer_capacity: Capacidad del replay buffer
        target_update_freq: Frecuencia de actualización de target network
        start_learning: Episodios antes de empezar a entrenar
        verbose: Si mostrar progreso
        
    Returns:
        agent: Agente DQN entrenado
        rewards: Recompensas por episodio
        avg_rewards: Recompensas promedio
        losses: Pérdidas por episodio
    """
    if verbose:
        print("🎲 Entrenando DQN en RandomObsBinaryRewardEnv")
        print("=" * 50)
    
    # Crear entorno
    env = RandomObsBinaryRewardEnv(seed=42)
    
    # Crear agente DQN
    agent = DQNAgent(
        state_size=2,  # Dos estados (one-hot encoded: [1,0] o [0,1])
        action_size=1,  # Una sola acción
        model_type="mlp",
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        seed=42
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    losses = []
    episode_rewards_buffer = []
    
    # Entrenamiento
    for episode in tqdm(range(n_episodes), desc="RandomObsBinaryRewardEnv", disable=not verbose):
        state, _ = env.reset()
        # Convertir estado a one-hot
        state_one_hot = state_to_one_hot(state, 2)
        total_reward = 0
        episode_losses = []
        done = False
        step = 0
        
        while not done:
            # Seleccionar acción
            action = agent.select_action(state_one_hot)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convertir siguiente estado a one-hot
            next_state_one_hot = state_to_one_hot(next_state, 2)
            
            # Almacenar transición
            agent.push_transition(state_one_hot, action, reward, next_state_one_hot, done)
            
            # Entrenar
            loss = agent.train_step(episode)
            if loss is not None:
                episode_losses.append(loss)
            
            state_one_hot = next_state_one_hot
            total_reward += reward
            step += 1
        
        # Guardar datos del episodio
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular promedio móvil
        if len(episode_rewards_buffer) >= 10:
            avg_rewards.append(np.mean(episode_rewards_buffer[-10:]))
        else:
            avg_rewards.append(np.mean(episode_rewards_buffer))
        
        # Guardar loss promedio del episodio
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0.0)
        
        # Decaer epsilon
        agent.decay_epsilon(episode)
        
        # Logging cada 100 episodios
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}/{n_episodes}, "
                  f"Recompensa: {total_reward:.3f}, "
                  f"Promedio (10): {avg_rewards[-1]:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Recompensa promedio final: {np.mean(rewards[-100:]):.3f}")
        print(f"Epsilon final: {agent.epsilon:.3f}")
    
    return agent, rewards, avg_rewards, losses


def train_two_step_env(n_episodes: int = 1000, learning_rate: float = 1e-3,
                      discount_factor: float = 0.99, epsilon_start: float = 1.0,
                      epsilon_min: float = 0.01, epsilon_decay: float = 0.001,
                      batch_size: int = 32, buffer_capacity: int = 10000,
                      target_update_freq: int = 100, start_learning: int = 100,
                      verbose: bool = True) -> Tuple[DQNAgent, List[float], List[float], List[float]]:
    """
    Entrena DQN en TwoStepDelayedRewardEnv.
    
    Este entorno tiene dos pasos: el primero sin recompensa (0) y el segundo
    con recompensa (+1). El agente debería aprender el valor descontado correcto
    para el primer estado.
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        learning_rate: Tasa de aprendizaje
        discount_factor: Factor de descuento
        epsilon_start: Epsilon inicial
        epsilon_min: Epsilon mínimo
        epsilon_decay: Tasa de decaimiento de epsilon
        batch_size: Tamaño del batch
        buffer_capacity: Capacidad del replay buffer
        target_update_freq: Frecuencia de actualización de target network
        start_learning: Episodios antes de empezar a entrenar
        verbose: Si mostrar progreso
        
    Returns:
        agent: Agente DQN entrenado
        rewards: Recompensas por episodio
        avg_rewards: Recompensas promedio
        losses: Pérdidas por episodio
    """
    if verbose:
        print("⏱️ Entrenando DQN en TwoStepDelayedRewardEnv")
        print("=" * 50)
    
    # Crear entorno
    env = TwoStepDelayedRewardEnv()
    
    # Crear agente DQN
    agent = DQNAgent(
        state_size=2,  # Dos estados (one-hot encoded: [1,0] o [0,1])
        action_size=1,  # Una sola acción
        model_type="mlp",
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        seed=42
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    losses = []
    episode_rewards_buffer = []
    
    # Entrenamiento
    for episode in tqdm(range(n_episodes), desc="TwoStepDelayedRewardEnv", disable=not verbose):
        state, _ = env.reset()
        # Convertir estado a one-hot
        state_one_hot = state_to_one_hot(state, 2)
        total_reward = 0
        episode_losses = []
        done = False
        step = 0
        
        while not done:
            # Seleccionar acción
            action = agent.select_action(state_one_hot)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convertir siguiente estado a one-hot
            next_state_one_hot = state_to_one_hot(next_state, 2)
            
            # Almacenar transición
            agent.push_transition(state_one_hot, action, reward, next_state_one_hot, done)
            
            # Entrenar
            loss = agent.train_step(episode)
            if loss is not None:
                episode_losses.append(loss)
            
            state_one_hot = next_state_one_hot
            total_reward += reward
            step += 1
        
        # Guardar datos del episodio
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular promedio móvil
        if len(episode_rewards_buffer) >= 10:
            avg_rewards.append(np.mean(episode_rewards_buffer[-10:]))
        else:
            avg_rewards.append(np.mean(episode_rewards_buffer))
        
        # Guardar loss promedio del episodio
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0.0)
        
        # Decaer epsilon
        agent.decay_epsilon(episode)
        
        # Logging cada 100 episodios
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}/{n_episodes}, "
                  f"Recompensa: {total_reward:.3f}, "
                  f"Promedio (10): {avg_rewards[-1]:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Recompensa promedio final: {np.mean(rewards[-100:]):.3f}")
        print(f"Epsilon final: {agent.epsilon:.3f}")
    
    return agent, rewards, avg_rewards, losses


def plot_training_results(results: Dict[str, Any], save_path: str = "plots/dqn_custom_envs_training.png"):
    """
    Crea gráficos individuales para cada entorno: loss vs episodio, reward vs episodio, y tablas de Q-values.
    
    Args:
        results: Diccionario con resultados de los tres entornos
        save_path: Ruta base donde guardar los gráficos
    """
    # Crear directorio si no existe
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Colores para cada entorno
    colors = {'constant': 'blue', 'random_obs': 'red', 'two_step': 'green'}
    names = {'constant': 'ConstantRewardEnv', 'random_obs': 'RandomObsBinaryRewardEnv', 'two_step': 'TwoStepDelayedRewardEnv'}
    
    # Crear gráfico individual para cada entorno
    for env_name, data in results.items():
        # Crear figura con 2 subplots (loss y reward)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Entrenamiento DQN - {names[env_name]}', fontsize=16, fontweight='bold')
        
        episodes = np.arange(1, len(data['rewards']) + 1)
        
        # 1. Loss vs Episodio
        if data['losses'] and any(loss > 0 for loss in data['losses']):
            # Filtrar losses válidos
            valid_losses = [loss for loss in data['losses'] if loss > 0]
            valid_episodes = episodes[:len(valid_losses)]
            
            ax1.plot(valid_episodes, valid_losses, color=colors[env_name], linewidth=2)
            ax1.set_xlabel('Episodios')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Loss vs Episodio - {names[env_name]}')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
        else:
            ax1.text(0.5, 0.5, 'No hay losses registrados', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'Loss vs Episodio - {names[env_name]}')
        
        # 2. Reward vs Episodio
        # Recompensas individuales
        ax2.plot(episodes, data['rewards'], color=colors[env_name], alpha=0.3, linewidth=1, label='Recompensas individuales')
        
        # Promedio móvil
        if len(data['rewards']) > 10:
            window_size = min(50, len(data['rewards']) // 10)
            moving_avg = []
            for j in range(len(data['rewards'])):
                start_idx = max(0, j - window_size + 1)
                window_rewards = data['rewards'][start_idx:j+1]
                moving_avg.append(np.mean(window_rewards))
            
            ax2.plot(episodes, moving_avg, color=colors[env_name], linewidth=3, alpha=0.8, label='Promedio móvil')
        
        ax2.set_xlabel('Episodios')
        ax2.set_ylabel('Recompensa')
        ax2.set_title(f'Reward vs Episodio - {names[env_name]}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico individual
        env_save_path = save_path.replace('.png', f'_{env_name}.png')
        try:
            plt.savefig(env_save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📊 Gráfico guardado en: {env_save_path}")
        except Exception as e:
            print(f"⚠️ Error guardando gráfico {env_name}: {e}")
            print("Mostrando gráfico sin guardar...")
        
        plt.show()
    
    # Crear tablas de Q-values
    print_training_tables(results)


def print_training_tables(results: Dict[str, Any]):
    """
    Imprime tablas de Q-values para cada entorno.
    
    Args:
        results: Diccionario con resultados de los tres entornos
    """
    # Definir nombres de entornos
    names = {'constant': 'ConstantRewardEnv', 'random_obs': 'RandomObsBinaryRewardEnv', 'two_step': 'TwoStepDelayedRewardEnv'}
    
    print("\n" + "="*80)
    print("📋 TABLAS DE Q-VALUES (DQN)")
    print("="*80)
    
    for env_name, data in results.items():
        agent = data['agent']
        
        if env_name == 'constant':
            print(f"\n🔵 ConstantRewardEnv:")
            print(f"{'Estado':<10} {'Acción':<10} {'Q-value Final':<15} {'Q-value Esperado':<15} {'Error':<10}")
            print("-" * 70)
            
            # Evaluar Q-value para estado 0
            with torch.no_grad():
                state_one_hot = state_to_one_hot(0, 1)
                state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                q_value = agent.q_network(state_tensor).item()
            
            q_expected = 1.0
            error = abs(q_value - q_expected)
            print(f"{'0':<10} {'0':<10} {q_value:<15.3f} {q_expected:<15.3f} {error:<10.3f}")
            
        elif env_name == 'random_obs':
            print(f"\n🔴 RandomObsBinaryRewardEnv:")
            print(f"{'Estado':<10} {'Acción':<10} {'Q-value Final':<15} {'Q-value Esperado':<15} {'Error':<10}")
            print("-" * 70)
            
            for state in [0, 1]:
                with torch.no_grad():
                    state_one_hot = state_to_one_hot(state, 2)
                    state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                    q_value = agent.q_network(state_tensor).item()
                
                q_expected = -1.0 if state == 0 else 1.0
                error = abs(q_value - q_expected)
                print(f"{state:<10} {'0':<10} {q_value:<15.3f} {q_expected:<15.3f} {error:<10.3f}")
                
        elif env_name == 'two_step':
            print(f"\n🟢 TwoStepDelayedRewardEnv:")
            print(f"{'Estado':<10} {'Acción':<10} {'Q-value Final':<15} {'Q-value Esperado':<15} {'Error':<10}")
            print("-" * 70)
            
            # Calcular valores esperados
            temp_env = TwoStepDelayedRewardEnv()
            expected_q = temp_env.get_optimal_q_values(0.99)
            
            for state in [0, 1]:
                with torch.no_grad():
                    state_one_hot = state_to_one_hot(state, 2)
                    state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                    q_value = agent.q_network(state_tensor).item()
                
                q_expected = expected_q[state]
                error = abs(q_value - q_expected)
                print(f"{state:<10} {'0':<10} {q_value:<15.3f} {q_expected:<15.3f} {error:<10.3f}")
    
    print("\n" + "="*80)
    print("📊 RESUMEN DE CONVERGENCIA (DQN)")
    print("="*80)
    
    for env_name, data in results.items():
        agent = data['agent']
        total_error = 0.0
        num_states = 0
        
        if env_name == 'constant':
            with torch.no_grad():
                state_one_hot = state_to_one_hot(0, 1)
                state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                q_value = agent.q_network(state_tensor).item()
            q_expected = 1.0
            total_error = abs(q_value - q_expected)
            num_states = 1
            
        elif env_name == 'random_obs':
            for state in [0, 1]:
                with torch.no_grad():
                    state_one_hot = state_to_one_hot(state, 2)
                    state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                    q_value = agent.q_network(state_tensor).item()
                q_expected = -1.0 if state == 0 else 1.0
                total_error += abs(q_value - q_expected)
                num_states += 1
                
        elif env_name == 'two_step':
            temp_env = TwoStepDelayedRewardEnv()
            expected_q = temp_env.get_optimal_q_values(0.99)
            for state in [0, 1]:
                with torch.no_grad():
                    state_one_hot = state_to_one_hot(state, 2)
                    state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                    q_value = agent.q_network(state_tensor).item()
                q_expected = expected_q[state]
                total_error += abs(q_value - q_expected)
                num_states += 1
        
        avg_error = total_error / num_states
        convergence_status = "✅ CONVERGIDO" if avg_error < 0.1 else "❌ NO CONVERGIDO"
        
        print(f"{names[env_name]:<25} | Error Promedio: {avg_error:.3f} | {convergence_status}")
    
    print("="*80)


def main():
    """
    Función principal que entrena los tres entornos con DQN y genera gráficos comparativos.
    """
    print("🚀 Entrenamiento DQN en Entornos Personalizados")
    print("=" * 60)
    
    # Parámetros de entrenamiento
    n_episodes = 1000
    learning_rate = 1e-3
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.001
    batch_size = 32
    buffer_capacity = 10000
    target_update_freq = 100
    start_learning = 100
    
    # Almacenar resultados
    results = {}
    
    # 1. Entrenar en ConstantRewardEnv
    print("\n" + "="*60)
    agent_constant, rewards_constant, avg_rewards_constant, losses_constant = train_constant_env(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        verbose=True
    )
    results['constant'] = {
        'agent': agent_constant,
        'rewards': rewards_constant,
        'avg_rewards': avg_rewards_constant,
        'losses': losses_constant
    }
    
    # 2. Entrenar en RandomObsBinaryRewardEnv
    print("\n" + "="*60)
    agent_random, rewards_random, avg_rewards_random, losses_random = train_random_obs_env(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        verbose=True
    )
    results['random_obs'] = {
        'agent': agent_random,
        'rewards': rewards_random,
        'avg_rewards': avg_rewards_random,
        'losses': losses_random
    }
    
    # 3. Entrenar en TwoStepDelayedRewardEnv
    print("\n" + "="*60)
    agent_two_step, rewards_two_step, avg_rewards_two_step, losses_two_step = train_two_step_env(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        start_learning=start_learning,
        verbose=True
    )
    results['two_step'] = {
        'agent': agent_two_step,
        'rewards': rewards_two_step,
        'avg_rewards': avg_rewards_two_step,
        'losses': losses_two_step
    }
    
    # 4. Generar gráficos comparativos
    print("\n" + "="*60)
    print("📊 Generando gráficos comparativos...")
    plot_training_results(results)
    
    # 5. Resumen final
    print("\n" + "="*60)
    print("📋 RESUMEN DE RESULTADOS DQN")
    print("="*60)
    
    # Definir nombres de entornos
    names = {'constant': 'ConstantRewardEnv', 'random_obs': 'RandomObsBinaryRewardEnv', 'two_step': 'TwoStepDelayedRewardEnv'}
    
    for env_name, data in results.items():
        final_avg = np.mean(data['rewards'][-50:]) if len(data['rewards']) >= 50 else np.mean(data['rewards'])
        final_loss = np.mean([l for l in data['losses'] if l > 0]) if any(l > 0 for l in data['losses']) else 0.0
        
        print(f"\n{names[env_name]}:")
        print(f"  - Recompensa promedio final: {final_avg:.3f}")
        print(f"  - Loss promedio final: {final_loss:.3f}")
        print(f"  - Epsilon final: {data['agent'].epsilon:.3f}")
    
    print(f"\n🎉 Entrenamiento DQN completado en los tres entornos!")


if __name__ == "__main__":
    main()
