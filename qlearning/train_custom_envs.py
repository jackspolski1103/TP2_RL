"""
Entrenamiento de Q-Learning en entornos personalizados.

Este script entrena agentes Q-Learning en los tres entornos personalizados:
1. ConstantRewardEnv: Recompensa constante
2. RandomObsBinaryRewardEnv: Observaciones aleatorias con recompensa binaria
3. TwoStepDelayedRewardEnv: Recompensa retrasada en dos pasos

Cada entorno tiene características específicas que permiten probar diferentes
aspectos del algoritmo Q-Learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Any

# Importar entornos personalizados
from envs import ConstantRewardEnv, RandomObsBinaryRewardEnv, TwoStepDelayedRewardEnv
from qlearning_agent import QLearningAgent


def train_constant_env(n_episodes: int = 1000, learning_rate: float = 0.1, 
                      discount_factor: float = 0.99, epsilon_start: float = 1.0,
                      epsilon_min: float = 0.01, epsilon_decay: float = 0.001,
                      verbose: bool = True) -> Tuple[Dict, List[float], List[float], List[float]]:
    """
    Entrena Q-Learning en ConstantRewardEnv.
    
    Este entorno siempre devuelve recompensa +1, por lo que el agente debería
    aprender que el valor Q es constante e igual a 1.
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        learning_rate: Tasa de aprendizaje
        discount_factor: Factor de descuento
        epsilon_start: Epsilon inicial
        epsilon_min: Epsilon mínimo
        epsilon_decay: Tasa de decaimiento de epsilon
        verbose: Si mostrar progreso
        
    Returns:
        q_table: Tabla Q entrenada
        rewards: Recompensas por episodio
        avg_rewards: Recompensas promedio
        losses: Pérdidas promedio
    """
    if verbose:
        print("🎯 Entrenando en ConstantRewardEnv")
        print("=" * 50)
    
    # Crear entorno
    env = ConstantRewardEnv()
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=1,  # Solo una acción
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    episode_rewards_buffer = []
    losses = []
    episode_losses_buffer = []
    
    # Entrenamiento
    for episode in tqdm(range(n_episodes), desc="ConstantRewardEnv", disable=not verbose):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar Q-table y obtener error temporal
            td_error = agent.update(state, action, reward, next_state, done)
            total_loss += td_error
            step_count += 1
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular pérdida promedio por paso
        avg_episode_loss = total_loss / max(step_count, 1)
        episode_losses_buffer.append(avg_episode_loss)
        
        # Decaer epsilon
        agent.decay_epsilon(episode, epsilon_decay)
        
        # Calcular promedio cada 50 episodios
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards_buffer)
            avg_loss = np.mean(episode_losses_buffer)
            avg_rewards.append(avg_reward)
            losses.append(avg_loss)
            episode_rewards_buffer = []
            episode_losses_buffer = []
            
            if verbose and (episode + 1) % 200 == 0:
                print(f"Episodio {episode + 1}/{n_episodes}, "
                      f"Recompensa promedio: {avg_reward:.3f}, "
                      f"Pérdida promedio: {avg_loss:.3f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Convertir tabla Q a dict
    q_table = dict(agent.q_table)
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Valor Q aprendido: {q_table[0][0]:.3f}")
        print(f"Valor esperado: 1.0")
        print(f"Error: {abs(q_table[0][0] - 1.0):.3f}")
        print(f"Pérdida promedio final: {losses[-1]:.3f}")
    
    return q_table, rewards, avg_rewards, losses


def train_random_obs_env(n_episodes: int = 1000, learning_rate: float = 0.1,
                        discount_factor: float = 0.99, epsilon_start: float = 1.0,
                        epsilon_min: float = 0.01, epsilon_decay: float = 0.001,
                        verbose: bool = True) -> Tuple[Dict, List[float], List[float], List[float]]:
    """
    Entrena Q-Learning en RandomObsBinaryRewardEnv.
    
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
        verbose: Si mostrar progreso
        
    Returns:
        q_table: Tabla Q entrenada
        rewards: Recompensas por episodio
        avg_rewards: Recompensas promedio
        losses: Pérdidas promedio
    """
    if verbose:
        print("🎲 Entrenando en RandomObsBinaryRewardEnv")
        print("=" * 50)
    
    # Crear entorno
    env = RandomObsBinaryRewardEnv(seed=42)
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=1,  # Solo una acción
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    episode_rewards_buffer = []
    losses = []
    episode_losses_buffer = []
    
    # Entrenamiento
    for episode in tqdm(range(n_episodes), desc="RandomObsBinaryRewardEnv", disable=not verbose):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar Q-table y obtener error temporal
            td_error = agent.update(state, action, reward, next_state, done)
            total_loss += td_error
            step_count += 1
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular pérdida promedio por paso
        avg_episode_loss = total_loss / max(step_count, 1)
        episode_losses_buffer.append(avg_episode_loss)
        
        # Decaer epsilon
        agent.decay_epsilon(episode, epsilon_decay)
        
        # Calcular promedio cada 50 episodios
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards_buffer)
            avg_loss = np.mean(episode_losses_buffer)
            avg_rewards.append(avg_reward)
            losses.append(avg_loss)
            episode_rewards_buffer = []
            episode_losses_buffer = []
            
            if verbose and (episode + 1) % 200 == 0:
                print(f"Episodio {episode + 1}/{n_episodes}, "
                      f"Recompensa promedio: {avg_reward:.3f}, "
                      f"Pérdida promedio: {avg_loss:.3f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Convertir tabla Q a dict
    q_table = dict(agent.q_table)
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Valor Q para estado 0: {q_table[0][0]:.3f}")
        print(f"Valor Q para estado 1: {q_table[1][0]:.3f}")
        print(f"Valores esperados: -1.0 y +1.0 respectivamente")
        print(f"Pérdida promedio final: {losses[-1]:.3f}")
    
    return q_table, rewards, avg_rewards, losses


def train_two_step_env(n_episodes: int = 1000, learning_rate: float = 0.1,
                       discount_factor: float = 0.99, epsilon_start: float = 1.0,
                       epsilon_min: float = 0.01, epsilon_decay: float = 0.001,
                       verbose: bool = True) -> Tuple[Dict, List[float], List[float], List[float]]:
    """
    Entrena Q-Learning en TwoStepDelayedRewardEnv.
    
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
        verbose: Si mostrar progreso
        
    Returns:
        q_table: Tabla Q entrenada
        rewards: Recompensas por episodio
        avg_rewards: Recompensas promedio
        losses: Pérdidas promedio
    """
    if verbose:
        print("⏱️ Entrenando en TwoStepDelayedRewardEnv")
        print("=" * 50)
    
    # Crear entorno
    env = TwoStepDelayedRewardEnv()
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=1,  # Solo una acción
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min
    )
    
    # Listas para tracking
    rewards = []
    avg_rewards = []
    episode_rewards_buffer = []
    losses = []
    episode_losses_buffer = []
    
    # Entrenamiento
    for episode in tqdm(range(n_episodes), desc="TwoStepDelayedRewardEnv", disable=not verbose):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar Q-table y obtener error temporal
            td_error = agent.update(state, action, reward, next_state, done)
            total_loss += td_error
            step_count += 1
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        episode_rewards_buffer.append(total_reward)
        
        # Calcular pérdida promedio por paso
        avg_episode_loss = total_loss / max(step_count, 1)
        episode_losses_buffer.append(avg_episode_loss)
        
        # Decaer epsilon
        agent.decay_epsilon(episode, epsilon_decay)
        
        # Calcular promedio cada 50 episodios
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards_buffer)
            avg_loss = np.mean(episode_losses_buffer)
            avg_rewards.append(avg_reward)
            losses.append(avg_loss)
            episode_rewards_buffer = []
            episode_losses_buffer = []
            
            if verbose and (episode + 1) % 200 == 0:
                print(f"Episodio {episode + 1}/{n_episodes}, "
                      f"Recompensa promedio: {avg_reward:.3f}, "
                      f"Pérdida promedio: {avg_loss:.3f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Convertir tabla Q a dict
    q_table = dict(agent.q_table)
    
    # Calcular valores óptimos esperados
    expected_q_values = env.get_optimal_q_values(discount_factor)
    
    if verbose:
        print(f"Entrenamiento completado!")
        print(f"Valor Q para estado 0: {q_table[0][0]:.3f} (esperado: {expected_q_values[0]:.3f})")
        print(f"Valor Q para estado 1: {q_table[1][0]:.3f} (esperado: {expected_q_values[1]:.3f})")
        print(f"Error estado 0: {abs(q_table[0][0] - expected_q_values[0]):.3f}")
        print(f"Error estado 1: {abs(q_table[1][0] - expected_q_values[1]):.3f}")
        print(f"Pérdida promedio final: {losses[-1]:.3f}")
    
    return q_table, rewards, avg_rewards, losses


def plot_training_results(results: Dict[str, Any], save_path: str = "plots/custom_envs_training.png"):
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
        fig.suptitle(f'Entrenamiento Q-Learning - {names[env_name]}', fontsize=16, fontweight='bold')
        
        episodes = np.arange(1, len(data['rewards']) + 1)
        
        # 1. Loss vs Episodio
        if 'losses' in data and len(data['losses']) > 0:
            # Usar las pérdidas reales trackeadas
            loss_episodes = np.arange(50, len(data['losses']) * 50 + 1, 50)
            ax1.plot(loss_episodes, data['losses'], color=colors[env_name], linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Episodios')
            ax1.set_ylabel('Pérdida Promedio (TD Error)')
            ax1.set_title(f'Loss vs Episodio - {names[env_name]}')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
        else:
            # Fallback: usar desviación estándar como proxy
            if len(data['rewards']) > 10:
                window_size = min(50, len(data['rewards']) // 10)
                loss_proxy = []
                for j in range(len(data['rewards'])):
                    start_idx = max(0, j - window_size + 1)
                    window_rewards = data['rewards'][start_idx:j+1]
                    loss_proxy.append(np.std(window_rewards))
                
                ax1.plot(episodes, loss_proxy, color=colors[env_name], linewidth=2)
                ax1.set_xlabel('Episodios')
                ax1.set_ylabel('Loss (Desv. Est.)')
                ax1.set_title(f'Loss vs Episodio - {names[env_name]}')
                ax1.grid(True, alpha=0.3)
                ax1.set_yscale('log')
        
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
    print("📋 TABLAS DE Q-VALUES")
    print("="*80)
    
    for env_name, data in results.items():
        q_table = data['q_table']
        
        if env_name == 'constant':
            print(f"\n🔵 ConstantRewardEnv:")
            print(f"{'Estado':<10} {'Acción':<10} {'Q-value Final':<15} {'Q-value Esperado':<15} {'Error':<10}")
            print("-" * 70)
            q_final = q_table[0][0]
            q_expected = 1.0
            error = abs(q_final - q_expected)
            print(f"{'0':<10} {'0':<10} {q_final:<15.3f} {q_expected:<15.3f} {error:<10.3f}")
            
        elif env_name == 'random_obs':
            print(f"\n🔴 RandomObsBinaryRewardEnv:")
            print(f"{'Estado':<10} {'Acción':<10} {'Q-value Final':<15} {'Q-value Esperado':<15} {'Error':<10}")
            print("-" * 70)
            for state in [0, 1]:
                q_final = q_table[state][0]
                q_expected = -1.0 if state == 0 else 1.0
                error = abs(q_final - q_expected)
                print(f"{state:<10} {'0':<10} {q_final:<15.3f} {q_expected:<15.3f} {error:<10.3f}")
                
        elif env_name == 'two_step':
            print(f"\n🟢 TwoStepDelayedRewardEnv:")
            print(f"{'Estado':<10} {'Acción':<10} {'Q-value Final':<15} {'Q-value Esperado':<15} {'Error':<10}")
            print("-" * 70)
            # Calcular valores esperados
            temp_env = TwoStepDelayedRewardEnv()
            expected_q = temp_env.get_optimal_q_values(0.99)
            for state in [0, 1]:
                q_final = q_table[state][0]
                q_expected = expected_q[state]
                error = abs(q_final - q_expected)
                print(f"{state:<10} {'0':<10} {q_final:<15.3f} {q_expected:<15.3f} {error:<10.3f}")
    
    print("\n" + "="*80)
    print("📊 RESUMEN DE CONVERGENCIA")
    print("="*80)
    
    for env_name, data in results.items():
        q_table = data['q_table']
        total_error = 0.0
        num_states = 0
        
        if env_name == 'constant':
            q_final = q_table[0][0]
            q_expected = 1.0
            total_error = abs(q_final - q_expected)
            num_states = 1
            
        elif env_name == 'random_obs':
            for state in [0, 1]:
                q_final = q_table[state][0]
                q_expected = -1.0 if state == 0 else 1.0
                total_error += abs(q_final - q_expected)
                num_states += 1
                
        elif env_name == 'two_step':
            temp_env = TwoStepDelayedRewardEnv()
            expected_q = temp_env.get_optimal_q_values(0.99)
            for state in [0, 1]:
                q_final = q_table[state][0]
                q_expected = expected_q[state]
                total_error += abs(q_final - q_expected)
                num_states += 1
        
        avg_error = total_error / num_states
        convergence_status = "✅ CONVERGIDO" if avg_error < 0.1 else "❌ NO CONVERGIDO"
        
        print(f"{names[env_name]:<25} | Error Promedio: {avg_error:.3f} | {convergence_status}")
    
    print("="*80)


def main():
    """
    Función principal que entrena los tres entornos y genera gráficos comparativos.
    """
    print("🚀 Entrenamiento Q-Learning en Entornos Personalizados")
    print("=" * 60)
    
    # Parámetros de entrenamiento
    n_episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.001
    
    # Almacenar resultados
    results = {}
    
    # 1. Entrenar en ConstantRewardEnv
    print("\n" + "="*60)
    q_table_constant, rewards_constant, avg_rewards_constant, losses_constant = train_constant_env(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        verbose=True
    )
    results['constant'] = {
        'q_table': q_table_constant,
        'rewards': rewards_constant,
        'avg_rewards': avg_rewards_constant,
        'losses': losses_constant
    }
    
    # 2. Entrenar en RandomObsBinaryRewardEnv
    print("\n" + "="*60)
    q_table_random, rewards_random, avg_rewards_random, losses_random = train_random_obs_env(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        verbose=True
    )
    results['random_obs'] = {
        'q_table': q_table_random,
        'rewards': rewards_random,
        'avg_rewards': avg_rewards_random,
        'losses': losses_random
    }
    
    # 3. Entrenar en TwoStepDelayedRewardEnv
    print("\n" + "="*60)
    q_table_two_step, rewards_two_step, avg_rewards_two_step, losses_two_step = train_two_step_env(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        verbose=True
    )
    results['two_step'] = {
        'q_table': q_table_two_step,
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
    print("📋 RESUMEN DE RESULTADOS")
    print("="*60)
    
    # Definir nombres de entornos para el resumen
    env_names = {'constant': 'ConstantRewardEnv', 'random_obs': 'RandomObsBinaryRewardEnv', 'two_step': 'TwoStepDelayedRewardEnv'}
    
    for env_name, data in results.items():
        q_table = data['q_table']
        final_avg = np.mean(data['rewards'][-50:]) if len(data['rewards']) >= 50 else np.mean(data['rewards'])
        
        print(f"\n{env_names[env_name]}:")
        print(f"  - Recompensa promedio final: {final_avg:.3f}")
        
        if env_name == 'constant':
            print(f"  - Valor Q aprendido: {q_table[0][0]:.3f} (esperado: 1.0)")
            print(f"  - Error: {abs(q_table[0][0] - 1.0):.3f}")
        elif env_name == 'random_obs':
            print(f"  - Valor Q estado 0: {q_table[0][0]:.3f} (esperado: -1.0)")
            print(f"  - Valor Q estado 1: {q_table[1][0]:.3f} (esperado: +1.0)")
            print(f"  - Error promedio: {abs(q_table[0][0] + 1.0) + abs(q_table[1][0] - 1.0):.3f}")
        elif env_name == 'two_step':
            # Crear instancia temporal del entorno para obtener valores óptimos
            temp_env = TwoStepDelayedRewardEnv()
            expected_q = temp_env.get_optimal_q_values(discount_factor)
            print(f"  - Valor Q estado 0: {q_table[0][0]:.3f} (esperado: {expected_q[0]:.3f})")
            print(f"  - Valor Q estado 1: {q_table[1][0]:.3f} (esperado: {expected_q[1]:.3f})")
            print(f"  - Error promedio: {abs(q_table[0][0] - expected_q[0]) + abs(q_table[1][0] - expected_q[1]):.3f}")
    
    print(f"\n🎉 Entrenamiento completado en los tres entornos!")


if __name__ == "__main__":
    main()
