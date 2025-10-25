#!/usr/bin/env python3
"""
Script para generar gráficos de Loss vs Episodios para Q-Learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qlearning_agent import train_qlearning
from train_custom_envs import train_constant_env, train_random_obs_env, train_two_step_env


def create_loss_vs_episodes_plot():
    """
    Crea gráficos de Loss vs Episodios para diferentes configuraciones de Q-Learning.
    """
    print("🎮 Generando gráficos de Loss vs Episodios")
    print("="*60)
    
    # Configuraciones de hiperparámetros
    configs = [
        {"alpha": 0.8, "gamma": 0.99, "decay_rate": 0.001, "color": "blue", "label": "α=0.8, γ=0.99"},
        {"alpha": 0.9, "gamma": 0.99, "decay_rate": 0.001, "color": "red", "label": "α=0.9, γ=0.99"},
        {"alpha": 0.3, "gamma": 0.99, "decay_rate": 0.001, "color": "green", "label": "α=0.3, γ=0.99"},
        {"alpha": 0.8, "gamma": 0.9, "decay_rate": 0.001, "color": "orange", "label": "α=0.8, γ=0.9"},
        {"alpha": 0.8, "gamma": 0.99, "decay_rate": 0.005, "color": "purple", "label": "α=0.8, γ=0.99, decay=0.005"},
    ]
    
    n_episodes = 2000
    eval_interval = 100
    
    print(f"Entrenando {len(configs)} configuraciones...")
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Entrenar cada configuración y graficar
    for i, config in enumerate(configs):
        print(f"📊 Configuración {i+1}/{len(configs)}: {config['label']}")
        
        # Entrenar agente
        q_table, rewards, avg_rewards, losses = train_qlearning(
            n_episodes=n_episodes,
            alpha=config['alpha'],
            gamma=config['gamma'],
            decay_rate=config['decay_rate'],
            eval_interval=eval_interval,
            verbose=False
        )
        
        # Calcular episodios correspondientes
        episodes = np.arange(eval_interval, len(losses) * eval_interval + 1, eval_interval)
        
        # Graficar curva de pérdidas
        plt.plot(episodes, losses, 
                color=config['color'], linewidth=2, 
                label=config['label'], marker='o', markersize=4)
    
    # Configurar gráfico
    plt.xlabel('Episodios', fontsize=12)
    plt.ylabel('Pérdida Promedio (TD Error)', fontsize=12)
    plt.title('Loss vs Episodios - Q-Learning en FrozenLake-v1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    
    # Ajustar layout
    plt.tight_layout()
    
    # Crear directorio si no existe
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Guardar figura
    save_path = plots_dir / "qlearning_loss_vs_episodes.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {save_path}")
    
    # Mostrar figura
    plt.show()
    
    print("✅ Gráfico de Loss vs Episodios generado exitosamente!")


def create_custom_envs_loss_plot():
    """
    Crea gráficos de Loss vs Episodios para los entornos personalizados.
    """
    print("\n🎮 Generando gráficos de Loss vs Episodios para entornos personalizados")
    print("="*70)
    
    # Parámetros de entrenamiento
    n_episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.001
    
    # Colores y nombres
    colors = {'constant': 'blue', 'random_obs': 'red', 'two_step': 'green'}
    names = {'constant': 'ConstantRewardEnv', 'random_obs': 'RandomObsBinaryRewardEnv', 'two_step': 'TwoStepDelayedRewardEnv'}
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Loss vs Episodios - Entornos Personalizados', fontsize=16, fontweight='bold')
    
    # Entrenar cada entorno
    envs = [
        ('constant', train_constant_env),
        ('random_obs', train_random_obs_env),
        ('two_step', train_two_step_env)
    ]
    
    for i, (env_name, train_func) in enumerate(envs):
        print(f"📊 Entrenando {names[env_name]}...")
        
        # Entrenar agente
        q_table, rewards, avg_rewards, losses = train_func(
            n_episodes=n_episodes,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            verbose=False
        )
        
        # Calcular episodios correspondientes (cada 50 episodios)
        episodes = np.arange(50, len(losses) * 50 + 1, 50)
        
        # Graficar en subplot correspondiente
        axes[i].plot(episodes, losses, 
                    color=colors[env_name], linewidth=2, 
                    marker='o', markersize=4)
        axes[i].set_xlabel('Episodios', fontsize=12)
        axes[i].set_ylabel('Pérdida Promedio (TD Error)', fontsize=12)
        axes[i].set_title(f'{names[env_name]}', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')
        
        # Agregar información adicional
        final_loss = losses[-1] if losses else 0
        axes[i].text(0.02, 0.98, f'Pérdida Final: {final_loss:.4f}', 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Ajustar layout
    plt.tight_layout()
    
    # Crear directorio si no existe
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Guardar figura
    save_path = plots_dir / "custom_envs_loss_vs_episodes.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {save_path}")
    
    # Mostrar figura
    plt.show()
    
    print("✅ Gráficos de Loss vs Episodios para entornos personalizados generados exitosamente!")


def main():
    """Función principal."""
    print("🚀 GENERADOR DE GRÁFICOS LOSS VS EPISODIOS")
    print("=" * 60)
    
    try:
        # Crear gráfico para FrozenLake-v1 con diferentes hiperparámetros
        create_loss_vs_episodes_plot()
        
        # Crear gráficos para entornos personalizados
        create_custom_envs_loss_plot()
        
        print("\n🎉 ¡Todos los gráficos generados exitosamente!")
        print("✅ Ahora puedes ver las curvas de pérdida vs episodios.")
        
    except Exception as e:
        print(f"\n❌ Error generando gráficos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
