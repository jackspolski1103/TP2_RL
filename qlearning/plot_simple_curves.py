#!/usr/bin/env python3
"""
Script para generar un gráfico simple de curvas de Q-Learning con hiperparámetros en los labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qlearning_agent import train_qlearning


def create_simple_curves_plot():
    """
    Crea un gráfico simple de curvas de entrenamiento con hiperparámetros en los labels.
    """
    print("🎮 Generando gráfico de curvas Q-Learning con hiperparámetros")
    print("="*60)
    
    # Configuraciones de hiperparámetros
    configs = [
        {"alpha": 0.8, "gamma": 0.99, "decay_rate": 0.001, "color": "blue"},
        {"alpha": 0.9, "gamma": 0.99, "decay_rate": 0.001, "color": "red"},
        {"alpha": 0.3, "gamma": 0.99, "decay_rate": 0.001, "color": "green"},
        {"alpha": 0.8, "gamma": 0.9, "decay_rate": 0.001, "color": "orange"},
        {"alpha": 0.8, "gamma": 0.99, "decay_rate": 0.005, "color": "purple"},
        {"alpha": 0.8, "gamma": 0.99, "decay_rate": 0.0005, "color": "brown"},
        {"alpha": 0.5, "gamma": 0.95, "decay_rate": 0.002, "color": "pink"},
    ]
    
    n_episodes = 3000
    eval_interval = 100
    
    print(f"Entrenando {len(configs)} configuraciones...")
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Entrenar cada configuración y graficar
    for i, config in enumerate(configs):
        print(f"📊 Configuración {i+1}/{len(configs)}: α={config['alpha']}, γ={config['gamma']}, decay={config['decay_rate']}")
        
        # Entrenar agente
        q_table, rewards, avg_rewards = train_qlearning(
            n_episodes=n_episodes,
            alpha=config['alpha'],
            gamma=config['gamma'],
            decay_rate=config['decay_rate'],
            eval_interval=eval_interval,
            verbose=False
        )
        
        # Calcular episodios correspondientes
        episodes = np.arange(eval_interval, len(avg_rewards) * eval_interval + 1, eval_interval)
        
        # Crear label con hiperparámetros
        label = f"α={config['alpha']}, γ={config['gamma']}, decay={config['decay_rate']}"
        
        # Graficar curva
        plt.plot(episodes, avg_rewards, 
                color=config['color'], linewidth=2, 
                label=label)
    
    # Configurar gráfico
    plt.xlabel('Episodios', fontsize=12)
    plt.ylabel('Recompensa Promedio', fontsize=12)
    plt.title('Curvas de Entrenamiento Q-Learning - FrozenLake-v1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Crear directorio si no existe
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Guardar figura
    save_path = plots_dir / "qlearning_curves_with_hyperparams.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {save_path}")
    
    # Mostrar figura
    plt.show()
    
    print("✅ Gráfico generado exitosamente!")


if __name__ == "__main__":
    create_simple_curves_plot()
