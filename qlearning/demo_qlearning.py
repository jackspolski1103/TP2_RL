#!/usr/bin/env python3
"""
Script de demostración del agente Q-Learning en FrozenLake-v1.

Este script muestra diferentes configuraciones de hiperparámetros
y permite comparar el rendimiento del agente.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qlearning_agent import train_qlearning, evaluate_qlearning, plot_training_curves


def compare_hyperparameters():
    """
    Compara diferentes configuraciones de hiperparámetros.
    """
    print("=== Comparación de Hiperparámetros ===\n")
    
    # Configuraciones a probar
    configs = [
        {"name": "Alpha Alto", "alpha": 0.9, "gamma": 0.99, "decay_rate": 0.001},
        {"name": "Alpha Medio", "alpha": 0.5, "gamma": 0.99, "decay_rate": 0.001},
        {"name": "Alpha Bajo", "alpha": 0.1, "gamma": 0.99, "decay_rate": 0.001},
        {"name": "Gamma Alto", "alpha": 0.8, "gamma": 0.99, "decay_rate": 0.001},
        {"name": "Gamma Bajo", "alpha": 0.8, "gamma": 0.9, "decay_rate": 0.001},
        {"name": "Decay Rápido", "alpha": 0.8, "gamma": 0.99, "decay_rate": 0.005},
        {"name": "Decay Lento", "alpha": 0.8, "gamma": 0.99, "decay_rate": 0.0005},
    ]
    
    results = {}
    
    for config in configs:
        print(f"Entrenando configuración: {config['name']}")
        print(f"  Alpha: {config['alpha']}, Gamma: {config['gamma']}, Decay: {config['decay_rate']}")
        
        # Entrenar con configuración específica
        q_table, rewards, avg_rewards = train_qlearning(
            n_episodes=3000,  # Menos episodios para comparación rápida
            alpha=config['alpha'],
            gamma=config['gamma'],
            decay_rate=config['decay_rate'],
            verbose=False
        )
        
        # Evaluar
        success_rate = evaluate_qlearning(q_table, verbose=False)
        
        # Guardar resultados
        results[config['name']] = {
            'avg_rewards': avg_rewards,
            'success_rate': success_rate,
            'final_performance': avg_rewards[-1] if avg_rewards else 0
        }
        
        print(f"  Tasa de éxito: {success_rate*100:.1f}%")
        print(f"  Rendimiento final: {results[config['name']]['final_performance']:.3f}\n")
    
    # Crear gráfico comparativo
    plot_comparison(results)
    
    return results


def plot_comparison(results, save_path="plots/qlearning_comparison.png"):
    """
    Crea un gráfico comparativo de diferentes configuraciones.
    
    Args:
        results: Diccionario con resultados de diferentes configuraciones
        save_path: Ruta donde guardar el gráfico
    """
    # Crear directorio si no existe
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Curvas de entrenamiento
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (name, data) in enumerate(results.items()):
        avg_rewards = data['avg_rewards']
        episodes = np.arange(100, len(avg_rewards) * 100 + 1, 100)
        ax1.plot(episodes, avg_rewards, color=colors[i], linewidth=2, 
                label=f"{name} (final: {data['final_performance']:.3f})")
    
    ax1.set_xlabel('Episodios')
    ax1.set_ylabel('Recompensa Promedio')
    ax1.set_title('Comparación de Curvas de Entrenamiento')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 2: Tasa de éxito final
    names = list(results.keys())
    success_rates = [results[name]['success_rate'] * 100 for name in names]
    
    bars = ax2.bar(range(len(names)), success_rates, color=colors[:len(names)], alpha=0.7)
    ax2.set_xlabel('Configuración')
    ax2.set_ylabel('Tasa de Éxito (%)')
    ax2.set_title('Tasa de Éxito Final por Configuración')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo guardado en: {save_path}")
    
    # Mostrar figura
    plt.show()


def demo_trained_agent():
    """
    Demuestra un agente entrenado jugando algunos episodios.
    """
    print("=== Demostración de Agente Entrenado ===\n")
    
    # Entrenar agente rápidamente
    print("Entrenando agente...")
    q_table, _, _ = train_qlearning(n_episodes=2000, verbose=False)
    
    # Evaluar con diferentes niveles de epsilon
    epsilons = [0.0, 0.1, 0.3]
    
    for epsilon in epsilons:
        print(f"\nEvaluando con epsilon = {epsilon} ({'greedy' if epsilon == 0 else 'epsilon-greedy'})")
        success_rate = evaluate_qlearning(q_table, epsilon=epsilon, n_eval_episodes=50, verbose=False)
        print(f"Tasa de éxito: {success_rate*100:.1f}%")


def analyze_q_table():
    """
    Analiza la tabla Q aprendida.
    """
    print("=== Análisis de Tabla Q ===\n")
    
    # Entrenar agente
    print("Entrenando agente para análisis...")
    q_table, _, _ = train_qlearning(n_episodes=3000, verbose=False)
    
    # Analizar algunos estados importantes
    action_names = ['Izquierda', 'Abajo', 'Derecha', 'Arriba']
    
    print("Valores Q para estados clave:")
    print("(Estado 0 = inicio, Estado 15 = meta)")
    
    # Estados de interés
    states_of_interest = [0, 1, 4, 5, 6, 9, 10, 13, 14, 15]
    
    for state in states_of_interest:
        if state in q_table:
            q_values = q_table[state]
            best_action = np.argmax(q_values)
            print(f"\nEstado {state:2d}: {q_values}")
            print(f"         Mejor acción: {action_names[best_action]} (valor: {q_values[best_action]:.3f})")
        else:
            print(f"\nEstado {state:2d}: No visitado durante el entrenamiento")


def main():
    """Función principal del demo."""
    print("🎮 Demo del Agente Q-Learning en FrozenLake-v1\n")
    
    # Menú de opciones
    while True:
        print("\n" + "="*50)
        print("Selecciona una opción:")
        print("1. Entrenamiento básico")
        print("2. Comparar hiperparámetros")
        print("3. Demostrar agente entrenado")
        print("4. Analizar tabla Q")
        print("5. Salir")
        print("="*50)
        
        choice = input("Opción (1-5): ").strip()
        
        if choice == '1':
            print("\n=== Entrenamiento Básico ===")
            q_table, rewards, avg_rewards = train_qlearning()
            success_rate = evaluate_qlearning(q_table)
            print(f"Success rate: {success_rate*100:.2f}%")
            plot_training_curves(avg_rewards)
            
        elif choice == '2':
            compare_hyperparameters()
            
        elif choice == '3':
            demo_trained_agent()
            
        elif choice == '4':
            analyze_q_table()
            
        elif choice == '5':
            print("¡Hasta luego! 👋")
            break
            
        else:
            print("Opción no válida. Por favor selecciona 1-5.")


if __name__ == "__main__":
    main()
