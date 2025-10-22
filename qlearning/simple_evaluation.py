#!/usr/bin/env python3
"""
Script simplificado para evaluar el desempeño del mejor modelo Q-Learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qlearning_agent import train_qlearning, evaluate_qlearning


def simple_evaluation():
    """
    Evalúa el mejor modelo con ambas políticas.
    """
    print("🎮 Evaluación del Mejor Modelo Q-Learning")
    print("="*50)
    
    # Mejor configuración: Decay Rápido
    print("📊 Configuración: Decay Rápido (Mejor)")
    print("   Alpha: 0.8, Gamma: 0.99, Decay: 0.005")
    print()
    
    # Entrenar el mejor modelo con más episodios para mejor convergencia
    print("🚀 Entrenando el mejor modelo con más episodios...")
    q_table, rewards, avg_rewards = train_qlearning(
        n_episodes=15000,  # Más episodios para ambiente determinístico
        alpha=0.1,  # Alpha más bajo para ambiente determinístico
        gamma=0.99,
        decay_rate=0.001,  # Decay más lento
        eval_interval=500,  # Menos salida
        verbose=True
    )
    
    print("\n" + "="*50)
    print("📈 EVALUACIÓN DEL DESEMPEÑO")
    print("="*50)
    
    # a. Evaluación con política greedy (ε = 0.0)
    print("\n🔍 a) Evaluación con política greedy (ε = 0.0):")
    success_rate_greedy = evaluate_qlearning(
        q_table, 
        n_eval_episodes=100, 
        epsilon=0.0, 
        verbose=True
    )
    
    # b. Evaluación con exploración fija (ε = 0.1)
    print("\n🔍 b) Evaluación con exploración fija (ε = 0.1):")
    success_rate_exploration = evaluate_qlearning(
        q_table, 
        n_eval_episodes=100, 
        epsilon=0.1, 
        verbose=True
    )
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES")
    print("="*50)
    
    print(f"✅ Política Greedy (ε = 0.0):     {success_rate_greedy*100:.1f}%")
    print(f"🔍 Con Exploración (ε = 0.1):     {success_rate_exploration*100:.1f}%")
    
    # Crear gráfico simple
    create_simple_plot(success_rate_greedy, success_rate_exploration)
    
    return success_rate_greedy, success_rate_exploration


def create_simple_plot(greedy_rate, exploration_rate):
    """
    Crea un gráfico simple de comparación.
    """
    # Crear directorio si no existe
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    
    # Datos
    policies = ['Política Greedy\n(ε = 0.0)', 'Con Exploración\n(ε = 0.1)']
    success_rates = [greedy_rate * 100, exploration_rate * 100]
    colors = ['#2E8B57', '#FF6B6B']
    
    # Crear gráfico de barras
    bars = plt.bar(policies, success_rates, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    
    # Configurar gráfico
    plt.ylabel('Tasa de Éxito (%)', fontsize=12)
    plt.title('Evaluación del Mejor Modelo Q-Learning\nFrozenLake-v1', 
              fontsize=14, fontweight='bold')
    plt.ylim(0, max(success_rates) * 1.2 if max(success_rates) > 0 else 10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    save_path = plots_dir / "qlearning_evaluation_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {save_path}")
    
    # Mostrar figura
    plt.show()


def main():
    """Función principal."""
    try:
        greedy_rate, exploration_rate = simple_evaluation()
        
        print("\n🎉 ¡Evaluación completada!")
        print(f"📊 Resultados finales:")
        print(f"   - Política Greedy: {greedy_rate*100:.1f}%")
        print(f"   - Con Exploración: {exploration_rate*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        raise


if __name__ == "__main__":
    main()
