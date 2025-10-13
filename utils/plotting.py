"""
Utilidades para visualización y plotting.

Este módulo contiene funciones para crear gráficos de entrenamiento,
visualizar valores Q, y guardar resultados de experimentos.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd


def plot_training_progress(scores, losses=None, epsilon_history=None, title="Training Progress", 
                          save_path=None, window_size=100):
    """
    Plotea el progreso de entrenamiento del agente.
    
    Args:
        scores: Lista de puntuaciones por episodio
        losses: Lista de pérdidas durante entrenamiento (opcional)
        epsilon_history: Historia de valores de epsilon (opcional)
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        window_size: Tamaño de ventana para promedio móvil
    """
    # TODO: Configurar estilo de gráficos
    plt.style.use('seaborn-v0_8')
    
    # Determinar número de subplots
    n_plots = 1
    if losses is not None:
        n_plots += 1
    if epsilon_history is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # TODO: Plot de puntuaciones
    ax_idx = 0
    axes[ax_idx].plot(scores, alpha=0.6, color='blue', label='Score por episodio')
    
    # Calcular y plotear promedio móvil
    if len(scores) >= window_size:
        moving_avg = pd.Series(scores).rolling(window=window_size).mean()
        axes[ax_idx].plot(moving_avg, color='red', linewidth=2, 
                         label=f'Promedio móvil ({window_size} episodios)')
    
    axes[ax_idx].set_xlabel('Episodio')
    axes[ax_idx].set_ylabel('Puntuación')
    axes[ax_idx].set_title(f'{title} - Puntuaciones')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)
    
    # TODO: Plot de pérdidas si están disponibles
    if losses is not None:
        ax_idx += 1
        axes[ax_idx].plot(losses, color='orange', alpha=0.7)
        axes[ax_idx].set_xlabel('Paso de entrenamiento')
        axes[ax_idx].set_ylabel('Pérdida')
        axes[ax_idx].set_title(f'{title} - Pérdida de entrenamiento')
        axes[ax_idx].grid(True, alpha=0.3)
    
    # TODO: Plot de epsilon si está disponible
    if epsilon_history is not None:
        ax_idx += 1
        axes[ax_idx].plot(epsilon_history, color='green', alpha=0.7)
        axes[ax_idx].set_xlabel('Episodio')
        axes[ax_idx].set_ylabel('Epsilon')
        axes[ax_idx].set_title(f'{title} - Decaimiento de Epsilon')
        axes[ax_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # TODO: Guardar si se especifica ruta
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_q_values(q_values, states=None, actions=None, title="Q-Values Heatmap", 
                  save_path=None):
    """
    Visualiza los valores Q como un heatmap.
    
    Args:
        q_values: Array 2D con valores Q (estados x acciones)
        states: Etiquetas para estados (opcional)
        actions: Etiquetas para acciones (opcional)
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    # TODO: Crear heatmap de valores Q
    plt.figure(figsize=(10, 8))
    
    # Crear heatmap
    sns.heatmap(q_values, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=actions if actions else range(q_values.shape[1]),
                yticklabels=states if states else range(q_values.shape[0]))
    
    plt.title(title)
    plt.xlabel('Acciones')
    plt.ylabel('Estados')
    
    # TODO: Guardar si se especifica ruta
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap guardado en: {save_path}")
    
    plt.show()


def plot_comparison(results_dict, metric='scores', title="Comparison", 
                   save_path=None, window_size=100):
    """
    Compara resultados de múltiples experimentos.
    
    Args:
        results_dict: Diccionario con resultados {nombre: datos}
        metric: Métrica a comparar ('scores', 'losses', etc.)
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico
        window_size: Tamaño de ventana para promedio móvil
    """
    # TODO: Crear gráfico de comparación
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (name, data) in enumerate(results_dict.items()):
        if metric in data:
            values = data[metric]
            plt.plot(values, alpha=0.3, color=colors[i])
            
            # Promedio móvil
            if len(values) >= window_size:
                moving_avg = pd.Series(values).rolling(window=window_size).mean()
                plt.plot(moving_avg, color=colors[i], linewidth=2, label=name)
    
    plt.xlabel('Episodio')
    plt.ylabel(metric.capitalize())
    plt.title(f'{title} - {metric.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # TODO: Guardar si se especifica ruta
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparación guardada en: {save_path}")
    
    plt.show()


def plot_policy_visualization(env, agent, title="Policy Visualization", save_path=None):
    """
    Visualiza la política aprendida por el agente.
    
    Args:
        env: Entorno de gymnasium
        agent: Agente entrenado
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico
    """
    # TODO: Implementar visualización de política
    # Esta función dependerá del tipo de entorno
    print("TODO: Implementar visualización de política específica para el entorno")
    pass


def save_plots(figures, base_path="plots", prefix=""):
    """
    Guarda múltiples figuras en archivos.
    
    Args:
        figures: Lista de figuras de matplotlib
        base_path: Directorio base para guardar
        prefix: Prefijo para nombres de archivo
    """
    # TODO: Implementar guardado de múltiples figuras
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    for i, fig in enumerate(figures):
        filename = f"{prefix}plot_{i+1}.png" if prefix else f"plot_{i+1}.png"
        filepath = Path(base_path) / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {filepath}")


def create_summary_report(results, save_path="results/summary_report.html"):
    """
    Crea un reporte HTML con resumen de resultados.
    
    Args:
        results: Diccionario con resultados de experimentos
        save_path: Ruta para guardar el reporte HTML
    """
    # TODO: Implementar generación de reporte HTML
    print("TODO: Implementar generación de reporte HTML con resultados")
    pass
