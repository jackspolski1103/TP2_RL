"""
Punto de entrada principal para el TP de Deep Q-Network (DQN).

Este módulo permite ejecutar diferentes experimentos de aprendizaje por refuerzo
utilizando Q-Learning tradicional y Deep Q-Networks en varios entornos.
"""

import argparse
import sys
from pathlib import Path

def main():
    """
    Función principal que maneja la ejecución de diferentes experimentos.
    
    Permite seleccionar entre:
    - Q-Learning tradicional
    - DQN en CartPole
    - DQN en Breakout
    - Entornos de prueba (constant, random_obs, two_step)
    """
    parser = argparse.ArgumentParser(description="TP Deep Q-Network")
    parser.add_argument("--algorithm", choices=["qlearning", "dqn"], 
                       default="dqn", help="Algoritmo a utilizar")
    parser.add_argument("--env", choices=["cartpole", "breakout", "constant", "random_obs", "two_step"],
                       default="cartpole", help="Entorno a utilizar")
    parser.add_argument("--train", action="store_true", help="Entrenar el agente")
    parser.add_argument("--test", action="store_true", help="Evaluar el agente")
    
    args = parser.parse_args()
    
    # TODO: Implementar lógica de ejecución según argumentos
    print(f"Ejecutando {args.algorithm} en entorno {args.env}")
    if args.train:
        print("Modo: Entrenamiento")
    if args.test:
        print("Modo: Evaluación")

if __name__ == "__main__":
    main()
