"""
Comandos de ejemplo para Google Colab.

Este script contiene ejemplos de cómo ejecutar el entrenamiento DQN
con diferentes configuraciones en Google Colab.
"""

def show_colab_commands():
    """Muestra comandos de ejemplo para Google Colab."""
    
    print("🚀 COMANDOS PARA GOOGLE COLAB")
    print("=" * 50)
    
    print("\n📦 1. INSTALACIÓN BÁSICA:")
    print("!git clone https://github.com/jackspolski1103/TP2_RL.git")
    print("%cd TP2_RL")
    print("!python install_colab.py")
    
    print("\n🏃 2. ENTRENAMIENTO BÁSICO (1000 episodios):")
    print("!python dqn/train_breakout.py --episodes 1000")
    
    print("\n⚡ 3. ENTRENAMIENTO RÁPIDO (500 episodios, sin TensorBoard):")
    print("!python dqn/train_breakout.py --episodes 500 --no-tensorboard")
    
    print("\n🎯 4. ENTRENAMIENTO PERSONALIZADO:")
    print("!python dqn/train_breakout.py \\")
    print("    --episodes 2000 \\")
    print("    --lr 0.001 \\")
    print("    --gamma 0.95 \\")
    print("    --epsilon-start 0.9 \\")
    print("    --epsilon-decay 0.01 \\")
    print("    --batch-size 64 \\")
    print("    --buffer-capacity 50000")
    
    print("\n🔬 5. EXPERIMENTO CON DIFERENTES PARÁMETROS:")
    print("# Aprendizaje más agresivo")
    print("!python dqn/train_breakout.py --episodes 1000 --lr 0.01 --epsilon-decay 0.01")
    print("\n# Aprendizaje más conservador")
    print("!python dqn/train_breakout.py --episodes 1000 --lr 0.001 --epsilon-decay 0.001")
    
    print("\n📊 6. EVALUAR MODELO ENTRENADO:")
    print("!python dqn/train_breakout.py --eval --model checkpoints/dqn_breakout.pt")
    
    print("\n📋 7. VER TODOS LOS PARÁMETROS DISPONIBLES:")
    print("!python dqn/train_breakout.py --help")
    
    print("\n" + "=" * 50)
    print("💡 CONSEJOS:")
    print("• Usa --no-tensorboard para entrenamiento más rápido")
    print("• Aumenta --episodes para mejor rendimiento")
    print("• Ajusta --lr entre 0.0001 y 0.01")
    print("• --epsilon-decay controla la velocidad de exploración")
    print("• --batch-size afecta la estabilidad del entrenamiento")

if __name__ == "__main__":
    show_colab_commands()
