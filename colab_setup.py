"""
Script de configuración para Google Colab.

Este script configura el entorno y los imports necesarios para ejecutar
el código DQN en Google Colab.
"""

import sys
import os
from pathlib import Path

def setup_colab_environment():
    """
    Configura el entorno para Google Colab.
    """
    # Agregar el directorio actual al path de Python
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Crear directorios necesarios
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("✅ Entorno configurado para Google Colab")
    print(f"Directorio de trabajo: {current_dir}")
    print(f"Python path actualizado")

if __name__ == "__main__":
    setup_colab_environment()
