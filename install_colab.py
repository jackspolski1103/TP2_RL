"""
Script de instalación para Google Colab.

Este script instala todas las dependencias necesarias para ejecutar
el código DQN en Google Colab.
"""

import subprocess
import sys
import os

def install_package(package):
    """Instala un paquete usando pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado correctamente")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")

def setup_colab():
    """Configura el entorno de Colab."""
    print("🚀 Configurando entorno para Google Colab...")
    
    # Lista de paquetes necesarios
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "gymnasium>=1.0.0",
        "gymnasium[atari]>=1.0.0",
        "gymnasium[classic_control]>=1.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
        "opencv-python>=4.5.0",
        "minatar"  # Paquete específico para MinAtar
    ]
    
    print("📦 Instalando dependencias...")
    for package in packages:
        install_package(package)
    
    # Crear directorios necesarios
    directories = ["checkpoints", "runs", "plots", "runs/breakout"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directorio creado: {directory}")
    
    print("\n✅ Configuración completada!")
    print("Ahora puedes ejecutar el entrenamiento de DQN en Breakout")

if __name__ == "__main__":
    setup_colab()
