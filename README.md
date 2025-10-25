# TP Deep Q-Network (DQN) - Aprendizaje por Refuerzo

Este proyecto implementa algoritmos de aprendizaje por refuerzo, incluyendo Q-Learning tradicional y Deep Q-Networks (DQN), para resolver diferentes entornos de control.

## Estructura del Proyecto

```
TP2_RL/
├── 📄 main.py                          # Punto de entrada principal con CLI
├── 📄 pyproject.toml                   # Configuración del proyecto con uv
├── 📄 requirements.txt                 # Dependencias pip (legacy)
├── 📄 README.md                        # Documentación principal
├── 📄 IMPLEMENTATION_SUMMARY.md        # Resumen de implementación completa
├── 📄 INFORME_COMPLETO.md              # Informe detallado con resultados
├── 📄 TP2 DQN-1.2.pdf                 # Enunciado del trabajo práctico
│
├── 📁 envs/                            # Entornos personalizados para testing
│   ├── constant_env.py                 # Entorno con recompensa constante (+1)
│   ├── random_obs_env.py               # Entorno con observaciones aleatorias
│   └── two_step_env.py                 # Entorno minimal de dos estados
│
├── 📁 qlearning/                       # Implementación Q-Learning tabular
│   ├── qlearning_agent.py              # Agente Q-Learning con tabla Q
│   ├── train_custom_envs.py           # Entrenamiento en entornos personalizados
│   ├── simple_evaluation.py           # Evaluación simple de agentes
│   ├── plot_simple_curves.py          # Visualización de curvas de aprendizaje
│   └── README.md                       # Documentación específica de Q-Learning
│
├── 📁 dqn/                             # Implementación Deep Q-Network
│   ├── dqn_agent.py                    # Agente DQN con MLP y CNN
│   ├── train_cartpole.py              # Entrenamiento en CartPole-v1
│   ├── train_breakout.py              # Entrenamiento en Breakout-v0
│   ├── train_custom_envs.py           # Entrenamiento en entornos personalizados
│   ├── train_cartpole_with_plots.py   # Entrenamiento con visualizaciones
│   ├── minatar_wrapper.py             # Wrapper para entornos MinAtar
│   └── README.md                       # Documentación específica de DQN
│
├── 📁 reinforce/                       # Implementación REINFORCE
│   └── reinforce.py                    # Algoritmo REINFORCE completo
│
├── 📁 utils/                           # Utilidades auxiliares
│   ├── replay_buffer.py               # Buffer de experiencias circular
│   └── plotting.py                    # Funciones de visualización
│
├── 📁 experiments/                     # Experimentos y comparaciones
│   ├── compare_dqn_vs_reinforce.py    # Comparación DQN vs REINFORCE
│   └── Captura de pantalla 2025-10-22 a la(s) 3.12.19 p. m..png
│
├── 📁 plots/                           # Gráficos generados
│   ├── dqn_cartpole_*.png             # Gráficos de entrenamiento DQN en CartPole
│   ├── dqn_custom_envs_*.png          # Gráficos DQN en entornos personalizados
│   ├── qlearning_*.png                # Gráficos de Q-Learning
│   └── reinforce_*.png                # Gráficos de REINFORCE
│
├── 📁 runs/                            # Logs de TensorBoard
│   ├── cartpole/                       # Logs de entrenamiento en CartPole
│   └── breakout/                       # Logs de entrenamiento en Breakout
│
├── 📁 checkpoints/                     # Modelos entrenados guardados
│   ├── dqn_cartpole.pt                # Modelo DQN entrenado en CartPole
│   └── dqn_breakout.pt                # Modelo DQN entrenado en Breakout
│
├── 📄 colab_commands.py               # Comandos de ejemplo para Google Colab
├── 📄 colab_setup.py                  # Configuración del entorno en Colab
├── 📄 install_colab.py                # Instalación de dependencias en Colab
└── 📄 uv.lock                         # Lock file de dependencias con uv
```

## Descripción Detallada de Archivos y Carpetas

### 📄 Archivos Principales

- **`main.py`**: Punto de entrada principal con interfaz de línea de comandos para ejecutar diferentes experimentos de RL
- **`pyproject.toml`**: Configuración moderna del proyecto usando `uv` como gestor de paquetes
- **`requirements.txt`**: Dependencias pip (legacy) para compatibilidad
- **`README.md`**: Documentación principal del proyecto
- **`IMPLEMENTATION_SUMMARY.md`**: Resumen detallado de la implementación completa
- **`INFORME_COMPLETO.md`**: Informe académico con resultados y análisis
- **`TP2 DQN-1.2.pdf`**: Enunciado original del trabajo práctico

### 📁 envs/ - Entornos Personalizados
Carpeta con entornos de prueba diseñados para validar algoritmos de RL:

- **`constant_env.py`**: Entorno que siempre devuelve recompensa +1, útil para verificar que la función de valor aprende correctamente
- **`random_obs_env.py`**: Entorno con observaciones aleatorias para probar robustez de algoritmos
- **`two_step_env.py`**: Entorno minimal de dos estados para debugging y pruebas básicas

### 📁 qlearning/ - Q-Learning Tabular
Implementación del algoritmo Q-Learning tradicional con tabla de valores:

- **`qlearning_agent.py`**: Agente Q-Learning con política epsilon-greedy y decaimiento de epsilon
- **`train_custom_envs.py`**: Scripts de entrenamiento en entornos personalizados
- **`simple_evaluation.py`**: Herramientas de evaluación de agentes Q-Learning
- **`plot_simple_curves.py`**: Visualización de curvas de aprendizaje
- **`README.md`**: Documentación específica del algoritmo Q-Learning

### 📁 dqn/ - Deep Q-Network
Implementación completa del algoritmo DQN siguiendo el paper original de Mnih et al. (2015):

- **`dqn_agent.py`**: Agente DQN con soporte para MLP (CartPole) y CNN (Breakout)
- **`train_cartpole.py`**: Entrenamiento en CartPole-v1 con logging a TensorBoard
- **`train_breakout.py`**: Entrenamiento en Breakout-v5 con preprocesamiento de imágenes
- **`train_custom_envs.py`**: Entrenamiento DQN en entornos personalizados
- **`train_cartpole_with_plots.py`**: Versión con visualizaciones adicionales
- **`minatar_wrapper.py`**: Wrapper para entornos MinAtar
- **`README.md`**: Documentación detallada del algoritmo DQN

### 📁 reinforce/ - REINFORCE
Implementación del algoritmo REINFORCE (Policy Gradient):

- **`reinforce.py`**: Algoritmo REINFORCE completo con logging a TensorBoard

### 📁 utils/ - Utilidades Auxiliares
Herramientas compartidas entre diferentes algoritmos:

- **`replay_buffer.py`**: Buffer de experiencias circular para DQN
- **`plotting.py`**: Funciones de visualización y generación de gráficos

### 📁 experiments/ - Experimentos y Comparaciones
Scripts para experimentos comparativos:

- **`compare_dqn_vs_reinforce.py`**: Comparación detallada entre DQN y REINFORCE en CartPole
- **`Captura de pantalla 2025-10-22 a la(s) 3.12.19 p. m..png`**: Captura de resultados

### 📁 plots/ - Gráficos Generados
Visualizaciones de resultados de entrenamiento:

- **`dqn_cartpole_*.png`**: Gráficos de entrenamiento DQN en CartPole (recompensas, loss, tiempo)
- **`dqn_custom_envs_*.png`**: Gráficos DQN en entornos personalizados
- **`qlearning_*.png`**: Gráficos de Q-Learning con diferentes hiperparámetros
- **`reinforce_*.png`**: Gráficos de entrenamiento REINFORCE

### 📁 runs/ - Logs de TensorBoard
Logs de entrenamiento para visualización en TensorBoard:

- **`cartpole/`**: Logs de entrenamiento en CartPole con diferentes algoritmos
- **`breakout/`**: Logs de entrenamiento en Breakout con DQN

### 📁 checkpoints/ - Modelos Entrenados
Modelos guardados después del entrenamiento:

- **`dqn_cartpole.pt`**: Modelo DQN entrenado en CartPole
- **`dqn_breakout.pt`**: Modelo DQN entrenado en Breakout

### 📄 Archivos de Google Colab
Scripts específicos para ejecutar en Google Colab:

- **`colab_commands.py`**: Comandos de ejemplo para entrenamiento en Colab
- **`colab_setup.py`**: Configuración del entorno en Colab
- **`install_colab.py`**: Instalación automática de dependencias en Colab

### 📄 Archivos de Configuración
- **`uv.lock`**: Lock file de dependencias generado por `uv`

## Instalación

### Método Recomendado (usando uv)

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd TP2_RL
```

2. Instala las dependencias con uv:
```bash
uv sync
```

3. Activa el entorno virtual:
```bash
source .venv/bin/activate
```

### Método Alternativo (usando pip)

```bash
pip install -r requirements.txt
```

### Para Google Colab

```bash
# Ejecutar en Colab
!python install_colab.py
```

## Uso

### Ejecutar desde main.py

```bash
# Entrenar DQN en CartPole
uv run main.py --algorithm dqn --env cartpole --train

# Evaluar agente entrenado
uv run main.py --algorithm dqn --env cartpole --test

# Usar Q-Learning en entorno simple
uv run main.py --algorithm qlearning --env two_step --train
```

### Entrenar directamente

```bash
# CartPole con DQN
uv run dqn/train_cartpole.py --episodes 1000

# Breakout con DQN (requiere mucho tiempo)
uv run dqn/train_breakout.py --episodes 10000

# Evaluar modelo entrenado
uv run dqn/train_cartpole.py --eval --model checkpoints/dqn_cartpole.pt

# Q-Learning en entornos personalizados
uv run qlearning/train_custom_envs.py

# Comparación DQN vs REINFORCE
uv run experiments/compare_dqn_vs_reinforce.py
```

### Comandos Específicos por Algoritmo

#### DQN (Deep Q-Network)
```bash
# Entrenamiento básico en CartPole
uv run dqn/train_cartpole.py

# Entrenamiento con parámetros personalizados
uv run dqn/train_cartpole.py --episodes 1500 --lr 0.001 --epsilon-decay 0.002

# Entrenamiento en Breakout (requiere GPU recomendada)
uv run dqn/train_breakout.py --episodes 2000

# Entrenamiento en entornos personalizados
uv run dqn/train_custom_envs.py
```

#### Q-Learning Tabular
```bash
# Entrenamiento en entornos personalizados
uv run qlearning/train_custom_envs.py

# Evaluación simple
uv run qlearning/simple_evaluation.py

# Visualización de curvas
uv run qlearning/plot_simple_curves.py
```

#### REINFORCE
```bash
# Entrenamiento REINFORCE en CartPole
uv run reinforce/reinforce.py
```

## Entornos Disponibles

### Entornos Estándar
- **CartPole**: Control clásico de equilibrio de poste
- **Breakout**: Juego de Atari con procesamiento de imágenes

### Entornos Personalizados
- **ConstantEnv**: Observaciones constantes para testing
- **RandomObsEnv**: Observaciones aleatorias para robustez
- **TwoStepEnv**: Entorno minimal de dos estados

## Algoritmos Implementados

### Q-Learning Tradicional
- Tabla de valores Q para espacios discretos
- Política epsilon-greedy
- Decaimiento de epsilon

### Deep Q-Network (DQN)
- Aproximación de función Q con redes neuronales
- Replay buffer para estabilidad
- Target network para reducir correlaciones
- Soporte para CNN (Atari) y redes densas (control clásico)

## Características

- **Modular**: Fácil agregar nuevos entornos y algoritmos
- **Configurable**: Hiperparámetros ajustables
- **Visualización**: Gráficos de entrenamiento y análisis
- **Reproducible**: Semillas para resultados consistentes
- **Extensible**: Base para implementar variantes de DQN

## ✅ Estado de Implementación

**TODAS LAS IMPLEMENTACIONES ESTÁN COMPLETAS:**

- ✅ **Entornos personalizados**: Implementados y funcionales
- ✅ **Q-Learning tabular**: Algoritmo completo con evaluación
- ✅ **DQN con PyTorch**: Implementación siguiendo paper original
- ✅ **Loops de entrenamiento**: Completos con logging
- ✅ **Funciones de visualización**: Gráficos automáticos
- ✅ **Integración con TensorBoard**: Logging completo
- ✅ **REINFORCE**: Algoritmo Policy Gradient implementado
- ✅ **Experimentación**: Comparaciones DQN vs REINFORCE

## 📊 Resultados y Gráficos

El proyecto incluye visualizaciones completas de todos los experimentos:

### Gráficos de Entrenamiento DQN
- **CartPole**: Curvas de recompensa, loss y tiempo de entrenamiento
- **Breakout**: Progreso de entrenamiento con preprocesamiento de imágenes
- **Entornos personalizados**: Validación en entornos de prueba

### Gráficos de Q-Learning
- **Curvas de hiperparámetros**: Comparación de diferentes configuraciones
- **Evaluación de rendimiento**: Análisis de convergencia

### Comparaciones Algorítmicas
- **DQN vs REINFORCE**: Análisis comparativo en CartPole
- **Métricas de rendimiento**: Tiempo de entrenamiento, estabilidad, convergencia

### TensorBoard
Todos los entrenamientos generan logs detallados en `runs/` para visualización interactiva:
```bash
tensorboard --logdir runs/
```

## Dependencias Principales

- PyTorch >= 2.0.0
- Gymnasium >= 1.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- tqdm >= 4.62.0

## 🚀 Características Avanzadas

### Soporte para Google Colab
- Scripts automáticos de instalación y configuración
- Comandos optimizados para entornos de Colab
- Detección automática de GPU

### Logging y Monitoreo
- **TensorBoard**: Logging completo de métricas de entrenamiento
- **Gráficos automáticos**: Generación de visualizaciones
- **Checkpoints**: Guardado automático de modelos

### Experimentación
- **Hiperparámetros**: Configuración flexible de todos los parámetros
- **Comparaciones**: Scripts para comparar diferentes algoritmos
- **Análisis**: Herramientas de evaluación y visualización

## Contribución

Este es un proyecto académico completo. Para extensiones futuras:

1. Implementar variantes de DQN (Double DQN, Dueling DQN, etc.)
2. Añadir más entornos de prueba
3. Implementar otros algoritmos de RL (A2C, PPO, etc.)
4. Mejorar la documentación y ejemplos

## Licencia

Proyecto académico para fines educativos.
