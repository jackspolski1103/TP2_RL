# DQN (Deep Q-Network) Implementation

Este directorio contiene la implementación completa del algoritmo Deep Q-Network (DQN) siguiendo estrictamente el **Algoritmo 2** del paper original de Mnih et al. (2015).

## 🎯 Archivos Principales

### `dqn_agent.py`
Implementación del agente DQN con:
- **Arquitectura MLP** para CartPole (Linear 64 → ReLU → Linear 64 → ReLU → Linear output)
- **Arquitectura CNN** para Breakout (basada en el paper original de DQN)
- **Red online y target** con hard updates cada N steps
- **Replay buffer circular** para estabilizar el entrenamiento
- **Política epsilon-greedy** con decaimiento exponencial
- **MSE loss** para entrenamiento (como en el paper original)
- **Optimizer Adam** con learning rate configurable

### `train_cartpole.py`
Script de entrenamiento para CartPole-v1:
- **Parámetros típicos**: 1000 episodios, γ=0.99, lr=1e-3, buffer=50k
- **Criterio de "resuelto"**: promedio ≥195 en 100 episodios consecutivos
- **Logging completo** a TensorBoard
- **Guardado automático** del mejor modelo

### `train_breakout.py`
Script de entrenamiento para Breakout-v5:
- **Preprocesamiento de imágenes**: 84x84 escala de grises
- **Frame stacking**: 4 frames consecutivos
- **Parámetros para Atari**: buffer=500k, start_learning=50k, lr=2.5e-4
- **Reward clipping**: [-1, +1] como en el paper
- **Life loss detection**: pérdida de vida como episodio terminado

## 🚀 Uso Rápido

### Entrenar en CartPole
```bash
# Entrenamiento básico
uv run dqn/train_cartpole.py

# Con parámetros personalizados
uv run dqn/train_cartpole.py --episodes 1500 --lr 0.001 --epsilon-decay 0.002

# Evaluar modelo entrenado
uv run dqn/train_cartpole.py --eval --render
```

### Entrenar en Breakout
```bash
# Entrenamiento básico (¡puede tardar horas!)
uv run dqn/train_breakout.py --episodes 1000

# Con parámetros personalizados
uv run dqn/train_breakout.py --episodes 2000 --lr 0.0001

# Evaluar modelo entrenado
uv run dqn/train_breakout.py --eval --render
```

## 📊 TensorBoard

### Cómo Lanzar TensorBoard

Para visualizar los logs de entrenamiento en tiempo real:

```bash
# Desde el directorio raíz del proyecto
tensorboard --logdir runs

# Para un entorno específico
tensorboard --logdir runs/cartpole
tensorboard --logdir runs/breakout

# Con puerto personalizado
tensorboard --logdir runs --port 6007
```

Luego abre tu navegador en: `http://localhost:6006`

### Métricas Registradas

El sistema registra automáticamente las siguientes métricas:

#### 📈 **Métricas de Entrenamiento**
- **`train/episode_reward`**: Recompensa total por episodio
- **`train/avg_reward_100`**: Promedio móvil de recompensas (ventana 100)
- **`train/loss`**: Pérdida MSE del entrenamiento
- **`train/epsilon`**: Valor actual de epsilon (exploración)

#### 🎮 **Métricas por Entorno**

**CartPole-v1:**
- Objetivo: Mantener el poste equilibrado
- Recompensa máxima por episodio: 500
- "Resuelto" cuando avg_reward_100 ≥ 195

**Breakout-v5:**
- Objetivo: Romper todos los ladrillos
- Recompensa variable según ladrillos rotos
- "Resuelto" cuando avg_reward_100 ≥ 30

### Estructura de Logs

```
runs/
├── cartpole/
│   └── 20241213_143022/  # Timestamp del entrenamiento
│       ├── events.out.tfevents.xxx
│       └── ...
└── breakout/
    └── 20241213_150045/
        ├── events.out.tfevents.xxx
        └── ...
```

## 🔧 Configuración de Hiperparámetros

### CartPole (MLP)
```python
# Configuración por defecto
episodes = 1000
gamma = 0.99
lr = 1e-3
buffer_capacity = 50000
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.001
batch_size = 32
target_update_freq = 1000
start_learning = 1000
```

### Breakout (CNN)
```python
# Configuración por defecto
episodes = 1000
gamma = 0.99
lr = 2.5e-4  # Más bajo para Atari
buffer_capacity = 500000  # Más grande
epsilon_start = 1.0
epsilon_min = 0.1  # Más alto para Atari
epsilon_decay = 0.0001  # Más lento
batch_size = 32
target_update_freq = 10000  # Menos frecuente
start_learning = 50000  # Mucho más alto
```

## 📁 Modelos Guardados

Los modelos entrenados se guardan automáticamente en:

```
checkpoints/
├── dqn_cartpole.pt    # Mejor modelo de CartPole
└── dqn_breakout.pt    # Mejor modelo de Breakout
```

### Cargar y Usar Modelos

```python
from dqn.dqn_agent import DQNAgent

# Crear agente
agent = DQNAgent(state_size=4, action_size=2, model_type="mlp")

# Cargar modelo entrenado
agent.load_model("checkpoints/dqn_cartpole.pt")

# Usar para predicción
action = agent.select_action(state, epsilon=0.0)  # Política greedy
```

## 🧪 Experimentos y Comparaciones

### Comparar con REINFORCE
```bash
# Generar comparación DQN vs REINFORCE
uv run experiments/compare_dqn_vs_reinforce.py \
    --dqn-tensorboard runs/cartpole/20241213_143022 \
    --reinforce-data path/to/reinforce_results.json \
    --output-dir plots
```

### Análisis de Resultados

Los scripts generan automáticamente:
1. **Gráficos de entrenamiento** (recompensa vs episodios)
2. **Análisis de eficiencia** (recompensa vs tiempo)
3. **Comparaciones** entre algoritmos
4. **Reportes de texto** con estadísticas

## 🐛 Troubleshooting

### Problemas Comunes

**1. CUDA out of memory (Breakout)**
```bash
# Reducir batch size
uv run dqn/train_breakout.py --batch-size 16

# O usar CPU
CUDA_VISIBLE_DEVICES="" uv run dqn/train_breakout.py
```

**2. TensorBoard no muestra datos**
```bash
# Verificar que los logs existen
ls -la runs/

# Lanzar con verbose
tensorboard --logdir runs --verbose
```

**3. Entrenamiento muy lento**
```bash
# Verificar que usa GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reducir frecuencia de logging
# (modificar código para log cada N episodios)
```

### Logs de Debug

Para debug detallado, activar logging en el código:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Referencias

- **Paper Original**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- **Gymnasium**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **TensorBoard**: [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)

## 🎓 Notas Académicas

Esta implementación sigue **estrictamente el Algoritmo 2** del paper original:
- ✅ Experience replay con buffer circular
- ✅ Target network con hard updates
- ✅ Epsilon-greedy con decaimiento exponencial
- ✅ MSE loss (no Huber por defecto)
- ✅ Adam optimizer
- ✅ Preprocesamiento estándar para Atari

**Diferencias con implementaciones modernas:**
- No usa Double DQN, Dueling DQN, o Prioritized Replay (extensiones posteriores)
- No usa Huber loss (aunque es común en implementaciones modernas)
- Frame stacking manual (no usa wrappers de Gymnasium)

Esto es intencional para mantener fidelidad al algoritmo original del paper.
