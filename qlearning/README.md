# Q-Learning Agent para FrozenLake-v1

Este módulo implementa un agente de Q-Learning tabular completo para el entorno FrozenLake-v1 (4x4) de Gymnasium, siguiendo la Parte 2 del TP de DQN.

## 🎯 Características Implementadas

### ✅ Clase QLearningAgent
- **Tabla Q**: Implementada con `defaultdict` para manejo automático de estados no visitados
- **Política epsilon-greedy**: Balanceo entre exploración y explotación
- **Actualización Q-Learning**: Regla estándar `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q[s',a'] − Q(s,a)]`
- **Decaimiento exponencial de epsilon**: `ε = min_epsilon + (max_epsilon - min_epsilon) * exp(-decay_rate * episode)`
- **Guardado/carga de modelos**: Serialización con pickle

### ✅ Función de Entrenamiento (`train_qlearning`)
- **Entorno**: FrozenLake-v1 (4x4) con superficie deslizante
- **Hiperparámetros por defecto**:
  - `n_episodes = 5000`
  - `alpha = 0.8` (tasa de aprendizaje)
  - `gamma = 0.99` (factor de descuento)
  - `epsilon_min = 0.1`
  - `decay_rate = 0.001`
- **Tracking de progreso**: Recompensas por episodio y promedios cada 100 episodios
- **Progreso visual**: Barra de progreso con `tqdm`

### ✅ Función de Evaluación (`evaluate_qlearning`)
- **Evaluación sin aprendizaje**: 100 episodios por defecto
- **Política configurable**: Greedy (ε=0.0) o epsilon-greedy
- **Métricas**: Tasa de éxito (porcentaje de llegadas a la meta)

### ✅ Visualización (`plot_training_curves`)
- **Gráfico de evolución**: Recompensa promedio vs episodios
- **Línea de tendencia suavizada**: Ventana móvil para mejor visualización
- **Guardado automático**: En `plots/qlearning_rewards.png`
- **Información adicional**: Rendimiento final en el gráfico

## 📊 Resultados Típicos

Con los hiperparámetros por defecto, el agente típicamente alcanza:
- **Tasa de éxito**: 20-30% (FrozenLake es un entorno desafiante)
- **Convergencia**: Alrededor de 2000-3000 episodios
- **Epsilon final**: ~0.1 (valor mínimo configurado)

## 🚀 Uso

### Ejecución Básica
```bash
# Entrenar y evaluar con configuración por defecto
uv run qlearning/qlearning_agent.py
```

### Uso Programático
```python
from qlearning.qlearning_agent import train_qlearning, evaluate_qlearning, plot_training_curves

# Entrenar agente
q_table, rewards, avg_rewards = train_qlearning(
    n_episodes=5000,
    alpha=0.8,
    gamma=0.99,
    epsilon_min=0.1,
    decay_rate=0.001
)

# Evaluar agente
success_rate = evaluate_qlearning(q_table, n_eval_episodes=100)
print(f"Tasa de éxito: {success_rate*100:.2f}%")

# Generar gráficos
plot_training_curves(avg_rewards)
```

### Demo Interactivo
```bash
# Ejecutar demo con múltiples opciones
uv run qlearning/demo_qlearning.py
```

## 📁 Archivos

- **`qlearning_agent.py`**: Implementación principal del agente Q-Learning
- **`demo_qlearning.py`**: Script de demostración interactivo con comparaciones
- **`README.md`**: Esta documentación

## 🔧 Hiperparámetros

### Parámetros Principales
- **`alpha` (learning_rate)**: Controla qué tan rápido aprende el agente
  - Alto (0.8-0.9): Aprendizaje rápido pero potencialmente inestable
  - Bajo (0.1-0.3): Aprendizaje lento pero más estable
  
- **`gamma` (discount_factor)**: Importancia de recompensas futuras
  - Alto (0.99): Considera recompensas a largo plazo
  - Bajo (0.9): Se enfoca en recompensas inmediatas
  
- **`decay_rate`**: Velocidad de decaimiento de epsilon
  - Alto (0.005): Transición rápida a explotación
  - Bajo (0.0005): Exploración prolongada

### Configuraciones Recomendadas

**Para convergencia rápida:**
```python
alpha=0.9, gamma=0.99, decay_rate=0.005
```

**Para estabilidad:**
```python
alpha=0.3, gamma=0.95, decay_rate=0.001
```

**Para exploración extendida:**
```python
alpha=0.5, gamma=0.99, decay_rate=0.0005
```

## 📈 Análisis de Resultados

El script genera automáticamente:

1. **Gráfico de entrenamiento**: Evolución de la recompensa promedio
2. **Métricas de rendimiento**: Tasa de éxito final
3. **Información de convergencia**: Epsilon final y estadísticas

Los gráficos se guardan en la carpeta `plots/` y son ideales para incluir en informes académicos.

## 🎮 Sobre FrozenLake-v1

FrozenLake es un entorno de navegación en cuadrícula donde:
- **Objetivo**: Llegar desde el inicio (S) hasta la meta (G)
- **Obstáculos**: Hoyos (H) que terminan el episodio sin recompensa
- **Superficie deslizante**: Las acciones tienen probabilidad de fallar
- **Recompensa**: +1 solo al llegar a la meta, 0 en otros casos

### Mapa 4x4
```
SFFF
FHFH
FFFH
HFFG
```

Donde:
- S: Inicio (Start)
- F: Superficie congelada (Frozen)
- H: Hoyo (Hole)
- G: Meta (Goal)

## 🔍 Debugging y Análisis

El módulo incluye herramientas para analizar el comportamiento del agente:

- **Visualización de tabla Q**: Ver valores aprendidos para estados clave
- **Comparación de hiperparámetros**: Evaluar diferentes configuraciones
- **Análisis de política**: Observar las acciones preferidas por estado

Esto facilita la comprensión del proceso de aprendizaje y la optimización de hiperparámetros.
