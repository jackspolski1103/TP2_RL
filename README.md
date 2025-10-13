# TP Deep Q-Network (DQN) - Aprendizaje por Refuerzo

Este proyecto implementa algoritmos de aprendizaje por refuerzo, incluyendo Q-Learning tradicional y Deep Q-Networks (DQN), para resolver diferentes entornos de control.

## Estructura del Proyecto

```
TP2_RL/
├── main.py                     # Punto de entrada principal
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
├── envs/                       # Entornos personalizados
│   ├── constant_env.py         # Entorno con observaciones constantes
│   ├── random_obs_env.py       # Entorno con observaciones aleatorias
│   └── two_step_env.py         # Entorno simple de dos pasos
├── qlearning/                  # Implementación Q-Learning tradicional
│   └── qlearning_agent.py      # Agente Q-Learning con tabla
├── dqn/                        # Implementación Deep Q-Network
│   ├── dqn_agent.py           # Agente DQN con redes neuronales
│   ├── train_cartpole.py      # Entrenamiento en CartPole
│   └── train_breakout.py      # Entrenamiento en Breakout
├── utils/                      # Utilidades auxiliares
│   ├── replay_buffer.py       # Buffer de experiencias
│   └── plotting.py            # Funciones de visualización
└── runs/                       # Logs de TensorBoard (vacía)
```

## Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd TP2_RL
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar desde main.py

```bash
# Entrenar DQN en CartPole
python main.py --algorithm dqn --env cartpole --train

# Evaluar agente entrenado
python main.py --algorithm dqn --env cartpole --test

# Usar Q-Learning en entorno simple
python main.py --algorithm qlearning --env two_step --train
```

### Entrenar directamente

```bash
# CartPole con DQN
python dqn/train_cartpole.py --episodes 1000

# Breakout con DQN (requiere mucho tiempo)
python dqn/train_breakout.py --episodes 10000

# Evaluar modelo entrenado
python dqn/train_cartpole.py --eval --model models/dqn_cartpole.pth
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

## TODO - Implementaciones Pendientes

Los archivos contienen la estructura y documentación completa, pero requieren implementación de:

- [ ] Lógica de entornos personalizados
- [ ] Algoritmo Q-Learning completo
- [ ] Implementación DQN con PyTorch
- [ ] Loops de entrenamiento
- [ ] Funciones de visualización
- [ ] Integración con TensorBoard

## Dependencias Principales

- PyTorch >= 2.0.0
- Gymnasium >= 1.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- tqdm >= 4.62.0

## Contribución

Este es un proyecto académico. Para contribuir:

1. Implementa las funciones marcadas con `# TODO:`
2. Añade tests para nuevas funcionalidades
3. Actualiza la documentación según sea necesario

## Licencia

Proyecto académico para fines educativos.
