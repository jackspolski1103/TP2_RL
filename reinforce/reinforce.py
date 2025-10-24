
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


# -----------------------------
# TensorBoard Logger
# -----------------------------
class TensorBoardLogger:
    def __init__(self, log_dir: str, env_name: str, experiment_name: Optional[str] = None):
        """
        Logger para TensorBoard con métricas específicas de REINFORCE.
        
        Args:
            log_dir: Directorio base para logs
            env_name: Nombre del entorno
            experiment_name: Nombre opcional del experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or f"reinforce_{env_name}_{timestamp}"
        self.log_path = os.path.join(log_dir, exp_name)
        
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.env_name = env_name
        print(f"📊 TensorBoard logs: {self.log_path}")
        print(f"   Para visualizar: tensorboard --logdir {log_dir}")
    
    def log_episode(self, episode: int, ep_return: float):
        """Log return por episodio."""
        self.writer.add_scalar(f'{self.env_name}/Episode_Return', ep_return, episode)
    
    def log_episode_averages(self, episode: int, avg_return_10: float):
        """Log promedio móvil de returns por episodio."""
        self.writer.add_scalar(f'{self.env_name}/Avg_Return_10', avg_return_10, episode)
    
    def log_episode_loss(self, episode: int, loss: float):
        """Log loss por episodio."""
        self.writer.add_scalar(f'{self.env_name}/Episode_Loss', loss, episode)
    
    def log_reference_baselines(self, episode: int, random_mean: float, always_means: Dict[int, float]):
        """Log de líneas base de referencia como curvas horizontales."""
        self.writer.add_scalar(f'{self.env_name}/Baseline_Random', random_mean, episode)
        for k, val in always_means.items():
            self.writer.add_scalar(f'{self.env_name}/Baseline_Always_{k}', val, episode)
    
    def log_all_returns_together(self, episode: int, ep_return: float, avg_return_10: float, 
                                random_mean: float, always_means: Dict[int, float]):
        """Log todas las métricas de returns en un solo gráfico usando add_scalars."""
        scalars_dict = {
            'Episode_Return': ep_return,
            'Avg_Return_10': avg_return_10,
            'Baseline_Random': random_mean,
        }
        
        # Agregar baselines de acciones siempre-k
        for k, val in always_means.items():
            scalars_dict[f'Baseline_Always_{k}'] = val
        
        self.writer.add_scalars(f'{self.env_name}/Returns', scalars_dict, episode)

    
    def close(self):
        """Cerrar el writer."""
        self.writer.close()


# -----------------------------
# Utils: obs -> vector & returns
# -----------------------------
def infer_obs_dim(observation_space: gym.Space) -> int:
    if isinstance(observation_space, gym.spaces.Discrete):
        return int(observation_space.n)
    if isinstance(observation_space, gym.spaces.Box):
        return int(np.prod(observation_space.shape))
    raise NotImplementedError("Solo se soporta Discrete o Box para observations.")


def preprocess_obs(obs: Any, observation_space: gym.Space) -> torch.Tensor:
    if isinstance(observation_space, gym.spaces.Discrete):
        n = observation_space.n
        one_hot = np.zeros(n, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return torch.tensor(one_hot, dtype=torch.float32)
    if isinstance(observation_space, gym.spaces.Box):
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return torch.tensor(arr, dtype=torch.float32)
    raise NotImplementedError("Solo se soporta Discrete o Box para observations.")


def compute_discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = np.zeros(len(rewards), dtype=np.float32)
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma * run
        G[t] = run
    return torch.tensor(G, dtype=torch.float32)


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 1:
        return torch.zeros_like(x)
    std = x.std(unbiased=False)
    if std < eps:  # Si la varianza es muy pequeña, no normalizar
        return x - x.mean()  # Solo centrar
    return (x - x.mean()) / (std + eps)



def estimate_random_and_trivial_returns(
    env: gym.Env,
    episodes: int = 200,
    max_steps: Optional[int] = None,
    seed: Optional[int] = 0
):
    """
    Estima retornos promedio para:
      - agente aleatorio
      - agentes triviales que siempre eligen la acción k (para cada k válido)
    Devuelve: (random_mean, {k: mean_k})
    """
    if seed is not None:
        np.random.seed(seed)

    # Cap de seguridad para evitar loops infinitos si el env no termina
    if max_steps is None:
        # intenta leer de la spec, sino usa 200 por defecto
        max_steps_local = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        max_steps_local = int(max_steps_local) if max_steps_local is not None else 200
    else:
        max_steps_local = int(max_steps)

    def run_policy(select_action):
        total = 0.0
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            ep_ret = 0.0
            while not done and steps < max_steps_local:
                a = int(select_action(obs))
                # si algo raro pasa, cae al sample aleatorio válido
                if isinstance(env.action_space, gym.spaces.Discrete) and not env.action_space.contains(a):
                    a = int(env.action_space.sample())
                obs, r, terminated, truncated, _ = env.step(a)
                ep_ret += float(r)
                done = terminated or truncated
                steps += 1
            total += ep_ret
        return total / episodes

    # Baseline random (siempre válido)
    random_mean = run_policy(lambda _: env.action_space.sample())

    # Baselines triviales "siempre acción k" usando el rango correcto [start, start+n)
    always_means: Dict[int, float] = {}
    if isinstance(env.action_space, gym.spaces.Discrete):
        start = int(getattr(env.action_space, "start", 0))
        n = int(env.action_space.n)
        valid_actions = [start + i for i in range(n)]
        for a_fixed in valid_actions:
            always_means[a_fixed] = run_policy(lambda _: a_fixed)

    return float(random_mean), always_means



# -----------------------------
# Policy network
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 20),
            nn.ELU(),
            nn.Linear(20, 20),
            nn.ELU(),
            nn.Linear(20, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


# -----------------------------
# Episode buffer
# -----------------------------
@dataclass
class Episode:
    obs: List[torch.Tensor]
    acts: List[int]
    rewards: List[float]

    def to_batch(self, gamma: float, normalize_returns: bool = True):
        obs_b = torch.stack(self.obs, dim=0)
        acts_b = torch.tensor(self.acts, dtype=torch.int64)
        returns = compute_discounted_returns(self.rewards, gamma)
        if normalize_returns and len(self.rewards) > 1:
            returns = normalize(returns)
        return obs_b, acts_b, returns


# -----------------------------
# REINFORCE Agent
# -----------------------------
class REINFORCEAgent:
    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.1,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        logger: Optional[TensorBoardLogger] = None,
    ) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete), "Acciones Discrete requeridas."
        self.env = env
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.device = torch.device(device) if device else torch.device("cpu")
        self.logger = logger

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.obs_dim = infer_obs_dim(env.observation_space)
        self.act_dim = int(env.action_space.n)

        self.policy = PolicyNet(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs: Any) -> Tuple[int, torch.Tensor]:
        x = preprocess_obs(obs, self.env.observation_space).to(self.device)
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    def run_episode(self, max_steps: Optional[int] = None) -> Episode:
        obs, _ = self.env.reset()
        done = False
        steps = 0
        ep = Episode(obs=[], acts=[], rewards=[])

        while not done:
            a, logp = self.select_action(obs)
            next_obs, r, terminated, truncated, _ = self.env.step(a)

            ep.obs.append(preprocess_obs(obs, self.env.observation_space))
            ep.acts.append(a)
            ep.rewards.append(float(r))

            # guardo temporalmente logp dentro del tensor de obs via attr (evita estructura paralela)
            ep.obs[-1].logp = logp  # monkey-patch: sólo para mantenerlo asociado

            obs = next_obs
            steps += 1
            done = terminated or truncated
            if max_steps is not None and steps >= max_steps:
                break

        return ep

    def update(self, episodes: List[Episode], training_step: int = 0) -> Dict[str, float]:
        """
        Actualiza la política usando REINFORCE vanilla.
        
        CAMBIO IMPORTANTE: La loss ahora se normaliza por episodio en lugar de por paso.
        Esto da valores más interpretables y consistentes entre entrenamiento y logging.
        """
        self.optimizer.zero_grad()
        total_steps = sum(len(ep.rewards) for ep in episodes)
        policy_loss_acc = torch.tensor(0.0, device=self.device)

        # (Opcional: asegurá vanilla puro)
        use_entropy = (self.entropy_coeff is not None) and (self.entropy_coeff > 0.0)
        # Si querés vanilla estricto, forzá:
        # use_entropy = False

        entropy_acc = torch.tensor(0.0, device=self.device)

        for ep in episodes:
            obs_b, acts_b, returns_b = ep.to_batch(self.gamma, normalize_returns=True) #normalizo por episodio no por batch, podria hacerlo a niver batch
            obs_b = obs_b.to(self.device)
            acts_b = acts_b.to(self.device)
            returns_b = returns_b.to(self.device)

            logits = self.policy(obs_b)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(acts_b)

            policy_loss_acc = policy_loss_acc - (log_probs * returns_b).sum()

            if use_entropy:
                entropy_acc = entropy_acc + dist.entropy().sum()

        # Promediar por cantidad total de pasos del batch (match con 1/N “efectivo”)
        # Calcular loss promedio por episodio (no por paso individual)
        # Esto es más intuitivo: la loss representa el costo promedio por episodio
        # en lugar de por paso individual, lo que da valores más interpretables
        num_episodes = len(episodes)
        loss = policy_loss_acc / max(1, num_episodes)
        if use_entropy:
            loss = loss - self.entropy_coeff * (entropy_acc / max(1, num_episodes))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # No se necesita logging aquí ya que se hace por episodio en el método train

        return {
            "loss": float(loss.item()),  # Retornar la loss normalizada por episodio
            "entropy_loss": float(((-entropy_acc) / max(1, num_episodes)).item()) if use_entropy else 0.0,
            "steps": total_steps,
            "episodes": num_episodes,
            "baseline": 0.0,
        }


    def train(
        self,
        num_episodes: int,
        batch_size: int = 1,
        max_steps_per_episode: Optional[int] = None,
        render_every: int = 0,
        verbose_every: int = 10,
        render_human_every: int = 0,
    ) -> Dict[str, List[float]]:
        history = {"ep_return": [], "loss": []}
        buffer: List[Episode] = []
        training_step = 0
        try:
            rnd_mean, always_means = estimate_random_and_trivial_returns(self.env, episodes=200, max_steps=max_steps_per_episode, seed=0)
        except Exception:
            # si el env no soporta múltiples resets seguidos o no es Discrete, lo omitimos silenciosamente
            rnd_mean, always_means = None, None
        for ep_idx in range(1, num_episodes + 1):
            ep = self.run_episode(max_steps=max_steps_per_episode)
            buffer.append(ep)
            ep_return = float(sum(ep.rewards))
            ep_length = len(ep.rewards)
            history["ep_return"].append(ep_return)
            
            # Log episodio individual
            if self.logger is not None:
                # Log todas las métricas de returns juntas en un solo gráfico
                if rnd_mean is not None and always_means is not None:
                    avg_return_10 = 0.0
                    if len(history["ep_return"]) >= 10:
                        avg_return_10 = np.mean(history["ep_return"][-10:])
                    
                    self.logger.log_all_returns_together(
                        ep_idx, ep_return, avg_return_10, rnd_mean, always_means
                    )
                else:
                    # Fallback si no hay baselines
                    self.logger.log_episode(ep_idx, ep_return)
                    if len(history["ep_return"]) >= 10:
                        avg_return_10 = np.mean(history["ep_return"][-10:])
                        self.logger.log_episode_averages(ep_idx, avg_return_10)

            if render_every and (ep_idx % render_every == 0):
                try:
                    self.env.render()
                except Exception:
                    pass

            if len(buffer) >= batch_size:
                training_step += 1
                stats = self.update(buffer, training_step)
                history["loss"].append(stats["loss"])
                
                # Log loss por episodio (el último episodio del batch)
                if self.logger is not None:
                    last_episode_in_batch = ep_idx
                    self.logger.log_episode_loss(last_episode_in_batch, stats["loss"])
                
                buffer.clear()

            if verbose_every and (ep_idx % verbose_every == 0):
                last_loss = history["loss"][-1] if len(history["loss"]) > 0 else float("nan")
                avg_return = np.mean(history["ep_return"][-10:]) if len(history["ep_return"]) >= 10 else np.mean(history["ep_return"])
                # Mostrar información más detallada sobre la loss
                loss_info = f"Loss: {last_loss:.3f}"
                if not np.isnan(last_loss):
                    loss_info += f" (episodio avg)"  # Indicar que es promedio por episodio
                print(f"[{ep_idx}/{num_episodes}] Return: {ep_return:.3f} | Avg Return(10): {avg_return:.3f} | {loss_info}")

        if len(buffer) > 0:
            training_step += 1
            stats = self.update(buffer, training_step)
            history["loss"].append(stats["loss"])

        return history


def train_agent(
    agent: REINFORCEAgent,
    num_episodes: int = 500,
    batch_size: int = 1,
    max_steps_per_episode: Optional[int] = None,
    render_every: int = 0,
    verbose_every: int = 10,
) -> Dict[str, List[float]]:
    return agent.train(
        num_episodes=num_episodes,
        batch_size=batch_size,
        max_steps_per_episode=max_steps_per_episode,
        render_every=render_every,
        verbose_every=verbose_every,
    )


def demo_trained_policy(agent: REINFORCEAgent, env: gym.Env, num_demos: int = 3) -> None:
    """Demostrar la política entrenada con visualización."""
    print(f"\n🎮 DEMO: Mostrando política entrenada para {env.__class__.__name__}")
    print("=" * 60)
    
    for demo_idx in range(num_demos):
        print(f"\n--- Demo {demo_idx + 1}/{num_demos} ---")
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        
        # Render inicial
        env.render()
        
        while not done and step < 20:  # Límite de seguridad
            # Seleccionar acción (sin exploración, usar acción más probable)
            x = preprocess_obs(obs, env.observation_space).to(agent.device)
            with torch.no_grad():
                logits = agent.policy(x)
                action_probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(action_probs).item()
            
            print(f"  Paso {step + 1}: obs={obs}, acción={action}, probs={action_probs.numpy()}")
            
            # Ejecutar acción
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            # Render después del paso
            env.render()
            
            # Pausa para mejor visualización
            import time
            time.sleep(0.5)
        
        print(f"  ✅ Demo terminada: {step} pasos, recompensa total: {total_reward}")
        
        if demo_idx < num_demos - 1:
            print("  (Esperando 1 segundo antes del siguiente demo...)")
            import time
            time.sleep(1.0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    import time
    
    # Crear directorio para logs de TensorBoard
    log_dir = "tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Solo entrenar en CartPole-v1
    print("🎮 Entrenando REINFORCE en CartPole-v1")
    print("="*50)
    
    # Crear entorno CartPole-v1
    env = gym.make("CartPole-v1")
    env_name = "CartPole-v1"
    
    # Hiperparámetros para CartPole
    lr, entropy_coeff, batch_size = 3e-3, 0.01, 5
    num_episodes = 5000  # Aumentado a 5000 episodios
    
    print(f"📊 Configuración:")
    print(f"   Episodios: {num_episodes}")
    print(f"   Learning Rate: {lr}")
    print(f"   Entropy Coeff: {entropy_coeff}")
    print(f"   Batch Size: {batch_size}")
    print()
    
    # Crear logger de TensorBoard
    logger = TensorBoardLogger(log_dir, env_name)
    
    # Medir tiempo de wall clock
    start_time = time.time()
    wall_times = []
    
    # Crear agente
    agent = REINFORCEAgent(env, lr=lr, gamma=0.99, entropy_coeff=entropy_coeff, seed=0, logger=logger)
    
    # Entrenar agente
    print("🚀 Iniciando entrenamiento...")
    history = train_agent(agent, num_episodes=num_episodes, batch_size=batch_size, verbose_every=50)
    
    # Calcular wall times para cada episodio
    for i in range(len(history['ep_return'])):
        wall_times.append(time.time() - start_time)
    
    # Cerrar logger
    logger.close()
    
    # Calcular estadísticas finales
    final_avg = np.mean(history['ep_return'][-10:])
    final_avg_100 = np.mean(history['ep_return'][-100:]) if len(history['ep_return']) >= 100 else final_avg
    
    print(f"\n📈 RESULTADOS FINALES:")
    print(f"   Últimos 10 episodios promedio: {final_avg:.3f}")
    print(f"   Últimos 100 episodios promedio: {final_avg_100:.3f}")
    print(f"   Tiempo total de entrenamiento: {wall_times[-1]:.1f} segundos")
    print(f"   Threshold CartPole (195): {'✅ ALCANZADO' if final_avg >= 195 else '❌ NO ALCANZADO'}")
    
    # Crear gráficos
    print(f"\n📊 Generando gráficos...")
    
    # 1. Gráfico de recompensa vs episodios
    plt.figure(figsize=(12, 8))
    
    # Recompensas por episodio
    episodes = list(range(1, len(history['ep_return']) + 1))
    plt.plot(episodes, history['ep_return'], alpha=0.3, color='lightblue', label='Recompensa por Episodio')
    
    # Promedio móvil de 10 episodios
    if len(history['ep_return']) >= 10:
        moving_avg_10 = []
        for i in range(len(history['ep_return'])):
            start_idx = max(0, i - 9)
            moving_avg_10.append(np.mean(history['ep_return'][start_idx:i+1]))
        plt.plot(episodes, moving_avg_10, color='blue', linewidth=2, label='Promedio Móvil (10 episodios)')
    
    # Promedio móvil de 100 episodios
    if len(history['ep_return']) >= 100:
        moving_avg_100 = []
        for i in range(len(history['ep_return'])):
            start_idx = max(0, i - 99)
            moving_avg_100.append(np.mean(history['ep_return'][start_idx:i+1]))
        plt.plot(episodes, moving_avg_100, color='red', linewidth=2, label='Promedio Móvil (100 episodios)')
    
    # Línea de referencia
    plt.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Threshold "Resuelto" (195)')
    
    plt.xlabel('Episodios', fontsize=12)
    plt.ylabel('Recompensa', fontsize=12)
    plt.title('REINFORCE en CartPole-v1 - Recompensa vs Episodios', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "reinforce_cartpole_episodes.png", dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: plots/reinforce_cartpole_episodes.png")
    plt.show()
    
    # 2. Gráfico de recompensa vs wall time
    plt.figure(figsize=(12, 8))
    
    # Convertir wall times a minutos
    wall_times_minutes = np.array(wall_times) / 60
    
    # Recompensas por episodio vs tiempo
    plt.plot(wall_times_minutes, history['ep_return'], alpha=0.3, color='lightblue', label='Recompensa por Episodio')
    
    # Promedio móvil de 10 episodios vs tiempo
    if len(history['ep_return']) >= 10:
        plt.plot(wall_times_minutes, moving_avg_10, color='blue', linewidth=2, label='Promedio Móvil (10 episodios)')
    
    # Promedio móvil de 100 episodios vs tiempo
    if len(history['ep_return']) >= 100:
        plt.plot(wall_times_minutes, moving_avg_100, color='red', linewidth=2, label='Promedio Móvil (100 episodios)')
    
    # Línea de referencia
    plt.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Threshold "Resuelto" (195)')
    
    plt.xlabel('Tiempo de Entrenamiento (minutos)', fontsize=12)
    plt.ylabel('Recompensa', fontsize=12)
    plt.title('REINFORCE en CartPole-v1 - Recompensa vs Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plt.savefig(plots_dir / "reinforce_cartpole_walltime.png", dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: plots/reinforce_cartpole_walltime.png")
    plt.show()
    
    # 3. Gráfico de loss vs episodios
    if len(history['loss']) > 0:
        plt.figure(figsize=(12, 6))
        
        # Loss por episodio (cada batch_size episodios)
        loss_episodes = list(range(batch_size, len(history['loss']) * batch_size + 1, batch_size))
        if len(loss_episodes) > len(history['loss']):
            loss_episodes = loss_episodes[:len(history['loss'])]
        
        plt.plot(loss_episodes, history['loss'], color='red', linewidth=2, label='Policy Loss')
        plt.xlabel('Episodios', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('REINFORCE en CartPole-v1 - Policy Loss vs Episodios', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.savefig(plots_dir / "reinforce_cartpole_loss.png", dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: plots/reinforce_cartpole_loss.png")
        plt.show()
    
    # Cerrar entorno
    env.close()
    
    print(f"\n🎉 Entrenamiento completado!")
    print(f"📊 Para ver los logs en TensorBoard:")
    print(f"   tensorboard --logdir {log_dir}")
    print(f"   Luego abre: http://localhost:6006")

