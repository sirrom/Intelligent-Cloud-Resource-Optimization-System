# ==============================================================================  
# ML CAPSTONE PROJECT  
# STAGE 4 — Reinforcement Learning for Cloud Resource Optimisation  
# STAGE 5 — Final Dashboard & Report  
# ==============================================================================  
# Dependencies:  
#   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow  
# ==============================================================================  

import os  
import json  
import random  
import warnings  
from collections import deque, defaultdict  

warnings.filterwarnings('ignore')  

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
import matplotlib.patches as mpatches  
import seaborn as sns  

import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras import layers  

from sklearn.preprocessing   import StandardScaler, MinMaxScaler  
from sklearn.decomposition   import PCA  
from sklearn.cluster         import KMeans  
from sklearn.metrics         import silhouette_score  

# ── Reproducibility ───────────────────────────────────────────────────────────  
SEED = 42  
np.random.seed(SEED)  
random.seed(SEED)  
tf.random.set_seed(SEED)  

# ── Global colour palette ─────────────────────────────────────────────────────  
C = {  
    'blue':    '#3498DB',  
    'red':     '#E74C3C',  
    'green':   '#2ECC71',  
    'orange':  '#F39C12',  
    'purple':  '#9B59B6',  
    'teal':    '#1ABC9C',  
    'dark':    '#2C3E50',  
    'grey':    '#95A5A6',  
    'yellow':  '#F1C40F',  
}  

ACTION_PALETTE = [C['blue'], C['green'], C['red'], C['orange']]  


# ==============================================================================  
# STAGE 4.1 — CLOUD COMPUTING ENVIRONMENT  (Custom MDP)  
# ==============================================================================  

class CloudComputingEnv:  
    """  
    Custom Reinforcement Learning Environment — Cloud Task Scheduler MDP.  

    ┌──────────────────────────────────────────────────────────────────────┐  
    │  STATE  (6-dimensional continuous vector, normalised [0, 1])         │  
    │    s[0]  cpu_usage            s[3]  power_consumption                │  
    │    s[1]  memory_usage         s[4]  execution_time                   │  
    │    s[2]  network_traffic      s[5]  energy_efficiency                │  
    ├──────────────────────────────────────────────────────────────────────┤  
    │  ACTIONS  (4 discrete)                                               │  
    │    0 → Scale DOWN    1 → HOLD    2 → Scale UP    3 → MIGRATE         │  
    ├──────────────────────────────────────────────────────────────────────┤  
    │  REWARD  (shaped, domain-driven)                                     │  
    │    +2.0   correct resource adjustment (scale-down/up when needed)    │  
    │    +1.0   balanced CPU utilisation (HOLD in sweet spot)              │  
    │    +0.5   energy efficiency bonus                                    │  
    │    -1.5   SLA violation  (execution_time > SLA_THRESHOLD)            │  
    │    -1.5   dangerous scale-down under heavy load                      │  
    │    -0.5   wasteful over-provisioning                                 │  
    │    -0.1   per-step cost  (encourages decisive behaviour)             │  
    └──────────────────────────────────────────────────────────────────────┘  

    The environment streams through the real cloud dataset row-by-row so  
    the agent faces genuinely varied workload patterns.  
    """  

    N_ACTIONS     = 4  
    N_BINS        = 5           # discretisation bins per feature (Q-table)  
    MAX_STEPS     = 200         # maximum steps per episode  
    SLA_THRESHOLD = 0.85        # normalised execution-time SLA limit  

    ACTION_NAMES  = {  
        0: 'Scale DOWN',  
        1: 'HOLD',  
        2: 'Scale UP',  
        3: 'MIGRATE',  
    }  

    # ── Construction ──────────────────────────────────────────────────────────  
    def __init__(self, data_array: np.ndarray, seed: int = SEED):  
        """  
        Parameters  
        ----------  
        data_array : np.ndarray  shape (N, 6)  
            Raw feature matrix from the cloud dataset.  
            Columns expected: cpu_usage, memory_usage, network_traffic,  
                              power_consumption, execution_time,  
                              energy_efficiency.  
        """  
        np.random.seed(seed)  
        random.seed(seed)  

        self.raw_data  = data_array.astype(np.float32)  
        self.n_samples = len(data_array)  

        # ── Min-max normalise every column to [0, 1] ──────────────────────  
        col_min = self.raw_data.min(axis=0)  
        col_max = self.raw_data.max(axis=0)  
        col_rng = col_max - col_min  
        col_rng[col_rng == 0] = 1.0          # avoid divide-by-zero  
        self.norm_data = (self.raw_data - col_min) / col_rng  

        self.state_size  = data_array.shape[1]   # 6  
        self.action_size = self.N_ACTIONS  

        self._reset_internals()  

    # ── Public API ────────────────────────────────────────────────────────────  
    def reset(self) -> np.ndarray:  
        """Reset environment to a random starting state. Returns initial state."""  
        self._reset_internals()  
        return self._cont_state()  

    def step(self, action: int):  
        """  
        Execute one step.  

        Returns  
        -------  
        next_state : np.ndarray  
        reward     : float  
        done       : bool  
        info       : dict  
        """  
        if not (0 <= action < self.N_ACTIONS):  
            raise ValueError(f"Invalid action: {action}. "  
                             f"Must be in [0, {self.N_ACTIONS - 1}]")  

        state  = self._cont_state()  
        reward = self._shaped_reward(action, state)  

        # advance data pointer  
        self.ptr        = (self.ptr + 1) % self.n_samples  
        self.step_count += 1  
        self.cum_reward += reward  

        next_state = self._cont_state()  
        done       = (self.step_count >= self.MAX_STEPS)  

        self.episode_log['rewards'].append(reward)  
        self.episode_log['actions'].append(action)  
        self.episode_log['states'].append(state.tolist())  

        return next_state, reward, done, {  
            'step':        self.step_count,  
            'cum_reward':  self.cum_reward,  
            'action_name': self.ACTION_NAMES[action],  
        }  

    def render(self):  
        s = self._cont_state()  
        print(f"  step={self.step_count:>4} | "  
              f"cpu={s[0]:.3f}  mem={s[1]:.3f}  "  
              f"net={s[2]:.3f}  pwr={s[3]:.3f}  "  
              f"etime={s[4]:.3f}  eeff={s[5]:.3f} | "  
              f"cum_r={self.cum_reward:+.3f}")  

    # ── Discrete state (for Q-table) ──────────────────────────────────────────  
    def discrete_state(self) -> tuple:  
        """Bin each dimension of the current normalised state."""  
        s    = self._cont_state()  
        bins = np.linspace(0, 1, self.N_BINS + 1)[1:-1]   # N_BINS-1 boundaries  
        return tuple(int(np.digitize(float(v), bins)) for v in s)  

    # ── Internals ─────────────────────────────────────────────────────────────  
    def _reset_internals(self):  
        self.ptr        = random.randint(0, self.n_samples - 1)  
        self.step_count = 0  
        self.cum_reward = 0.0  
        self.episode_log = defaultdict(list)  

    def _cont_state(self) -> np.ndarray:  
        return self.norm_data[self.ptr].copy()  

    def _shaped_reward(self, action: int,  
                        state:  np.ndarray) -> float:  
        """  
        Domain-specific reward function.  
        Columns: [cpu, mem, net, power, exec_time, energy_eff]  
        """  
        cpu   = float(state[0])  
        mem   = float(state[1])  
        pwr   = float(state[3])  
        etime = float(state[4]) if len(state) > 4 else cpu  
        eeff  = float(state[5]) if len(state) > 5 else (1.0 - pwr)  

        r = -0.10   # per-step cost  

        # ── Action-specific shaping ────────────────────────────────────────  
        if action == 0:          # Scale DOWN  
            if cpu < 0.30 and mem < 0.30:  
                r += 2.00        # resources genuinely under-used  
            elif cpu > 0.70:  
                r -= 1.50        # dangerous — overloaded system  
            else:  
                r += 0.30  

        elif action == 1:        # HOLD  
            if 0.30 <= cpu <= 0.70:  
                r += 1.00        # balanced utilisation  
            else:  
                r -= 0.30  

        elif action == 2:        # Scale UP  
            if cpu > 0.75 or mem > 0.75:  
                r += 2.00        # prevented impending overload  
            elif cpu < 0.30:  
                r -= 0.50        # wasteful over-provision  
            else:  
                r += 0.40  

        elif action == 3:        # MIGRATE  
            if pwr > 0.70:  
                r += 1.80        # justified — power is high  
            else:  
                r -= 0.50        # unnecessary migration cost  

        # ── SLA penalty ───────────────────────────────────────────────────  
        if etime > self.SLA_THRESHOLD:  
            r -= 1.50  

        # ── Energy efficiency bonus ───────────────────────────────────────  
        if eeff > 0.70:  
            r += 0.50  

        return float(r)  


# ==============================================================================  
# STAGE 4.2 — TABULAR Q-LEARNING AGENT  
# ==============================================================================  

class QLearningAgent:  
    """  
    Classic tabular Q-Learning with ε-greedy exploration.  

    Update rule:  
        Q(s,a) ← Q(s,a) + α · [ r + γ · max_{a'} Q(s',a') − Q(s,a) ]  

    Parameters  
    ----------  
    n_actions  : number of discrete actions  
    alpha      : learning rate  
    gamma      : discount factor  
    epsilon    : initial exploration probability  
    eps_min    : minimum epsilon (after full decay)  
    eps_decay  : multiplicative decay applied after each episode  
    """  

    def __init__(self,  
                 n_actions:  int,  
                 alpha:      float = 0.10,  
                 gamma:      float = 0.95,  
                 epsilon:    float = 1.00,  
                 eps_min:    float = 0.05,  
                 eps_decay:  float = 0.997):  

        self.n_actions = n_actions  
        self.alpha     = alpha  
        self.gamma     = gamma  
        self.epsilon   = epsilon  
        self.eps_min   = eps_min  
        self.eps_decay = eps_decay  

        # Q-table: maps discrete state tuple → Q-value array  
        self.Q = defaultdict(lambda: np.zeros(n_actions,  
                                               dtype=np.float64))  

        # Training history  
        self.log = {  
            'ep_rewards':  [],  
            'ep_steps':    [],  
            'epsilon':     [],  
            'td_errors':   [],  
            'q_table_sz':  [],  
        }  

    # ── Action selection ──────────────────────────────────────────────────────  
    def act(self, state: tuple) -> int:  
        """ε-greedy policy over discrete state."""  
        if random.random() < self.epsilon:  
            return random.randrange(self.n_actions)  
        return int(np.argmax(self.Q[state]))  

    # ── Q-update ──────────────────────────────────────────────────────────────  
    def update(self, s, a, r, s_next, done) -> float:  
        """Apply one Q-Learning update and return |TD error|."""  
        q_next  = 0.0 if done else float(np.max(self.Q[s_next]))  
        td_err  = r + self.gamma * q_next - self.Q[s][a]  
        self.Q[s][a] += self.alpha * td_err  
        return abs(td_err)  

    # ── Epsilon decay ─────────────────────────────────────────────────────────  
    def decay_epsilon(self):  
        self.epsilon = max(self.eps_min,  
                           self.epsilon * self.eps_decay)  

    # ── Training loop ─────────────────────────────────────────────────────────  
    def train(self,  
              env:           CloudComputingEnv,  
              n_episodes:    int = 600,  
              verbose_every: int = 100) -> dict:  
        """  
        Train the Q-Learning agent for n_episodes episodes.  

        Returns  
        -------  
        dict  Training log with rewards, TD errors, epsilon, Q-table size.  
        """  
        print("\n" + "=" * 70)  
        print("  STAGE 4.2 — Tabular Q-Learning Agent")  
        print(f"  Episodes={n_episodes}  α={self.alpha}  γ={self.gamma}  "  
              f"ε₀={self.epsilon}  ε_min={self.eps_min}  "  
              f"ε_decay={self.eps_decay}")  
        print("=" * 70)  

        for ep in range(n_episodes):  
            env.reset()  
            ep_r  = 0.0  
            ep_td = []  

            for _ in range(env.MAX_STEPS):  
                s      = env.discrete_state()  
                a      = self.act(s)  
                _, r, done, _ = env.step(a)  
                s2     = env.discrete_state()  

                td = self.update(s, a, r, s2, done)  
                ep_td.append(td)  
                ep_r += r  
                if done:  
                    break  

            self.decay_epsilon()  
            self.log['ep_rewards'].append(ep_r)  
            self.log['ep_steps'].append(env.step_count)  
            self.log['epsilon'].append(self.epsilon)  
            self.log['td_errors'].append(float(np.mean(ep_td)))  
            self.log['q_table_sz'].append(len(self.Q))  

            if (ep + 1) % verbose_every == 0:  
                avg_r  = np.mean(self.log['ep_rewards'][-100:])  
                avg_td = np.mean(self.log['td_errors'][-100:])  
                print(f"  Ep {ep+1:>5}/{n_episodes}  │  "  
                      f"AvgReward(100)={avg_r:>9.3f}  │  "  
                      f"AvgTD={avg_td:.5f}  │  "  
                      f"ε={self.epsilon:.5f}  │  "  
                      f"Q-entries={len(self.Q):>7,}")  

        best = max(self.log['ep_rewards'])  
        print(f"\n  ✅ Q-Learning done  │  "  
              f"Best ep reward={best:.3f}  │  "  
              f"Final Q-table size={len(self.Q):,}")  
        return self.log  

    # ── Policy inspection ─────────────────────────────────────────────────────  
    def greedy_action_distribution(self) -> np.ndarray:  
        """  
        Fraction of Q-table states where each action is the greedy best.  
        """  
        counts = np.zeros(self.n_actions, dtype=int)  
        for q_vals in self.Q.values():  
            counts[int(np.argmax(q_vals))] += 1  
        total = counts.sum()  
        return counts / total if total > 0 else counts.astype(float)  

    # ── Greedy evaluation ─────────────────────────────────────────────────────  
    def evaluate(self,  
                 env:    CloudComputingEnv,  
                 n_eval: int = 50) -> dict:  
        """Run n_eval greedy episodes (ε=0). Returns evaluation metrics."""  
        saved_eps   = self.epsilon  
        self.epsilon = 0.0  

        rewards     = []  
        all_actions = []  
        sla_hits    = 0  
        total_steps = 0  

        for _ in range(n_eval):  
            env.reset()  
            ep_r = 0.0  
            for _ in range(env.MAX_STEPS):  
                a = self.act(env.discrete_state())  
                s2, r, done, _ = env.step(a)  
                ep_r        += r  
                total_steps += 1  
                all_actions.append(a)  
                if s2[4] > env.SLA_THRESHOLD:  
                    sla_hits += 1  
                if done:  
                    break  
            rewards.append(ep_r)  

        self.epsilon = saved_eps  
        ac = np.bincount(all_actions,  
                          minlength=env.N_ACTIONS).astype(float)  
        return {  
            'rewards':       rewards,  
            'mean_reward':   float(np.mean(rewards)),  
            'std_reward':    float(np.std(rewards)),  
            'max_reward':    float(np.max(rewards)),  
            'min_reward':    float(np.min(rewards)),  
            'action_counts': ac,  
            'action_dist':   ac / ac.sum(),  
            'sla_rate':      sla_hits / max(total_steps, 1),  
        }  


# ==============================================================================  
# STAGE 4.3 — DEEP Q-NETWORK (DQN) AGENT  
# ==============================================================================  

class DQNAgent:  
    """  
    Deep Q-Network with:  
      • Experience Replay buffer (deque with max capacity)  
      • Target Network           (periodically synced from online net)  
      • ε-greedy exploration     (exponential decay)  
      • Huber loss               (robust to outlier TD targets)  

    Architecture:  
        Input(state_size) → Dense(256, ReLU) → BatchNorm  
                          → Dense(128, ReLU) → Dropout(0.1)  
                          → Dense(64,  ReLU)  
                          → Dense(action_size, Linear)  [Q-values]  
    """  

    def __init__(self,  
                 state_size:   int,  
                 action_size:  int,  
                 lr:           float = 1e-3,  
                 gamma:        float = 0.95,  
                 epsilon:      float = 1.00,  
                 eps_min:      float = 0.05,  
                 eps_decay:    float = 0.997,  
                 batch_size:   int   = 64,  
                 memory_cap:   int   = 20_000,  
                 target_sync:  int   = 10):  

        self.state_size  = state_size  
        self.action_size = action_size  
        self.gamma       = gamma  
        self.epsilon     = epsilon  
        self.eps_min     = eps_min  
        self.eps_decay   = eps_decay  
        self.batch_size  = batch_size  
        self.target_sync = target_sync  
        self.sync_ctr    = 0  

        # Replay buffer  
        self.memory      = deque(maxlen=memory_cap)  

        # Online and target networks  
        self.online_net  = self._build_network(lr, name='Online_DQN')  
        self.target_net  = self._build_network(lr, name='Target_DQN')  
        self._sync_target()  

        # Training history  
        self.log = {  
            'ep_rewards': [],  
            'ep_steps':   [],  
            'epsilon':    [],  
            'losses':     [],  
        }  

    # ── Network builder ───────────────────────────────────────────────────────  
    def _build_network(self, lr: float,  
                        name: str = 'DQN') -> keras.Model:  
        """Build Q-network mapping state vector → Q-values."""  
        inp = keras.Input(shape=(self.state_size,), name='state_input')  
        x   = layers.Dense(256, activation='relu',  
                             kernel_initializer='he_uniform')(inp)  
        x   = layers.BatchNormalization()(x)  
        x   = layers.Dense(128, activation='relu',  
                             kernel_initializer='he_uniform')(x)  
        x   = layers.Dropout(0.10)(x)  
        x   = layers.Dense(64, activation='relu',  
                             kernel_initializer='he_uniform')(x)  
        out = layers.Dense(self.action_size, activation='linear',  
                            name='q_output')(x)  

        model = keras.Model(inputs=inp, outputs=out, name=name)  
        model.compile(  
            optimizer=keras.optimizers.Adam(learning_rate=lr,  
                                             clipnorm=1.0),  
            loss=tf.keras.losses.Huber(delta=1.0),  
        )  
        return model  

    # ── Target sync ───────────────────────────────────────────────────────────  
    def _sync_target(self):  
        """Hard-copy weights from online → target network."""  
        self.target_net.set_weights(self.online_net.get_weights())  

    # ── Replay buffer ─────────────────────────────────────────────────────────  
    def remember(self, s, a, r, s2, done):  
        self.memory.append((  
            s.astype(np.float32),  
            int(a),  
            float(r),  
            s2.astype(np.float32),  
            bool(done),  
        ))  

    # ── Action selection ──────────────────────────────────────────────────────  
    def act(self, state: np.ndarray) -> int:  
        """ε-greedy action selection over continuous state."""  
        if random.random() < self.epsilon:  
            return random.randrange(self.action_size)  
        q = self.online_net(  
            state[np.newaxis].astype(np.float32),  
            training=False).numpy()[0]  
        return int(np.argmax(q))  

    # ── Minibatch update ──────────────────────────────────────────────────────  
    def replay(self) -> float:  
        """Sample minibatch from replay buffer and update online network."""  
        if len(self.memory) < self.batch_size:  
            return 0.0  

        batch   = random.sample(self.memory, self.batch_size)  
        states  = np.array([t[0] for t in batch], np.float32)  
        actions = np.array([t[1] for t in batch], np.int32)  
        rewards = np.array([t[2] for t in batch], np.float32)  
        nexts   = np.array([t[3] for t in batch], np.float32)  
        dones   = np.array([t[4] for t in batch], np.float32)  

        # Fixed Q-targets using target network  
        q_next   = self.target_net(nexts, training=False).numpy()  
        max_next = q_next.max(axis=1)  
        targets  = rewards + self.gamma * max_next * (1.0 - dones)  

        # Update only the chosen action's Q-value  
        q_curr = self.online_net(states, training=False).numpy()  
        q_curr[np.arange(self.batch_size), actions] = targets  

        hist = self.online_net.fit(  
            states, q_curr,  
            epochs=1, batch_size=self.batch_size,  
            verbose=0,  
        )  
        return float(hist.history['loss'][0])  

    # ── Epsilon decay ─────────────────────────────────────────────────────────  
    def decay_epsilon(self):  
        self.epsilon = max(self.eps_min,  
                           self.epsilon * self.eps_decay)  

    # ── Training loop ─────────────────────────────────────────────────────────  
    def train(self,  
              env:           CloudComputingEnv,  
              n_episodes:    int = 500,  
              verbose_every: int = 50) -> dict:  
        """  
        Train the DQN agent for n_episodes episodes.  

        Returns  
        -------  
        dict  Training log with rewards, losses, epsilon.  
        """  
        print("\n" + "=" * 70)  
        print("  STAGE 4.3 — Deep Q-Network (DQN) Agent")  
        print(f"  Episodes={n_episodes}  γ={self.gamma}  "  
              f"batch={self.batch_size}  "  
              f"target_sync_every={self.target_sync}  "  
              f"memory_cap={self.memory.maxlen:,}")  
        print("=" * 70)  

        for ep in range(n_episodes):  
            state  = env.reset()  
            ep_r   = 0.0  
            ep_ls  = []  
            steps  = 0  

            for _ in range(env.MAX_STEPS):  
                action             = self.act(state)  
                next_s, r, done, _ = env.step(action)  
                self.remember(state, action, r, next_s, done)  
                loss = self.replay()  
                if loss > 0:  
                    ep_ls.append(loss)  
                state  = next_s  
                ep_r  += r  
                steps += 1  
                if done:  
                    break  

            # Periodic target sync  
            self.sync_ctr += 1  
            if self.sync_ctr % self.target_sync == 0:  
                self._sync_target()  

            self.decay_epsilon()  
            self.log['ep_rewards'].append(ep_r)  
            self.log['ep_steps'].append(steps)  
            self.log['epsilon'].append(self.epsilon)  
            self.log['losses'].append(  
                float(np.mean(ep_ls)) if ep_ls else 0.0)  

            if (ep + 1) % verbose_every == 0:  
                avg_r = np.mean(self.log['ep_rewards'][-100:])  
                avg_l = np.mean([x for x in  
                                  self.log['losses'][-100:] if x > 0]  
                                 or [0])  
                print(f"  Ep {ep+1:>5}/{n_episodes}  │  "  
                      f"AvgReward(100)={avg_r:>9.3f}  │  "  
                      f"AvgLoss={avg_l:.6f}  │  "  
                      f"ε={self.epsilon:.5f}  │  "  
                      f"Buffer={len(self.memory):>7,}")  

        best = max(self.log['ep_rewards'])  
        print(f"\n  ✅ DQN done  │  Best ep reward={best:.3f}")  
        return self.log  

    # ── Inference helpers ─────────────────────────────────────────────────────  
    def q_values(self, state: np.ndarray) -> np.ndarray:  
        return self.online_net(  
            state[np.newaxis].astype(np.float32),  
            training=False).numpy()[0]  

    def predict_actions(self, states: np.ndarray) -> np.ndarray:  
        q = self.online_net.predict(  
            states.astype(np.float32), verbose=0)  
        return np.argmax(q, axis=1)  

    # ── Greedy evaluation ─────────────────────────────────────────────────────  
    def evaluate(self,  
                 env:    CloudComputingEnv,  
                 n_eval: int = 50) -> dict:  
        """Run n_eval greedy episodes (ε=0). Returns evaluation metrics."""  
        saved_eps   = self.epsilon  
        self.epsilon = 0.0  

        rewards     = []  
        all_actions = []  
        sla_hits    = 0  
        total_steps = 0  

        for _ in range(n_eval):  
            state = env.reset()  
            ep_r  = 0.0  
            for _ in range(env.MAX_STEPS):  
                a = self.act(state)  
                state, r, done, _ = env.step(a)  
                ep_r        += r  
                total_steps += 1  
                all_actions.append(a)  
                if state[4] > env.SLA_THRESHOLD:  
                    sla_hits += 1  
                if done:  
                    break  
            rewards.append(ep_r)  

        self.epsilon = saved_eps  
        ac = np.bincount(all_actions,  
                          minlength=env.N_ACTIONS).astype(float)  
        return {  
            'rewards':       rewards,  
            'mean_reward':   float(np.mean(rewards)),  
            'std_reward':    float(np.std(rewards)),  
            'max_reward':    float(np.max(rewards)),  
            'min_reward':    float(np.min(rewards)),  
            'action_counts': ac,  
            'action_dist':   ac / ac.sum(),  
            'sla_rate':      sla_hits / max(total_steps, 1),  
        }  


# ==============================================================================  
# STAGE 4.4 — RANDOM BASELINE AGENT  
# ==============================================================================  

class RandomAgent:  
    """Random policy baseline for comparison."""  

    def __init__(self, n_actions: int):  
        self.n_actions = n_actions  
        self.epsilon   = 1.0      # kept for interface compatibility  

    def act(self, state) -> int:  
        return random.randrange(self.n_actions)  

    def evaluate(self,  
                 env:    CloudComputingEnv,  
                 n_eval: int = 50) -> dict:  
        rewards     = []  
        all_actions = []  
        sla_hits    = 0  
        total_steps = 0  

        for _ in range(n_eval):  
            env.reset()  
            ep_r = 0.0  
            for _ in range(env.MAX_STEPS):  
                a = self.act(None)  
                s2, r, done, _ = env.step(a)  
                ep_r        += r  
                total_steps += 1  
                all_actions.append(a)  
                if s2[4] > env.SLA_THRESHOLD:  
                    sla_hits += 1  
                if done:  
                    break  
            rewards.append(ep_r)  

        ac = np.bincount(all_actions,  
                          minlength=env.N_ACTIONS).astype(float)  
        return {  
            'rewards':       rewards,  
            'mean_reward':   float(np.mean(rewards)),  
            'std_reward':    float(np.std(rewards)),  
            'max_reward':    float(np.max(rewards)),  
            'min_reward':    float(np.min(rewards)),  
            'action_counts': ac,  
            'action_dist':   ac / ac.sum(),  
            'sla_rate':      sla_hits / max(total_steps, 1),  
        }  


# ==============================================================================  
# STAGE 4.5 — COMPREHENSIVE RL VISUALISATION  (15-panel dashboard)  
# ==============================================================================  

def smooth(arr, w: int = 20) -> np.ndarray:  
    """Rolling-mean smoothing with min_periods=1."""  
    return pd.Series(arr).rolling(w, min_periods=1).mean().values  


def visualise_rl(ql_log:     dict,  
                  dqn_log:    dict,  
                  ql_eval:    dict,  
                  dqn_eval:   dict,  
                  rand_eval:  dict,  
                  ql_agent:   QLearningAgent,  
                  dqn_agent:  DQNAgent,  
                  env:        CloudComputingEnv,  
                  save_path:  str = 'stage4_rl_dashboard.png') -> None:  
    """  
    Produce a 5×3 dashboard covering:  
      Row 1  — Training reward curves + QL vs DQN comparison  
      Row 2  — Epsilon decay · TD error · DQN Huber loss  
      Row 3  — Greedy action distributions · Q-table growth  
      Row 4  — Evaluation boxplot · SLA bars · DQN Q-value heatmap  
      Row 5  — Cumulative reward · Episode length · Reward improvement  
    """  

    fig = plt.figure(figsize=(24, 32))  
    gs  = gridspec.GridSpec(5, 3, figure=fig,  
                              hspace=0.52, wspace=0.36)  

    # ── helpers ───────────────────────────────────────────────────────────────  
    def _title(ax, txt, fs=11):  
        ax.set_title(txt, fontweight='bold', fontsize=fs)  

    def _reward_panel(ax, rewards, color, label, w=30):  
        raw = np.array(rewards)  
        ax.plot(raw, alpha=0.18, color=color, linewidth=0.7)  
        sm  = smooth(raw, w)  
        ax.plot(sm,  color=color, linewidth=2.4, label=f'{label} (smoothed)')  
        ax.fill_between(range(len(raw)),  
                         sm - raw.std() * 0.25,  
                         sm + raw.std() * 0.25,  
                         alpha=0.12, color=color)  
        last100 = np.mean(raw[-100:])  
        ax.axhline(last100, color=C['green'], linestyle='--',  
                    linewidth=1.6,  
                    label=f'Last-100 mean={last100:.2f}')  
        ax.set_xlabel('Episode')  
        ax.set_ylabel('Total Reward')  
        ax.legend(fontsize=8)  

    # ═══════════════════════════════════════════════════════════════  
    # ROW 1 — Reward curves  
    # ═══════════════════════════════════════════════════════════════  

    # 1-A: Q-Learning reward  
    ax = fig.add_subplot(gs[0, 0])  
    _reward_panel(ax, ql_log['ep_rewards'], C['blue'], 'Q-Learning')  
    _title(ax, 'Q-Learning\nEpisode Total Reward')  

    # 1-B: DQN reward  
    ax = fig.add_subplot(gs[0, 1])  
    _reward_panel(ax, dqn_log['ep_rewards'], C['red'], 'DQN')  
    _title(ax, 'DQN\nEpisode Total Reward')  

    # 1-C: QL vs DQN overlay  
    ax   = fig.add_subplot(gs[0, 2])  
    n_ep = min(len(ql_log['ep_rewards']),  
               len(dqn_log['ep_rewards']))  
    ax.plot(smooth(np.array(ql_log['ep_rewards'])[:n_ep], 30),  
             color=C['blue'], linewidth=2.4, label='Q-Learning')  
    ax.plot(smooth(np.array(dqn_log['ep_rewards'])[:n_ep], 30),  
             color=C['red'],  linewidth=2.4, label='DQN',  
             linestyle='--')  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Smoothed Reward')  
    ax.legend(fontsize=9)  
    _title(ax, 'Q-Learning vs DQN\nSmoothed Reward Comparison')  

    # ═══════════════════════════════════════════════════════════════  
    # ROW 2 — Diagnostics  
    # ═══════════════════════════════════════════════════════════════  

    # 2-A: Epsilon decay (both agents)  
    ax = fig.add_subplot(gs[1, 0])  
    ax.plot(ql_log['epsilon'],  color=C['blue'], linewidth=2.2,  
             label='Q-Learning ε')  
    ax.plot(dqn_log['epsilon'], color=C['red'],  linewidth=2.2,  
             label='DQN ε', linestyle='--')  
    ax.axhline(0.05, color=C['grey'], linestyle=':',  
                linewidth=1.6, label='ε_min = 0.05')  
    ax.fill_between(range(len(ql_log['epsilon'])),  
                     ql_log['epsilon'], 0.05,  
                     alpha=0.08, color=C['blue'])  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Epsilon (ε)')  
    ax.legend(fontsize=8)  
    _title(ax, 'ε-Greedy Exploration Decay\nBoth Agents')  

    # 2-B: Q-Learning TD error  
    ax = fig.add_subplot(gs[1, 1])  
    td  = np.array(ql_log['td_errors'])  
    ax.plot(td, alpha=0.22, color=C['blue'], linewidth=0.8)  
    ax.plot(smooth(td, 20), color=C['blue'], linewidth=2.4,  
             label='Smoothed mean |TD error|')  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Mean |TD Error|')  
    ax.legend(fontsize=8)  
    _title(ax, 'Q-Learning\nMean TD Error per Episode')  

    # 2-C: DQN Huber loss (log scale)  
    ax = fig.add_subplot(gs[1, 2])  
    ls       = np.array(dqn_log['losses'])  
    nz_idx   = np.where(ls > 0)[0]  
    nz_vals  = ls[nz_idx]  
    if len(nz_vals) > 5:  
        ax.semilogy(nz_idx, nz_vals, alpha=0.22,  
                     color=C['red'], linewidth=0.8)  
        ax.semilogy(nz_idx, smooth(nz_vals, 20),  
                     color=C['red'], linewidth=2.4,  
                     label='Smoothed Huber Loss')  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Huber Loss (log scale)')  
    ax.legend(fontsize=8)  
    _title(ax, 'DQN\nHuber Loss During Training')  

    # ═══════════════════════════════════════════════════════════════  
    # ROW 3 — Policy + Q-table  
    # ═══════════════════════════════════════════════════════════════  

    x_pos = np.arange(env.N_ACTIONS)  
    names = list(env.ACTION_NAMES.values())  

    # 3-A: Q-Learning greedy policy distribution  
    ax = fig.add_subplot(gs[2, 0])  
    ql_dist = ql_agent.greedy_action_distribution() * 100  
    bars    = ax.bar(x_pos, ql_dist, color=ACTION_PALETTE,  
                      edgecolor='white', linewidth=0.8, width=0.6)  
    ax.set_xticks(x_pos)  
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)  
    ax.set_ylabel('% of Q-Table States')  
    for bar, val in zip(bars, ql_dist):  
        ax.text(bar.get_x() + bar.get_width() / 2,  
                bar.get_height() + 0.4,  
                f'{val:.1f}%', ha='center',  
                fontsize=9, fontweight='bold')  
    _title(ax, 'Q-Learning Learned Policy\nGreedy Action Distribution')  

    # 3-B: DQN eval policy distribution  
    ax = fig.add_subplot(gs[2, 1])  
    dqn_dist = dqn_eval['action_dist'] * 100  
    bars     = ax.bar(x_pos, dqn_dist, color=ACTION_PALETTE,  
                       edgecolor='white', linewidth=0.8, width=0.6)  
    ax.set_xticks(x_pos)  
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)  
    ax.set_ylabel('% of Evaluation Steps')  
    for bar, val in zip(bars, dqn_dist):  
        ax.text(bar.get_x() + bar.get_width() / 2,  
                bar.get_height() + 0.4,  
                f'{val:.1f}%', ha='center',  
                fontsize=9, fontweight='bold')  
    _title(ax, 'DQN Learned Policy\nGreedy Action Distribution')  

    # 3-C: Q-table growth  
    ax = fig.add_subplot(gs[2, 2])  
    qt = ql_log['q_table_sz']  
    ax.plot(qt, color=C['blue'], linewidth=2.4)  
    ax.fill_between(range(len(qt)), 0, qt,  
                     alpha=0.12, color=C['blue'])  
    ax.axhline(max(qt), color=C['red'], linestyle='--',  
                linewidth=1.8,  
                label=f'Max = {max(qt):,} states')  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Unique State-Action Entries')  
    ax.legend(fontsize=9)  
    _title(ax, 'Q-Table Growth\nUnique States Visited During Training')  

    # ═══════════════════════════════════════════════════════════════  
    # ROW 4 — Evaluation  
    # ═══════════════════════════════════════════════════════════════  

    # 4-A: Reward boxplot — all three agents  
    ax  = fig.add_subplot(gs[3, 0])  
    bdata  = [ql_eval['rewards'], dqn_eval['rewards'],  
               rand_eval['rewards']]  
    blabels = ['Q-Learning', 'DQN', 'Random']  
    bcolors = [C['blue'], C['red'], C['grey']]  
    bp = ax.boxplot(bdata, labels=blabels, patch_artist=True,  
                     medianprops=dict(color='black', linewidth=2.2),  
                     whiskerprops=dict(linewidth=1.6),  
                     capprops=dict(linewidth=1.6),  
                     flierprops=dict(marker='o', markersize=3,  
                                     alpha=0.5))  
    for patch, col in zip(bp['boxes'], bcolors):  
        patch.set_facecolor(col)  
        patch.set_alpha(0.65)  
    for i, (rewards, jx) in enumerate(zip(bdata, [1, 2, 3])):  
        jitter = np.random.normal(jx, 0.06, len(rewards))  
        ax.scatter(jitter, rewards,  
                    s=18, alpha=0.35,  
                    color=bcolors[i], zorder=3)  
    ax.set_ylabel('Episode Reward')  
    _title(ax, 'Evaluation Reward Distribution\n'  
                '50 Greedy Episodes · All Agents')  

    # 4-B: SLA violation bar chart  
    ax = fig.add_subplot(gs[3, 1])  
    agents   = ['Q-Learning', 'DQN', 'Random\nBaseline']  
    sla_vals = [ql_eval['sla_rate']   * 100,  
                 dqn_eval['sla_rate']  * 100,  
                 rand_eval['sla_rate'] * 100]  
    bar_clrs = [C['blue'], C['red'], C['grey']]  
    bars     = ax.bar(agents, sla_vals, color=bar_clrs,  
                       edgecolor='white', linewidth=0.8,  
                       width=0.5)  
    ax.axhline(10, color=C['red'], linestyle='--',  
                linewidth=2.0, label='10% SLA Target')  
    ax.set_ylabel('SLA Violation %')  
    ax.set_ylim(0, max(sla_vals) * 1.40)  
    ax.legend(fontsize=9)  
    for bar, val in zip(bars, sla_vals):  
        ax.text(bar.get_x() + bar.get_width() / 2,  
                bar.get_height() + 0.3,  
                f'{val:.1f}%', ha='center',  
                fontsize=10, fontweight='bold',  
                color=C['red'] if val > 10 else C['green'])  
    _title(ax, 'SLA Violation Rate\n(Lower is Better ↓)')  

    # 4-C: DQN Q-value heatmap  
    ax = fig.add_subplot(gs[3, 2])  
    sample_idx    = np.random.choice(len(env.norm_data),  
                                      size=14, replace=False)  
    sample_states = env.norm_data[sample_idx]  
    q_matrix      = dqn_agent.online_net.predict(  
        sample_states.astype(np.float32), verbose=0)  
    im = ax.imshow(q_matrix, aspect='auto',  
                    cmap='RdYlGn', interpolation='nearest')  
    ax.set_xticks(range(env.N_ACTIONS))  
    ax.set_xticklabels(names, rotation=20,  
                        ha='right', fontsize=8)  
    ax.set_yticks(range(len(sample_states)))  
    ax.set_yticklabels([f'State {i+1}'  
                         for i in range(len(sample_states))],  
                        fontsize=8)  
    plt.colorbar(im, ax=ax, label='Q-Value',  
                  fraction=0.046, pad=0.04)  
    # Highlight best action per row  
    for row in range(len(sample_states)):  
        best_col = int(np.argmax(q_matrix[row]))  
        rect = plt.Rectangle((best_col - 0.5, row - 0.5),  
                               1, 1, fill=False,  
                               edgecolor='black',  
                               linewidth=2.5, linestyle='--')  
        ax.add_patch(rect)  
    _title(ax, 'DQN Q-Value Heatmap\n'  
                '14 Sample States × 4 Actions\n'  
                '(dashed = greedy best)')  

    # ═══════════════════════════════════════════════════════════════  
    # ROW 5 — Extra analytics  
    # ═══════════════════════════════════════════════════════════════  

    # 5-A: Cumulative reward over training  
    ax = fig.add_subplot(gs[4, 0])  
    ql_cum  = np.cumsum(ql_log['ep_rewards'])  
    dqn_cum = np.cumsum(dqn_log['ep_rewards'])  
    n_min   = min(len(ql_cum), len(dqn_cum))  
    ax.plot(ql_cum[:n_min],  color=C['blue'],  
             linewidth=2.2, label='Q-Learning')  
    ax.plot(dqn_cum[:n_min], color=C['red'],  
             linewidth=2.2, label='DQN', linestyle='--')  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Cumulative Reward')  
    ax.legend(fontsize=9)  
    _title(ax, 'Cumulative Reward\nOver Training Episodes')  

    # 5-B: Episode length (steps per episode)  
    ax = fig.add_subplot(gs[4, 1])  
    ax.plot(smooth(np.array(ql_log['ep_steps']),  20),  
             color=C['blue'], linewidth=2.2, label='Q-Learning')  
    ax.plot(smooth(np.array(dqn_log['ep_steps']), 20),  
             color=C['red'],  linewidth=2.2, label='DQN',  
             linestyle='--')  
    ax.axhline(env.MAX_STEPS, color=C['grey'],  
                linestyle=':', linewidth=1.6,  
                label=f'MAX_STEPS={env.MAX_STEPS}')  
    ax.set_xlabel('Episode')  
    ax.set_ylabel('Steps per Episode')  
    ax.legend(fontsize=8)  
    _title(ax, 'Episode Length\n(Steps per Episode)')  

    # 5-C: Reward improvement vs Random baseline  
    ax   = fig.add_subplot(gs[4, 2])  
    base = rand_eval['mean_reward']  
    improvements = {  
        'Q-Learning': ql_eval['mean_reward'],  
        'DQN':        dqn_eval['mean_reward'],  
        'Random':     base,  
    }  
    pct_imp = {k: ((v - base) / abs(base)) * 100  
                if base != 0 else 0.0  
                for k, v in improvements.items()}  
    bar_colors_imp = [C['blue'], C['red'], C['grey']]  
    bars_imp = ax.bar(list(pct_imp.keys()),  
                       list(pct_imp.values()),  
                       color=bar_colors_imp,  
                       edgecolor='white', linewidth=0.8,  
                       width=0.5)  
    ax.axhline(0, color='black', linewidth=1.2)  
    ax.set_ylabel('% Improvement over Random Baseline')  
    for bar, val in zip(bars_imp, pct_imp.values()):  
        ypos = bar.get_height() + (0.5 if val >= 0 else -2.5)  
        ax.text(bar.get_x() + bar.get_width() / 2,  
                ypos, f'{val:+.1f}%',  
                ha='center', fontsize=10, fontweight='bold',  
                color=C['green'] if val > 0 else C['red'])  
    _title(ax, 'Mean Reward Improvement\nvs Random Baseline (%)')  

    # ─────────────────────────────────────────────────────────────────────────  
    plt.suptitle(  
        'Stage 4 — Reinforcement Learning for Cloud Resource Optimisation\n'  
        'Q-Learning vs DQN  |  Training Dynamics · Policy Analysis · '  
        'Evaluation · SLA Metrics',  
        fontsize=15, fontweight='bold', y=1.005,  
    )  
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  
    plt.show()  
    print(f"\n  ✅ Stage 4 dashboard saved → {save_path}")  


# ==============================================================================  
# STAGE 4.6 — MAIN STAGE 4 RUNNER  
# ==============================================================================  

def run_stage4(df: pd.DataFrame) -> dict:  
    """  
    Full Stage 4 pipeline.  

    Parameters  
    ----------  
    df : pd.DataFrame  
        Cloud-performance dataset with numeric feature columns.  

    Returns  
    -------  
    dict   Contains env, agents, logs, eval results, summary DataFrame.  
    """  

    print("\n" + "═" * 70)  
    print("  STAGE 4 — REINFORCEMENT LEARNING PIPELINE")  
    print("═" * 70)  

    # ── Feature matrix ────────────────────────────────────────────────────────  
    preferred = ['cpu_usage', 'memory_usage', 'network_traffic',  
                 'power_consumption', 'execution_time',  
                 'energy_efficiency']  
    available = [c for c in preferred if c in df.columns]  
    if len(available) < 4:  
        available = (df.select_dtypes(include=[np.number])  
                       .columns[:6].tolist())  

    data_arr = df[available].dropna().values.astype(np.float32)  
    print(f"\n  Features used  : {available}")  
    print(f"  Data shape     : {data_arr.shape}")  

    # ── Pad to 6 columns if needed ────────────────────────────────────────────  
    if data_arr.shape[1] < 6:  
        pad = np.zeros((len(data_arr),  
                         6 - data_arr.shape[1]),  
                        dtype=np.float32)  
        data_arr = np.hstack([data_arr, pad])  

    # ── Environment ───────────────────────────────────────────────────────────  
    env = CloudComputingEnv(data_arr, seed=SEED)  
    print(f"\n  Environment    : CloudComputingEnv")  
    print(f"  State size     : {env.state_size}")  
    print(f"  Action size    : {env.action_size}")  
    print(f"  Max steps/ep   : {env.MAX_STEPS}")  
    print(f"  SLA threshold  : {env.SLA_THRESHOLD}")  

    # ── Q-Learning ────────────────────────────────────────────────────────────  
    ql_agent = QLearningAgent(  
        n_actions = env.N_ACTIONS,  
        alpha     = 0.10,  
        gamma     = 0.95,  
        epsilon   = 1.00,  
        eps_min   = 0.05,  
        eps_decay = 0.997,  
    )  
    ql_log = ql_agent.train(env, n_episodes=600, verbose_every=100)  

    # ── DQN ───────────────────────────────────────────────────────────────────  
    dqn_agent = DQNAgent(  
        state_size  = env.state_size,  
        action_size = env.action_size,  
        lr          = 1e-3,  
        gamma       = 0.95,  
        epsilon     = 1.00,  
        eps_min     = 0.05,  
        eps_decay   = 0.997,  
        batch_size  = 64,  
        memory_cap  = 20_000,  
        target_sync = 10,  
    )  
    dqn_log = dqn_agent.train(env, n_episodes=500, verbose_every=50)  

    # ── Random baseline ───────────────────────────────────────────────────────  
    rand_agent = RandomAgent(n_actions=env.N_ACTIONS)  

    # ── Evaluation ────────────────────────────────────────────────────────────  
    print("\n" + "─" * 70)  
    print("  GREEDY POLICY EVALUATION  (50 episodes each)")  
    print("─" * 70)  

    ql_eval   = ql_agent.evaluate(env,  n_eval=50)  
    dqn_eval  = dqn_agent.evaluate(env, n_eval=50)  
    rand_eval = rand_agent.evaluate(env, n_eval=50)  

    for name, evl in [('Q-Learning', ql_eval),  
                       ('DQN',        dqn_eval),  
                       ('Random',     rand_eval)]:  
        print(f"\n  [{name}]")  
        print(f"    Mean Reward    : {evl['mean_reward']:>9.3f}")  
        print(f"    Std  Reward    : {evl['std_reward']:>9.3f}")  
        print(f"    Max  Reward    : {evl['max_reward']:>9.3f}")  
        print(f"    Min  Reward    : {evl['min_reward']:>9.3f}")  
        print(f"    SLA Violation  : {evl['sla_rate']*100:>8.2f}%")  
        print(f"    Action dist    :", end='')  
        for act_name, pct in zip(env.ACTION_NAMES.values(),  
                                  evl['action_dist'] * 100):  
            print(f"  {act_name}={pct:.1f}%", end='')  
        print()  

    # ── Visualise ─────────────────────────────────────────────────────────────  
    visualise_rl(  
        ql_log    = ql_log,  
        dqn_log   = dqn_log,  
        ql_eval   = ql_eval,  
        dqn_eval  = dqn_eval,  
        rand_eval = rand_eval,  
        ql_agent  = ql_agent,  
        dqn_agent = dqn_agent,  
        env       = env,  
        save_path = 'stage4_rl_dashboard.png',  
    )  

    # ── Summary table ─────────────────────────────────────────────────────────  
    print("\n" + "═" * 70)  
    print("  STAGE 4 — FINAL SUMMARY TABLE")  
    print("═" * 70)  
    df_sum = pd.DataFrame({  
        'Agent':             ['Q-Learning', 'DQN', 'Random'],  
        'Episodes':          [600, 500, '—'],  
        'Best Ep Reward':    [f"{max(ql_log['ep_rewards']):.3f}",  
                              f"{max(dqn_log['ep_rewards']):.3f}",  
                              '—'],  
        'Final Avg(100)':    [f"{np.mean(ql_log['ep_rewards'][-100:]):.3f}",  
                              f"{np.mean(dqn_log['ep_rewards'][-100:]):.3f}",  
                              '—'],  
        'Eval Mean Reward':  [f"{ql_eval['mean_reward']:.3f}",  
                              f"{dqn_eval['mean_reward']:.3f}",  
                              f"{rand_eval['mean_reward']:.3f}"],  
        'Eval Std Reward':   [f"{ql_eval['std_reward']:.3f}",  
                              f"{dqn_eval['std_reward']:.3f}",  
                              f"{rand_eval['std_reward']:.3f}"],  
        'SLA Violation %':   [f"{ql_eval['sla_rate']*100:.2f}%",  
                              f"{dqn_eval['sla_rate']*100:.2f}%",  
                              f"{rand_eval['sla_rate']*100:.2f}%"],  
    })  
    print(df_sum.to_string(index=False))  
    df_sum.to_csv('stage4_summary.csv', index=False)  
    print("\n  ✅ Stage 4 complete  │  summary → stage4_summary.csv")  

    return {  
        'env':        env,  
        'ql_agent':   ql_agent,  
        'dqn_agent':  dqn_agent,  
        'rand_agent': rand_agent,  
        'ql_log':     ql_log,  
        'dqn_log':    dqn_log,  
        'ql_eval':    ql_eval,  
        'dqn_eval':   dqn_eval,  
        'rand_eval':  rand_eval,  
        'summary':    df_sum,  
    }  


# ==============================================================================  
# STAGE 5 — FINAL DASHBOARD & REPORT  
# ==============================================================================  

class ResultsAggregator:  
    """  
    Collects stage outputs and produces the master comparison table  
    plus a JSON report for reproducibility.  
    """  

    def __init__(self):  
        self.regression     = {}  
        self.classification = {}  
        self.clustering     = {}  
        self.learning_theory= {}  
        self.rl             = {}  

    # ── Ingest helpers ────────────────────────────────────────────────────────  
    def add_regression(self, df_res: pd.DataFrame):  
        best = df_res.iloc[0]  
        self.regression = {  
            'best_model':  str(best.get('Model', '—')),  
            'test_r2':     float(best.get('Test R²', 0)),  
            'test_rmse':   float(best.get('RMSE', 0)),  
            'test_mae':    float(best.get('MAE', 0)),  
            'all_models':  df_res.to_dict(orient='records'),  
        }  
        print(f"  [REG]  Best → {self.regression['best_model']}  "  
              f"R²={self.regression['test_r2']:.4f}")  

    def add_classification(self, df_res: pd.DataFrame):  
        best = df_res.iloc[0]  
        self.classification = {  
            'best_model': str(best.get('Model', '—')),  
            'test_acc':   float(best.get('Test Acc', 0)),  
            'f1_score':   float(best.get('F1', 0)),  
            'cv_mean':    float(best.get('CV Mean', 0)),  
            'all_models': df_res.to_dict(orient='records'),  
        }  
        print(f"  [CLF]  Best → {self.classification['best_model']}  "  
              f"Acc={self.classification['test_acc']:.4f}")  

    def add_clustering(self, clust_res: dict):  
        self.clustering = {  
            algo: {  
                'silhouette': float(res.get('score', 0)),
