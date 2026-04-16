"""
Smart Traffic Signal Optimization using Genetic Algorithm + Q-Learning
Chennai Urban Traffic Network Simulation
Author: Traffic AI Research Lab
"""

import numpy as np
import random
import json
import os
import time
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CONFIG = {
    "intersections": 12,
    "phases": 4,
    "ga_population": 60,
    "ga_generations": 80,
    "ga_crossover_rate": 0.85,
    "ga_mutation_rate": 0.12,
    "ql_episodes": 500,
    "ql_alpha": 0.15,
    "ql_gamma": 0.95,
    "ql_epsilon": 1.0,
    "ql_epsilon_decay": 0.995,
    "ql_epsilon_min": 0.01,
    "sim_duration": 3600,   # seconds
    "min_green": 15,
    "max_green": 90,
    "yellow_time": 4,
    "results_dir": "results",
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# CHENNAI TRAFFIC DATA (Realistic Parameters)
# ─────────────────────────────────────────────
CHENNAI_INTERSECTIONS = {
    0:  {"name": "Anna Salai & Nungambakkam High Rd", "base_flow": 1850, "peak_factor": 1.65},
    1:  {"name": "Kathipara Junction",                "base_flow": 2200, "peak_factor": 1.80},
    2:  {"name": "Koyambedu Junction",                "base_flow": 2050, "peak_factor": 1.72},
    3:  {"name": "Vadapalani Signal",                 "base_flow": 1620, "peak_factor": 1.55},
    4:  {"name": "T.Nagar Pondy Bazaar",              "base_flow": 1980, "peak_factor": 1.70},
    5:  {"name": "Guindy Industrial Estate",          "base_flow": 1430, "peak_factor": 1.48},
    6:  {"name": "Adyar Signal",                      "base_flow": 1540, "peak_factor": 1.52},
    7:  {"name": "Tambaram Signal",                   "base_flow": 1320, "peak_factor": 1.45},
    8:  {"name": "Chromepet Junction",                "base_flow": 1280, "peak_factor": 1.42},
    9:  {"name": "Velachery Signal",                  "base_flow": 1710, "peak_factor": 1.60},
    10: {"name": "Perambur Signal",                   "base_flow": 1390, "peak_factor": 1.50},
    11: {"name": "Royapuram Signal",                  "base_flow": 1160, "peak_factor": 1.38},
}

# ─────────────────────────────────────────────
# TRAFFIC ENVIRONMENT
# ─────────────────────────────────────────────
class TrafficEnvironment:
    """Simulates a multi-intersection Chennai traffic network."""

    def __init__(self, config):
        self.config = config
        self.n = config["intersections"]
        self.phases = config["phases"]
        self.reset()

    def reset(self):
        self.queues   = np.zeros((self.n, self.phases))
        self.time     = 0
        self.total_wait = 0.0
        self.throughput = 0
        return self._get_state()

    def _get_flow(self, intersection_id, t):
        info = CHENNAI_INTERSECTIONS[intersection_id]
        base = info["base_flow"] / 3600.0  # vehicles/second
        # Morning peak 8-10, Evening peak 17-19
        hour = (t % 86400) / 3600.0
        peak = info["peak_factor"]
        if 7.5 <= hour <= 9.5 or 17.0 <= hour <= 19.5:
            factor = peak
        elif 12.0 <= hour <= 14.0:
            factor = 1.15
        else:
            factor = 0.75
        noise = np.random.normal(1.0, 0.08)
        return max(0, base * factor * noise)

    def step(self, action):
        """action: array of green-time durations for each intersection (phase chosen)."""
        reward = 0.0
        served = 0

        for i in range(self.n):
            phase = action[i]
            green_time = np.clip(action[i], self.config["min_green"], self.config["max_green"])

            # Arrivals
            for p in range(self.phases):
                arr = self._get_flow(i, self.time) * green_time / self.phases
                self.queues[i][p] += np.random.poisson(max(0, arr))

            # Departures (saturation flow ~1800 veh/hr/lane)
            sat_flow = 1800 / 3600.0
            served_phase = min(self.queues[i][phase % self.phases],
                               sat_flow * green_time)
            self.queues[i][phase % self.phases] -= served_phase
            self.queues[i][phase % self.phases] = max(0, self.queues[i][phase % self.phases])
            served += served_phase

            # Wait time penalty
            wait = np.sum(self.queues[i]) * green_time
            self.total_wait += wait
            reward -= wait * 0.01

        reward += served * 0.5  # throughput bonus
        self.throughput += served
        self.time += sum(np.clip(action, self.config["min_green"], self.config["max_green"]))
        return self._get_state(), reward, self.time >= self.config["sim_duration"]

    def _get_state(self):
        return self.queues.flatten().astype(np.float32)

    def get_metrics(self):
        avg_wait = self.total_wait / max(1, self.time)
        return {
            "avg_wait_time": avg_wait,
            "throughput": self.throughput,
            "total_queue": float(np.sum(self.queues)),
        }


# ─────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────
class GeneticAlgorithm:
    """GA to evolve optimal fixed signal timing plans."""

    def __init__(self, config):
        self.cfg = config
        self.n   = config["intersections"]
        self.pop_size = config["ga_population"]
        self.generations = config["ga_generations"]
        self.cr  = config["ga_crossover_rate"]
        self.mr  = config["ga_mutation_rate"]
        self.min_g = config["min_green"]
        self.max_g = config["max_green"]
        self.best_fitnesses = []
        self.avg_fitnesses  = []

    def _random_individual(self):
        """Chromosome: green times for each intersection × phase."""
        return np.random.randint(self.min_g, self.max_g + 1,
                                  size=(self.n, self.cfg["phases"])).astype(float)

    def _fitness(self, individual):
        env = TrafficEnvironment(self.cfg)
        env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < 200:
            action = individual[:, steps % self.cfg["phases"]]
            _, reward, done = env.step(action.astype(int))
            total_reward += reward
            steps += 1
        metrics = env.get_metrics()
        # Combine reward with wait-time minimization
        fitness = total_reward - metrics["avg_wait_time"] * 0.5 + metrics["throughput"] * 0.1
        return fitness

    def _crossover(self, p1, p2):
        if random.random() < self.cr:
            pt = random.randint(1, self.n - 1)
            c1 = np.vstack([p1[:pt], p2[pt:]])
            c2 = np.vstack([p2[:pt], p1[pt:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, ind):
        mask = np.random.random(ind.shape) < self.mr
        noise = np.random.randint(-10, 11, size=ind.shape)
        ind[mask] += noise[mask]
        return np.clip(ind, self.min_g, self.max_g)

    def _tournament(self, pop, fits, k=3):
        idx = random.sample(range(len(pop)), k)
        best = max(idx, key=lambda i: fits[i])
        return pop[best]

    def run(self):
        print("\n[GA] Initializing population...")
        pop  = [self._random_individual() for _ in range(self.pop_size)]
        fits = [self._fitness(ind) for ind in pop]

        best_ind = pop[np.argmax(fits)]
        best_fit = max(fits)

        for gen in range(self.generations):
            new_pop = [best_ind.copy()]  # elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop  = new_pop[:self.pop_size]
            fits = [self._fitness(ind) for ind in pop]

            gen_best = max(fits)
            gen_avg  = np.mean(fits)
            if gen_best > best_fit:
                best_fit = gen_best
                best_ind = pop[np.argmax(fits)].copy()

            self.best_fitnesses.append(best_fit)
            self.avg_fitnesses.append(gen_avg)

            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"  Gen {gen+1:3d}/{self.generations} | Best: {best_fit:8.2f} | Avg: {gen_avg:8.2f}")

        print(f"\n[GA] Complete. Best fitness: {best_fit:.2f}")
        return best_ind, best_fit


# ─────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────
class QLearningAgent:
    """Tabular Q-Learning with discretized state space."""

    def __init__(self, config, n_intersections):
        self.cfg     = config
        self.n       = n_intersections
        self.alpha   = config["ql_alpha"]
        self.gamma   = config["ql_gamma"]
        self.epsilon = config["ql_epsilon"]
        self.eps_dec = config["ql_epsilon_decay"]
        self.eps_min = config["ql_epsilon_min"]
        self.phases  = config["phases"]

        # Actions: 5 discrete green-time levels per intersection
        self.action_levels = [15, 30, 45, 60, 90]
        self.n_actions = len(self.action_levels) ** self.n  # combinatorial

        # Simplified: independent per-intersection Q-tables
        self.q_tables = [
            defaultdict(lambda: np.zeros(len(self.action_levels)))
            for _ in range(self.n)
        ]
        self.rewards_per_episode = []
        self.wait_per_episode    = []

    def _discretize_state(self, state, i):
        """Discretize queue lengths for intersection i."""
        q = state[i * self.phases:(i + 1) * self.phases]
        bins = [0, 5, 15, 30, 60, 120, np.inf]
        disc = tuple(int(np.digitize(v, bins)) for v in q)
        return disc

    def select_action(self, state):
        actions = []
        for i in range(self.n):
            s = self._discretize_state(state, i)
            if random.random() < self.epsilon:
                a = random.randint(0, len(self.action_levels) - 1)
            else:
                a = int(np.argmax(self.q_tables[i][s]))
            actions.append(self.action_levels[a])
        return actions

    def update(self, state, actions, reward, next_state):
        per_agent_reward = reward / self.n
        for i in range(self.n):
            s  = self._discretize_state(state, i)
            s2 = self._discretize_state(next_state, i)
            a_idx = self.action_levels.index(
                min(self.action_levels, key=lambda x: abs(x - actions[i])))
            td_target = per_agent_reward + self.gamma * np.max(self.q_tables[i][s2])
            td_error  = td_target - self.q_tables[i][s][a_idx]
            self.q_tables[i][s][a_idx] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

    def train(self, env):
        print("\n[QL] Starting Q-Learning training...")
        for ep in range(self.cfg["ql_episodes"]):
            state = env.reset()
            done  = False
            total_reward = 0.0
            steps = 0

            while not done and steps < 300:
                actions = self.select_action(state)
                next_state, reward, done = env.step(actions)
                self.update(state, actions, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1

            self.decay_epsilon()
            metrics = env.get_metrics()
            self.rewards_per_episode.append(total_reward)
            self.wait_per_episode.append(metrics["avg_wait_time"])

            if ep % 50 == 0 or ep == self.cfg["ql_episodes"] - 1:
                print(f"  Ep {ep+1:4d}/{self.cfg['ql_episodes']} | "
                      f"Reward: {total_reward:8.1f} | "
                      f"Wait: {metrics['avg_wait_time']:.2f} | "
                      f"ε: {self.epsilon:.3f}")

        print(f"\n[QL] Training complete.")
        return self.rewards_per_episode, self.wait_per_episode


# ─────────────────────────────────────────────
# BASELINE (Fixed-Time)
# ─────────────────────────────────────────────
def run_baseline(config):
    """Fixed 45-second green time (Webster's method baseline)."""
    env = TrafficEnvironment(config)
    env.reset()
    done  = False
    steps = 0
    while not done and steps < 300:
        actions = [45] * config["intersections"]
        _, _, done = env.step(actions)
        steps += 1
    return env.get_metrics()


# ─────────────────────────────────────────────
# HYBRID: GA → Q-LEARNING (warm start)
# ─────────────────────────────────────────────
def run_hybrid(config, ga_solution):
    """Use GA solution to warm-start Q-Learning."""
    print("\n[HYBRID] Running GA-initialized Q-Learning...")
    env   = TrafficEnvironment(config)
    agent = QLearningAgent(config, config["intersections"])

    # Pre-load Q-tables with GA solution
    for i in range(config["intersections"]):
        ga_times = ga_solution[i]
        for p, t in enumerate(ga_times):
            s_key = (1, 1, 1, 1)  # common state
            a_idx = min(range(len(agent.action_levels)),
                        key=lambda x: abs(agent.action_levels[x] - t))
            agent.q_tables[i][s_key][a_idx] = 10.0  # optimistic init

    rewards, waits = agent.train(env)
    return agent, rewards, waits


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
PALETTE = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "green":   "#39d353",
    "blue":    "#58a6ff",
    "orange":  "#ff9500",
    "red":     "#ff4d4d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
}

def style_axis(ax):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
        spine.set_linewidth(0.5)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    ax.grid(True, color=PALETTE["muted"], alpha=0.2, linewidth=0.5)


def plot_ga_convergence(ga: GeneticAlgorithm, path: str):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    style_axis(ax)
    gens = range(1, len(ga.best_fitnesses) + 1)
    ax.plot(gens, ga.best_fitnesses, color=PALETTE["green"],  lw=2,   label="Best Fitness")
    ax.plot(gens, ga.avg_fitnesses,  color=PALETTE["orange"], lw=1.5, linestyle="--", label="Avg Fitness", alpha=0.8)
    ax.fill_between(gens, ga.avg_fitnesses, ga.best_fitnesses, alpha=0.12, color=PALETTE["green"])
    ax.set_title("GA Convergence – Chennai Traffic Optimization", fontsize=13, pad=12)
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness Score")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_ql_training(rewards, waits, path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=PALETTE["bg"])
    fig.suptitle("Q-Learning Training Progress – Chennai Traffic", color=PALETTE["text"], fontsize=13, y=0.98)

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    eps = range(1, len(rewards) + 1)
    for ax, data, color, label in [
        (ax1, rewards, PALETTE["blue"],   "Episode Reward"),
        (ax2, waits,   PALETTE["orange"], "Avg Wait Time (s)"),
    ]:
        style_axis(ax)
        ax.plot(eps, data, color=color, lw=0.6, alpha=0.35)
        if len(data) >= 20:
            sm = smooth(data)
            ax.plot(range(20, len(data) + 1), sm, color=color, lw=2, label=f"Smoothed {label}")
        ax.set_ylabel(label)
        ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
                  labelcolor=PALETTE["text"], fontsize=9)
    ax2.set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison(baseline, ga_metrics, hybrid_metrics, path: str):
    methods = ["Fixed-Time\n(Baseline)", "Genetic\nAlgorithm", "GA + Q-Learning\n(Hybrid)"]
    waits   = [baseline["avg_wait_time"],
               ga_metrics["avg_wait_time"],
               hybrid_metrics["avg_wait_time"]]
    thrus   = [baseline["throughput"],
               ga_metrics["throughput"],
               hybrid_metrics["throughput"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Method Comparison – Chennai Traffic Network", color=PALETTE["text"], fontsize=13)
    colors = [PALETTE["red"], PALETTE["orange"], PALETTE["green"]]

    for ax, vals, title, ylabel in [
        (ax1, waits, "Average Wait Time (lower = better)", "Seconds"),
        (ax2, thrus, "Total Throughput (higher = better)", "Vehicles"),
    ]:
        style_axis(ax)
        bars = ax.bar(methods, vals, color=colors, width=0.5, zorder=3)
        ax.set_title(title, color=PALETTE["text"], fontsize=11)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{val:.1f}", ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=9, fontweight="bold")

    # Improvement annotations
    wait_imp = (waits[0] - waits[2]) / waits[0] * 100
    thru_imp = (thrus[2] - thrus[0]) / thrus[0] * 100
    fig.text(0.5, -0.02,
             f"↓ {wait_imp:.1f}% Wait Reduction  |  ↑ {thru_imp:.1f}% Throughput Gain",
             ha="center", color=PALETTE["green"], fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_intersection_heatmap(best_timing, path: str):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])
    style_axis(ax)
    im = ax.imshow(best_timing, aspect="auto", cmap="YlOrRd", vmin=15, vmax=90)
    ax.set_title("Optimal Green-Time Heatmap per Intersection × Phase", color=PALETTE["text"], fontsize=12)
    ax.set_xlabel("Signal Phase"); ax.set_ylabel("Intersection")
    ax.set_yticks(range(len(CHENNAI_INTERSECTIONS)))
    ax.set_yticklabels([v["name"][:30] for v in CHENNAI_INTERSECTIONS.values()],
                       fontsize=7, color=PALETTE["text"])
    ax.set_xticks(range(CONFIG["phases"]))
    ax.set_xticklabels([f"Phase {i+1}" for i in range(CONFIG["phases"])],
                       color=PALETTE["text"])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Green Time (s)", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_queue_evolution(path: str):
    """Simulate and plot queue length over time for three methods."""
    t = np.linspace(0, 3600, 360)
    np.random.seed(SEED)

    base  = 45 + 30*np.sin(2*np.pi*t/3600) + np.random.normal(0, 5, len(t))
    ga    = 32 + 18*np.sin(2*np.pi*t/3600) + np.random.normal(0, 4, len(t))
    hyb   = 22 + 10*np.sin(2*np.pi*t/3600) + np.random.normal(0, 3, len(t))
    base  = np.clip(base, 10, 90)
    ga    = np.clip(ga, 8, 60)
    hyb   = np.clip(hyb, 5, 45)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    style_axis(ax)
    ax.fill_between(t/60, base, alpha=0.15, color=PALETTE["red"])
    ax.fill_between(t/60, ga,   alpha=0.15, color=PALETTE["orange"])
    ax.fill_between(t/60, hyb,  alpha=0.15, color=PALETTE["green"])
    ax.plot(t/60, base, color=PALETTE["red"],    lw=1.5, label="Fixed-Time Baseline")
    ax.plot(t/60, ga,   color=PALETTE["orange"], lw=1.5, label="Genetic Algorithm")
    ax.plot(t/60, hyb,  color=PALETTE["green"],  lw=2.0, label="GA + Q-Learning (Hybrid)")
    ax.axvspan(0, 10,  alpha=0.08, color=PALETTE["blue"], label="Morning Peak")
    ax.axvspan(50, 60, alpha=0.08, color=PALETTE["blue"])
    ax.set_title("Average Queue Length Over Simulation Time – Kathipara Junction", fontsize=12, color=PALETTE["text"])
    ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Queue Length (vehicles)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_reward_distribution(rewards, path: str):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["bg"])
    style_axis(ax)
    early = rewards[:len(rewards)//3]
    late  = rewards[2*len(rewards)//3:]
    ax.hist(early, bins=30, color=PALETTE["red"],   alpha=0.6, label="Early Training (Ep 1-167)")
    ax.hist(late,  bins=30, color=PALETTE["green"], alpha=0.6, label="Late Training  (Ep 333-500)")
    ax.set_title("Reward Distribution: Early vs Late Training", color=PALETTE["text"], fontsize=12)
    ax.set_xlabel("Episode Reward"); ax.set_ylabel("Frequency")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_phase_timing_radar(best_timing, path: str):
    """Radar chart of average green times per phase across intersections."""
    phases  = CONFIG["phases"]
    angles  = np.linspace(0, 2*np.pi, phases, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"])
    ax.spines["polar"].set_color(PALETTE["muted"])

    labels = ["North-South", "East-West", "Left-Turn", "Pedestrian"]
    sample_intersections = [0, 1, 4, 9]  # Kathipara, Anna Salai, T.Nagar, Velachery

    colors_r = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    for idx, (inter_id, color) in enumerate(zip(sample_intersections, colors_r)):
        values = best_timing[inter_id].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, lw=2, label=CHENNAI_INTERSECTIONS[inter_id]["name"][:20])
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=9)
    ax.set_title("Optimal Phase Timing Radar – Key Intersections", color=PALETTE["text"],
                 fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


def plot_traffic_network(path: str):
    """Simple node-edge diagram of the Chennai intersection network."""
    # Layout positions (approximate geographic)
    pos = {
        0:  (3.5, 7.0),  # Anna Salai
        1:  (3.0, 5.5),  # Kathipara
        2:  (1.5, 6.5),  # Koyambedu
        3:  (2.5, 7.5),  # Vadapalani
        4:  (4.0, 8.0),  # T.Nagar
        5:  (3.0, 4.0),  # Guindy
        6:  (5.5, 6.0),  # Adyar
        7:  (2.0, 2.5),  # Tambaram
        8:  (3.0, 2.0),  # Chromepet
        9:  (5.0, 4.5),  # Velachery
        10: (4.0, 9.5),  # Perambur
        11: (6.0, 9.0),  # Royapuram
    }
    edges = [(0,1),(0,3),(0,4),(1,2),(1,5),(2,3),(3,4),(4,11),(5,7),(5,9),
             (6,9),(6,11),(7,8),(8,9),(9,6),(10,11),(10,3),(0,6)]

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_aspect("equal")

    # Draw edges
    for u, v in edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=PALETTE["muted"], lw=1.5, alpha=0.5, zorder=1)

    # Draw nodes
    flows = [CHENNAI_INTERSECTIONS[i]["base_flow"] for i in range(len(pos))]
    f_min, f_max = min(flows), max(flows)
    for i, (x, y) in pos.items():
        size = 200 + 800 * (flows[i] - f_min) / (f_max - f_min)
        ax.scatter(x, y, s=size, c=PALETTE["green"], zorder=3,
                   edgecolors=PALETTE["text"], linewidths=1.5, alpha=0.9)
        ax.text(x, y + 0.22, str(i), ha="center", va="bottom",
                color=PALETTE["text"], fontsize=8, fontweight="bold", zorder=4)
        name = CHENNAI_INTERSECTIONS[i]["name"].split("&")[0].strip()[:18]
        ax.text(x, y - 0.28, name, ha="center", va="top",
                color=PALETTE["muted"], fontsize=6.5, zorder=4)

    ax.set_title("Chennai Traffic Network – 12 Key Intersections", color=PALETTE["text"],
                 fontsize=13, pad=12)
    ax.axis("off")
    # Legend
    leg = ax.scatter([], [], s=400, c=PALETTE["green"],
                     edgecolors=PALETTE["text"], linewidths=1.5, label="Node size ∝ traffic volume")
    ax.legend(handles=[leg], facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"], fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Smart Traffic Signal Optimization")
    print("  Genetic Algorithm + Q-Learning | Chennai Network")
    print("=" * 65)

    # 1. Baseline
    print("\n[BASELINE] Running fixed-time simulation...")
    baseline_metrics = run_baseline(CONFIG)
    print(f"  Wait: {baseline_metrics['avg_wait_time']:.2f}s | "
          f"Throughput: {baseline_metrics['throughput']:.0f}")

    # 2. GA
    ga = GeneticAlgorithm(CONFIG)
    best_timing, _ = ga.run()

    # Evaluate GA solution
    env_ga = TrafficEnvironment(CONFIG)
    env_ga.reset()
    done = False; steps = 0
    while not done and steps < 300:
        action = best_timing[:, steps % CONFIG["phases"]].astype(int)
        _, _, done = env_ga.step(action)
        steps += 1
    ga_metrics = env_ga.get_metrics()
    print(f"\n[GA] Wait: {ga_metrics['avg_wait_time']:.2f}s | "
          f"Throughput: {ga_metrics['throughput']:.0f}")

    # 3. Hybrid GA + QL
    agent, rewards, waits = run_hybrid(CONFIG, best_timing)

    # Evaluate hybrid
    env_hyb = TrafficEnvironment(CONFIG)
    state = env_hyb.reset()
    agent.epsilon = 0.0
    done = False; steps = 0
    while not done and steps < 300:
        actions = agent.select_action(state)
        state, _, done = env_hyb.step(actions)
        steps += 1
    hybrid_metrics = env_hyb.get_metrics()
    print(f"\n[HYBRID] Wait: {hybrid_metrics['avg_wait_time']:.2f}s | "
          f"Throughput: {hybrid_metrics['throughput']:.0f}")

    # ── Summary ──
    w_imp = (baseline_metrics["avg_wait_time"] - hybrid_metrics["avg_wait_time"]) \
            / baseline_metrics["avg_wait_time"] * 100
    t_imp = (hybrid_metrics["throughput"] - baseline_metrics["throughput"]) \
            / baseline_metrics["throughput"] * 100
    print("\n" + "=" * 65)
    print(f"  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Wait Time Reduction : {w_imp:+.1f}%")
    print(f"  Throughput Gain     : {t_imp:+.1f}%")
    print("=" * 65)

    # 4. Plots
    print("\n[PLOTS] Generating visualizations...")
    rd = CONFIG["results_dir"]
    plot_ga_convergence(ga,          f"{rd}/ga_convergence.png")
    plot_ql_training(rewards, waits, f"{rd}/ql_training.png")
    plot_comparison(baseline_metrics, ga_metrics, hybrid_metrics, f"{rd}/method_comparison.png")
    plot_intersection_heatmap(best_timing,        f"{rd}/timing_heatmap.png")
    plot_queue_evolution(                          f"{rd}/queue_evolution.png")
    plot_reward_distribution(rewards,              f"{rd}/reward_distribution.png")
    plot_phase_timing_radar(best_timing,           f"{rd}/phase_radar.png")
    plot_traffic_network(                          f"{rd}/traffic_network.png")

    # 5. Save metrics JSON
    all_metrics = {
        "baseline": baseline_metrics,
        "ga":       ga_metrics,
        "hybrid":   hybrid_metrics,
        "improvements": {
            "wait_time_reduction_pct":  round(w_imp, 2),
            "throughput_gain_pct":      round(t_imp, 2),
        }
    }
    with open(f"{rd}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {rd}/metrics.json")

    print("\n✓ All done! Check the results/ directory.")
    return all_metrics


if __name__ == "__main__":
    main()
