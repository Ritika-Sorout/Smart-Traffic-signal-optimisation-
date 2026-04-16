# 🚦 Smart Traffic Signal Optimization
### Genetic Algorithm + Q-Learning for Chennai Urban Traffic Network

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📋 Overview

This project implements a **hybrid Genetic Algorithm + Q-Learning** framework to optimize traffic signal timings across 12 key intersections in Chennai, Tamil Nadu. The system achieves:

| Metric | Baseline (Fixed) | GA Only | GA + Q-Learning |
|--------|-----------------|---------|-----------------|
| Avg Wait Time (s) | 44.4 | 26.0 | 22.5* |
| Throughput (veh) | 303 | 817 | 860* |
| Queue Length | 932 | 384 | 310* |

> *Projected after extended training (1000+ episodes)

**Key Results:**
- ✅ **~41% reduction** in average wait time (GA phase)
- ✅ **~170% increase** in throughput
- ✅ **Convergence** achieved in 80 GA generations

---

## 🗺️ Intersections Modeled

| ID | Location | Base Flow (vph) |
|----|----------|-----------------|
| 0 | Anna Salai & Nungambakkam High Rd | 1850 |
| 1 | Kathipara Junction | 2200 |
| 2 | Koyambedu Junction | 2050 |
| 3 | Vadapalani Signal | 1620 |
| 4 | T.Nagar Pondy Bazaar | 1980 |
| 5 | Guindy Industrial Estate | 1430 |
| 6 | Adyar Signal | 1540 |
| 7 | Tambaram Signal | 1320 |
| 8 | Chromepet Junction | 1280 |
| 9 | Velachery Signal | 1710 |
| 10 | Perambur Signal | 1390 |
| 11 | Royapuram Signal | 1160 |

---

## 🛠️ System Architecture

```
┌──────────────────────────────────────────────┐
│           Chennai Traffic Network            │
│         (12 Intersections, 4 Phases)         │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Traffic Environment │  ← Simulation Engine
    │  (Queue + Flow Model)│
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │         PHASE 1: Genetic Algorithm       │
    │  • Population: 60 individuals            │
    │  • Chromosome: Green times per phase     │
    │  • Selection: Tournament (k=3)           │
    │  • Crossover: Single-point (rate=0.85)   │
    │  • Mutation: Gaussian noise (rate=0.12)  │
    │  • Generations: 80 (with elitism)        │
    └──────────┬──────────────────────────────┘
               │ Best solution (warm-start)
    ┌──────────▼──────────────────────────────┐
    │       PHASE 2: Q-Learning Agent          │
    │  • State: Discretized queue lengths      │
    │  • Action: 5 green-time levels           │
    │  • Reward: -wait_time + throughput       │
    │  • ε-greedy exploration (decay=0.995)    │
    │  • Independent per-intersection tables   │
    └──────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Optimal Signal     │
    │   Timing Plans       │
    └─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/traffic-optimization.git
cd traffic-optimization
pip install -r requirements.txt
```

### 2. Run Full Optimization

```bash
python src/main.py
```

### 3. Generate Traffic Data Only

```bash
python generate_data.py
```

### 4. View Results

All outputs are saved to `results/`:
- `ga_convergence.png` — GA fitness over generations
- `ql_training.png` — Q-Learning reward & wait-time curves
- `method_comparison.png` — Baseline vs GA vs Hybrid bar charts
- `timing_heatmap.png` — Optimal green times (intersection × phase)
- `queue_evolution.png` — Queue length over simulation time
- `reward_distribution.png` — Early vs late training reward distribution
- `phase_radar.png` — Radar chart of phase timings
- `traffic_network.png` — Chennai intersection network diagram
- `metrics.json` — All numeric results

---

## 📐 Mathematical Formulation

### Fitness Function (GA)
```
F(x) = Σᵢ [−λ·Wᵢ(x) + μ·Tᵢ(x)]
```
Where:
- `Wᵢ` = average wait time at intersection i
- `Tᵢ` = throughput at intersection i  
- `λ = 0.5`, `μ = 0.1` (tuned weights)

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ·max_{a'} Q(s',a') − Q(s,a)]
```
- `α = 0.15` (learning rate)
- `γ = 0.95` (discount factor)
- `ε` decays from 1.0 → 0.01 over 500 episodes

### Webster's Delay Formula (Baseline)
```
d = C(1−λ)²/[2(1−λx)] + x²/[2q(1−x)]
```

---

## 📁 Project Structure

```
traffic-optimization/
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
├── generate_data.py        ← Traffic data generator
├── src/
│   └── main.py             ← GA + QL implementation (all-in-one)
├── data/
│   └── traffic_data.csv    ← Simulated Chennai traffic data (1440 rows)
└── results/
    ├── ga_convergence.png
    ├── ql_training.png
    ├── method_comparison.png
    ├── timing_heatmap.png
    ├── queue_evolution.png
    ├── reward_distribution.png
    ├── phase_radar.png
    ├── traffic_network.png
    └── metrics.json
```

---

## 🔧 Configuration

Edit `CONFIG` dict in `src/main.py`:

```python
CONFIG = {
    "intersections":    12,
    "phases":           4,
    "ga_population":    60,    # Increase for better solutions
    "ga_generations":   80,
    "ga_mutation_rate": 0.12,
    "ql_episodes":      500,   # Increase for convergence
    "ql_alpha":         0.15,
    "ql_gamma":         0.95,
    "min_green":        15,    # seconds
    "max_green":        90,    # seconds
}
```

---

## 📚 Academic References

1. Webster, F.V. (1958). *Traffic Signal Settings*. Road Research Technical Paper No. 39.
2. Srinivas, N., Deb, K. (1994). Multiobjective optimization using nondominated sorting in GAs. *Evolutionary Computation*.
3. Watkins, C., Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3-4), 279–292.
4. Sutton, R.S., Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

---

## 👤 Author

**Traffic AI Research Lab** | Chennai, Tamil Nadu  
Contact: research@trafficai.edu.in

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
