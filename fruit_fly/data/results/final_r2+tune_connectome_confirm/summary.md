# Results: final_r2, tune_connectome_confirm

Held-out: 100 fresh test episodes per seed, never seen in training. Mean ± standard deviation over seeds. Sampled = PPO sampling (DQN: epsilon-greedy); greedy = most-likely action.

| Condition | Seeds | Fruit (sampled) | Poison (sampled) | Timeout (sampled) | Fruit (greedy) | Poison (greedy) | Steps to fruit (sampled) | Steps to 80% training success* |
|---|---|---|---|---|---|---|---|---|
| PPO · MLP | 5 | 97% ± 2% | 1% ± 1% | 2% ± 2% | 72% ± 9% | 1% ± 1% | 41 ± 4 | 155k (5/5) |
| PPO · Bio connectome | 5 | 85% ± 10% | 4% ± 2% | 11% ± 9% | 49% ± 10% | 1% ± 1% | 53 ± 5 | 369k (5/5) |
| PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | 5 | 91% ± 13% | 1% ± 1% | 8% ± 13% | 60% ± 23% | 0% ± 0% | 53 ± 22 | 355k (5/5) |
| PPO · Random connectome | 5 | 87% ± 6% | 4% ± 3% | 10% ± 8% | 55% ± 14% | 1% ± 1% | 50 ± 6 | 436k (5/5) |
| PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | 5 | 94% ± 5% | 1% ± 2% | 5% ± 4% | 64% ± 12% | 0% ± 0% | 51 ± 8 | 268k (5/5) |
| DQN · MLP | 5 | 5% ± 3% | 1% ± 1% | 94% ± 3% | 4% ± 4% | 0% ± 0% | 41 ± 39 | never (0/5) |
| *Random agent* (reference) | 5 | 9% ± 2% | 8% ± 4% | 83% ± 5% | – | – | 60 ± 21 | – |
| *Odor rule* (reference) | 5 | 56% ± 11% | 44% ± 11% | 0% ± 0% | – | – | 22 ± 2 | – |
| *Informed rule* (reference) | 5 | 97% ± 2% | 3% ± 2% | 0% ± 0% | – | – | 25 ± 1 | – |

\* Median training steps until the success rate over the last 50 training episodes first reached 80%, over the seeds that reached it (count in brackets).

## Comparisons (sampled fruit rate per seed)

| A | B | A − B | Unpaired permutation p | Paired sign-flip p (seeds) |
|---|---|---|---|---|
| PPO · MLP | PPO · Bio connectome | +12% | 0.032 | 0.125 (5) |
| PPO · MLP | PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | +6% | 0.579 | 0.750 (5) |
| PPO · MLP | PPO · Random connectome | +10% | 0.008 | 0.062 (5) |
| PPO · MLP | PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | +3% | 0.246 | 0.125 (5) |
| PPO · MLP | DQN · MLP | +92% | 0.008 | 0.062 (5) |
| PPO · Bio connectome | PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | -6% | 0.476 | 0.562 (5) |
| PPO · Bio connectome | PPO · Random connectome | -1% | 0.833 | 0.688 (5) |
| PPO · Bio connectome | PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | -8% | 0.159 | 0.188 (5) |
| PPO · Bio connectome | DQN · MLP | +80% | 0.008 | 0.062 (5) |
| PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | PPO · Random connectome | +4% | 0.540 | 0.562 (5) |
| PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | -3% | 0.881 | 0.938 (5) |
| PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | DQN · MLP | +86% | 0.008 | 0.062 (5) |
| PPO · Random connectome | PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | -7% | 0.111 | 0.125 (5) |
| PPO · Random connectome | DQN · MLP | +82% | 0.008 | 0.062 (5) |
| PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | DQN · MLP | +89% | 0.008 | 0.062 (5) |

Exact tests over all permutations. The paired test matches seeds (same odor pair and test episodes); with n seeds its smallest possible p is 2/2^n (0.0625 for 5 seeds), so read effect sizes and per-seed values, not only p.

## Per-seed sampled fruit rate

| Condition | 100 | 101 | 102 | 103 | 104 |
|---|---|---|---|---|---|
| PPO · MLP | 99% | 94% | 99% | 98% | 95% |
| PPO · Bio connectome | 93% | 97% | 86% | 79% | 71% |
| PPO · Bio connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | 67% | 98% | 96% | 98% | 96% |
| PPO · Random connectome | 91% | 89% | 91% | 86% | 76% |
| PPO · Random connectome · ent_coef=0.034 · learning_rate=0.0028 · n_steps=2048 | 98% | 89% | 99% | 94% | 88% |
| DQN · MLP | 2% | 2% | 10% | 4% | 6% |
