# Random search: connectome

Agents: ppo_bio, ppo_random_graph. Sampler seed 2028; trial 0 = defaults, trials 1-4 random. Validation seeds [0, 1], 1,000,000 steps per run.

Search space: `learning_rate` log-uniform [3e-05, 0.003]; `ent_coef` log-uniform [0.0001, 0.1]; `n_steps` one of {256, 512, 1024, 2048, 4096}.

Score = held-out fruit rate (sampled actions) averaged over validation seeds and over the agents (joint tuning).

| Rank | Trial | Settings | ppo_bio | ppo_random_graph | Score | Poison |
|---|---|---|---|---|---|---|
| 1 | 2 | learning_rate=0.0028, ent_coef=0.034, n_steps=2048 | 98.0% | 98.0% | 98.0% | 1% |
| 2 | 0 | defaults | 88.5% | 74.0% | 81.2% | 4% |
| 3 | 1 | learning_rate=0.00018, ent_coef=0.006, n_steps=512 | 88.0% | 65.5% | 76.8% | 4% |
| 4 | 3 | learning_rate=0.00019, ent_coef=0.031, n_steps=2048 | 89.5% | 48.5% | 69.0% | 4% |
| 5 | 4 | learning_rate=0.00018, ent_coef=0.00017, n_steps=1024 | 46.0% | 44.0% | 45.0% | 3% |

**Winner: trial 2** (learning_rate=0.0028, ent_coef=0.034, n_steps=2048), score 98.0%.
