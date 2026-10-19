# Random search: ppo_mlp

Agents: ppo_mlp. Sampler seed 2026; trial 0 = defaults, trials 1-12 random. Validation seeds [0, 1, 2], 300,000 steps per run.

Search space: `learning_rate` log-uniform [3e-05, 0.003]; `ent_coef` log-uniform [0.0001, 0.1]; `n_steps` one of {256, 512, 1024, 2048, 4096}.

Score = held-out fruit rate (sampled actions) averaged over validation seeds.

| Rank | Trial | Settings | ppo_mlp | Score | Poison |
|---|---|---|---|---|---|
| 1 | 9 | learning_rate=0.00034, ent_coef=0.002, n_steps=512 | 87.7% | 87.7% | 4% |
| 2 | 0 | defaults | 85.3% | 85.3% | 3% |
| 3 | 12 | learning_rate=0.00022, ent_coef=0.00079, n_steps=1024 | 80.0% | 80.0% | 0% |
| 4 | 11 | learning_rate=0.00016, ent_coef=0.00039, n_steps=1024 | 79.7% | 79.7% | 5% |
| 5 | 2 | learning_rate=0.00017, ent_coef=0.0012, n_steps=1024 | 76.3% | 76.3% | 1% |
| 6 | 8 | learning_rate=0.00011, ent_coef=0.00048, n_steps=512 | 63.0% | 63.0% | 4% |
| 7 | 1 | learning_rate=6.8e-05, ent_coef=0.0083, n_steps=512 | 47.0% | 47.0% | 1% |
| 8 | 6 | learning_rate=0.00096, ent_coef=0.0035, n_steps=2048 | 37.3% | 37.3% | 2% |
| 9 | 7 | learning_rate=0.0013, ent_coef=0.0022, n_steps=1024 | 30.0% | 30.0% | 3% |
| 10 | 3 | learning_rate=0.0011, ent_coef=0.052, n_steps=2048 | 25.0% | 25.0% | 6% |
| 11 | 4 | learning_rate=0.00061, ent_coef=0.00079, n_steps=256 | 14.3% | 14.3% | 2% |
| 12 | 10 | learning_rate=3.2e-05, ent_coef=0.0022, n_steps=2048 | 12.7% | 12.7% | 4% |
| 13 | 5 | learning_rate=0.0026, ent_coef=0.057, n_steps=512 | 7.0% | 7.0% | 7% |

**Winner: trial 9** (learning_rate=0.00034, ent_coef=0.002, n_steps=512), score 87.7%.
