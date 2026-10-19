# Random search: dqn_mlp

Agents: dqn_mlp. Sampler seed 2027; trial 0 = defaults, trials 1-8 random. Validation seeds [0, 1, 2], 300,000 steps per run.

Search space: `dqn_learning_rate` log-uniform [3e-05, 0.003]; `dqn_target_update_interval` one of {250, 500, 1000, 2000, 5000}; `dqn_exploration_fraction` uniform [0.05, 0.5]; `dqn_exploration_final_eps` one of {0.01, 0.05, 0.1}.

Score = held-out fruit rate (sampled actions) averaged over validation seeds.

| Rank | Trial | Settings | dqn_mlp | Score | Poison |
|---|---|---|---|---|---|
| 1 | 7 | dqn_learning_rate=0.00045, dqn_target_update_interval=5000, dqn_exploration_fraction=0.46, dqn_exploration_final_eps=0.1 | 7.3% | 7.3% | 2% |
| 2 | 0 | defaults | 7.0% | 7.0% | 1% |
| 3 | 8 | dqn_learning_rate=3.6e-05, dqn_target_update_interval=5000, dqn_exploration_fraction=0.49, dqn_exploration_final_eps=0.01 | 6.7% | 6.7% | 0% |
| 4 | 5 | dqn_learning_rate=0.0019, dqn_target_update_interval=2000, dqn_exploration_fraction=0.48, dqn_exploration_final_eps=0.01 | 4.7% | 4.7% | 2% |
| 5 | 3 | dqn_learning_rate=0.00012, dqn_target_update_interval=250, dqn_exploration_fraction=0.3, dqn_exploration_final_eps=0.05 | 4.3% | 4.3% | 0% |
| 6 | 6 | dqn_learning_rate=0.00093, dqn_target_update_interval=500, dqn_exploration_fraction=0.13, dqn_exploration_final_eps=0.1 | 4.3% | 4.3% | 2% |
| 7 | 1 | dqn_learning_rate=3.1e-05, dqn_target_update_interval=250, dqn_exploration_fraction=0.087, dqn_exploration_final_eps=0.05 | 4.0% | 4.0% | 1% |
| 8 | 4 | dqn_learning_rate=4.5e-05, dqn_target_update_interval=2000, dqn_exploration_fraction=0.22, dqn_exploration_final_eps=0.01 | 3.7% | 3.7% | 0% |
| 9 | 2 | dqn_learning_rate=0.0003, dqn_target_update_interval=5000, dqn_exploration_fraction=0.37, dqn_exploration_final_eps=0.05 | 1.7% | 1.7% | 1% |

**Winner: trial 7** (dqn_learning_rate=0.00045, dqn_target_update_interval=5000, dqn_exploration_fraction=0.46, dqn_exploration_final_eps=0.1), score 7.3%.
