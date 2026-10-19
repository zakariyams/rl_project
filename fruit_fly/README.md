## Reinforcement Learning Final Project

Student group YZM

Group members:
- Mohammad Abbas
- Yoseph Naoom
- Zakariya Mohamed Shafe

# Fruit-fly odor navigation with a connectome-constrained RL agent

Does the wiring of a real fly brain help a reinforcement-learning agent? A
memoryless fly in a 2D arena must find a fruit and avoid a poison using only
smell: 21 receptor channels on each side of its head. We compare a PPO agent
whose actor is the larval *Drosophila* antennal-lobe circuit (Berck et al. 2016)
with the same circuit randomly rewired, a conventional MLP, and DQN, plus a random
agent and two hand-coded rules as baselines.

| Agent | Algorithm | Policy network |
|---|---|---|
| PPO-MLP (reference) | PPO | 64x64 MLP |
| PPO-Bio | PPO | connectome actor, biological wiring |
| PPO-Random | PPO | connectome actor, degree-preserving shuffled wiring |
| DQN-MLP | DQN | 64x64 MLP |
| Random agent, odor rule, informed rule | none | hand-coded baselines |

The final results are in
[`data/results/final_r2+tune_connectome_confirm+tune_ppo_mlp_confirm+tune_dqn_mlp_confirm/summary.md`](data/results/final_r2+tune_connectome_confirm+tune_ppo_mlp_confirm+tune_dqn_mlp_confirm/summary.md),
with figures next to it and in [`data/results/figures/`](data/results/figures/).

## Code structure

```
environment/   the POMDP (report section 3)
  config.py        FruitEnvConfig: arena, actions, odor model, rewards (defaults = the task in the report)
  fruit_env.py     FruitEnv (Gymnasium): hidden state, reset, step dynamics, rewards, termination
  odor.py          odor concentration and the 2 x 21 receptor-channel observation
  receptors.py     synthetic receptor response profiles of the fruit and the poison odor
connectome/    the biological network (report section 4.3)
  berck2016.py     reads the published larval antennal-lobe connectivity tables
  graph.py         the 140-neuron graph used as actor; ORN inputs and projection-neuron readout
  randomize.py     degree-preserving shuffle of that graph (the random-wiring control)
  policy.py        ConnectomeNet / ConnectomePolicy: the circuit as a trainable PPO actor
agents/        every agent behind one interface (BaseAgent: reset, act)
  random_agent.py, rule_agents.py (odor rule, informed rule), trained_agent.py (a trained model)
rl/            learning
  config.py        TrainingConfig (all hyperparameters) and the agent IDs
  models.py        builds and loads the PPO / DQN model of each agent
  sensory.py       fixed log-odor input stage shared by all networks (+ sensory_torch.py)
  training.py      train_run: training, checkpoints, logging, then held-out evaluation
  evaluation.py    the held-out test protocol, identical for every agent
  tuning.py        random-search spaces, sampler seeds and ranking (report section 5.4)
  storage.py       run directories under data/runs/ and their files
experiments/   command-line scripts
  train.py         train agents
  tune.py          hyperparameter search: plan, run, select, confirm
  results.py       report tables, learning curves and held-out figure
  figures.py       trajectory figure and tuning figure
  runs.py          list saved runs; zip experiments as a snapshot
tests/         pytest suite
data/raw/berck2016/   connectivity tables (eLife 14859 supplementary files 1 and 2, CC BY 4.0)
data/results/         tables, rankings and figures of the report
```

Where the problem formulation lives in the code:

| Report | Code |
|---|---|
| State, initial-state distribution | `EnvironmentState`, `FruitEnv.reset` in `environment/fruit_env.py` |
| Observation (eq. 3-5) | `OdorField.read` in `environment/odor.py`, `odor_profiles` in `environment/receptors.py` |
| Actions and dynamics (eq. 6-7) | `FruitEnv.step` |
| Reward (eq. 8) | `FruitEnv.step`; values in `FruitEnvConfig` |
| Log-odor input stage | `rl/sensory.py` |
| Hyperparameters (defaults) | `TrainingConfig` in `rl/config.py` |
| Tuning search spaces | `SEARCHES` in `rl/tuning.py` |
| Evaluation protocol | `rl/evaluation.py` |

## Setup

Python 3.13 on Windows was used for every run (`requirements-lock.txt` lists the
exact versions; `requirements.txt` the direct dependencies).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest                     # about 20 seconds; includes checks of the connectome tables
```

The connectivity tables are included in `data/raw/berck2016/`. They are
supplementary files 1 and 2 of Berck, Khandelwal et al., "The wiring diagram of
a glomerular olfactory system", eLife 5:e14859 (2016),
https://doi.org/10.7554/eLife.14859, licensed CC BY 4.0.

## Reproducing the results

Every command below is run from the repository root. Each training run gets a
directory under `data/runs/<id>/` (config, training log, model, evaluation; see
`rl/storage.py`). Runs are single-threaded on the CPU, so several commands can
run in parallel in separate terminals.

**1. Final runs** (Table 2, default hyperparameters): 4 agents x 5 test seeds,
1,000,000 steps each, evaluated on 100 held-out episodes per seed.

```powershell
python -m experiments.train --agent ppo_mlp          --seed 100 101 102 103 104 --experiment final_r2
python -m experiments.train --agent ppo_bio          --seed 100 101 102 103 104 --experiment final_r2
python -m experiments.train --agent ppo_random_graph --seed 100 101 102 103 104 --experiment final_r2
python -m experiments.train --agent dqn_mlp          --seed 100 101 102 103 104 --experiment final_r2
```

**2. Hyperparameter tuning** (Table 4): random search on the validation seeds,
then the winning configuration retrained on the test seeds. The trials are drawn
from fixed sampler seeds in `rl/tuning.py`; `plan` prints them.

```powershell
python -m experiments.tune plan ppo_mlp
python -m experiments.tune run ppo_mlp --workers 4       # train all trials
python -m experiments.tune select ppo_mlp                # rank -> data/results/tune_ppo_mlp/
python -m experiments.tune confirm ppo_mlp --workers 4   # winner on test seeds -> tune_ppo_mlp_confirm
```

The same four commands with `dqn_mlp` and `connectome` (Bio and Random tuned
together) complete the search. `run` and `confirm` reuse runs that already exist,
so an interrupted search is continued by running the command again.

**3. Tables and figures** (Tables 2-3 and Figures 1-3; nothing is trained):

```powershell
python -m experiments.results final_r2 tune_connectome_confirm tune_ppo_mlp_confirm tune_dqn_mlp_confirm
python -m experiments.figures tuning
python -m experiments.figures trajectories --experiment final_r2 --seed 100
```

The trained runs are not stored in git (64 MB without the DQN replay buffers;
26 MB zipped). This packs all of them, tuning runs included, into a zip in
`data/snapshots/`. Unzip its `runs/` folder into `data/runs/` to restore them:

```powershell
python -m experiments.runs snapshot final_r2 tune_ppo_mlp tune_dqn_mlp tune_connectome tune_ppo_mlp_confirm tune_dqn_mlp_confirm tune_connectome_confirm
```

### Seeds and reproducibility

- Test seeds 100-104 are used for the final and confirmation runs, and
  validation seeds 0-2 for tuning (0-1 for the connectome search).
- The training seed fixes the network initialization, the action sampling, the
  training arena's episode sequence, the fruit/poison odor pair and, for
  PPO-Random, the shuffled graph.
- Held-out episode `i` of training seed `s` is reset with seed
  `1_000_000_000 + 1000 * s + i`, a range that training never uses. All agents of
  a seed, including the random agent and the rules, play exactly these episodes.
- With the same library versions on the same machine, rerunning a configuration
  gives bitwise-identical training logs and evaluations. Other platforms or
  library versions may give slightly different numbers.
- The git tag `experiments-final` marks the code exactly as it was when the runs
  were trained. The code has been reorganized since then without changing how
  agents are trained or evaluated.
