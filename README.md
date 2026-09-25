## Meta-Cultural Competency in action



### Initialization

```
pip install -r requirements.txt
```

Create a `config.json` in the repository root holding your Hugging Face token (needed for gated models like Llama) and your Gemini API key:

```json
{
    "hf_token": "",
    "GEMINI_API_KEY": ""
}
```

All scripts should be run from the repository root. They read `config.json`, the dataset and the simulator output by relative path.

### Experimental setup

**Conditions** (what the agent is shown about the user):

| Condition | Demographics | Rated history |
|---|---|---|
| `demographics_only` | ✓ | |
| `context_only` | | ✓ |
| `demographics_context` | ✓ | ✓ |
| `none` | | |

Under `none`, the agent only learns about the user through the answers to its own questions.

**Schemas** (reward structure):

| Schema | Explore cost | Exploit gain |
|---|---|---|
| 1 (expensive) | −5 | +2 |
| 2 (cheap) | −2 | +2 |

**Movie visibility**: `unseen` or `seen`.

Fixed across runs: 20 turns, 4 movies per turn, initial budget 10, temperature 1.0, 10 history movies (diverse) for the history conditions.

### Running experiments

#### 1. User simulation

Creates the simulated user set. Run the user simulator first, as it is required for both executors.

```
python user_simulator.py
```

This writes `user_simulator_output.json`, holding every user's answers to the 15 questions:

- Q1–Q6: demographics
- Q7–Q12: preferences
- Q13–Q15: distractors


#### 2a. Gemini

```
bash run_ablations_gemini.sh                                  # schema 1, gemini-2.5-flash, all conditions
bash run_ablations_gemini.sh --schema 2                       # schema 2
bash run_ablations_gemini.sh --model pro --thinking medium    # gemini-2.5-pro with thinking
bash run_ablations_gemini.sh --condition context_only         # single condition
bash run_ablations_gemini.sh --rerun                          # restart every run from turn 1
```

Each call runs the selected conditions under both visibilities. It submits one batch job per ablation per step and polls them until done (`--poll_interval`, default 30s).

Runs resume automatically:
- users with a `summary.json` are skipped;
- users with a `checkpoint.json` continue from their next turn.

Set `PYTHON` to use a specific interpreter. Logs go to `ablation_logs/<model>/`. The output is saved to `rl_explore_exploit_results/<model>/`

#### 2b. Open-weight models (vLLM)

```
python executor.py --condition demographics_context --schema 1 --movie_visibility unseen
```

`--condition` accepts `demographics_only`, `context_only` or `demographics_context`.

#### Output

Every run writes one directory per user and run tag. Gemini results go under `rl_explore_exploit_results/<model>/`; `executor.py` writes directly under `rl_explore_exploit_results/`.

```
rl_explore_exploit_results/[<model>/]user_<id>/<run_tag>/
    turn_log.json      # per-turn decisions, questions, answers and predictions
    predictions.csv    # exploit predictions against ground truth
    summary.json       # final budget, accuracy and decision counts
```

### Evaluation

1. Export trajectories for each model:

   ```
   python export_trajectories.py --results-root rl_explore_exploit_results/<model> \
       --model-name <model> --out traj_<model>.csv
   ```

   The data is stored in one row for each user at a given setting. Each row contains the decision sequence (`R` = exploit, `R!` = budget-forced exploit, a number = the utility of an explore slot) and the e_{l}@5 / e_{l}@10 numerator and denominator for each discount factor γ in `--gammas`.

   Steps 2 and 3 read trajectories from `../trajectories/`, using the file names listed in `TRAJECTORIES` in `bootstrap_overall.py`. Save each export there under its listed name.

2. Compute the metrics:

   ```
   python make_normalized_metrics.py                                 # seen, expensive, γ = 0.9
   python make_normalized_metrics.py --visibility unseen --cost cheap --gamma 0.5
   ```

   All metrics lie in [0, 1], and higher is better:

   - **Sens**: spread of the exploration rate across context levels.
   - **Qual@5**: mean Exp@5 across context levels.
   - **Align**: whether exploration falls as context grows while question quality stays independent of context.

   Outputs: per-metric CSVs, figures (`figures/`, PDF and PNG), and `model_ranks.csv`, which ranks models on each metric and on their mean.

3. Estimate uncertainty and compare models:

   ```
   python bootstrap_overall.py                                       # seen, expensive, γ = 0.9, Qual@5
   python bootstrap_overall.py --gamma 0.5 --cost -2 --k 10 --correction fdr_bh
   ```

   Resamples users (5,000 replicates by default), applying the same draw to every model. Reports a 95% CI for Sens, Qual@K, Align and Overall, and each model's median rank with a 95% interval on Overall. Also tests every model pair on Overall, correcting for multiple comparisons (Holm by default).

   Outputs: `bootstrap_overall_*.csv` (one row per model), `bootstrap_pairs_*.csv` (one row per model pair), and `bootstrap_draws_*.csv` (every replicate's values).


