## Meta-Cultural Competency in action


### Initialization

```
pip install -r requirements.txt
```

**Gemini API key** (for `executor_gemini.py`): create a `config.json` in the repository root:

```json
{
    "GEMINI_API_KEY": ""
}
```

**Hugging Face token** (for open-weight models; gated repos such as meta-llama need an account with approved access). Provide it in one of these ways:

- `export HF_TOKEN=hf_...`
- a `.env` file containing `HF_TOKEN=hf_...`, in the repository root or its parent folder (`run_ablations.sh` and `download_model.sh` load it)
- `huggingface-cli login`, which saves the token to `~/.cache/huggingface/token`

`config.json` and `.env` are gitignored; never commit them.

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

This writes `user_simulator_output.json`, holding every user's answers to the 15 questions.

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

Download the model first. Models are cached in `../hf_models` (override with `HF_HUB_CACHE`):

Example: download Llama-3.1-8B-Instruct (~16 GB) and run it

```
bash download_model.sh                                          # update the script to download a different model
MODEL=meta-llama/Llama-3.1-8B-Instruct bash download_model.sh
```

Then run one configuration:

```
python executor.py --condition demographics_context --schema 1 --movie_visibility unseen
python executor.py --model meta-llama/Llama-3.1-8B-Instruct --condition none --schema 2 --movie_visibility seen
```

`--condition` accepts `none`, `demographics_only`, `context_only` or `demographics_context`. `--model` takes any Hugging Face model id (default `meta-llama/Llama-3.1-70B-Instruct`). Llama-3 models use a fixed Llama-3 chat template; other models use their tokenizer's chat template. Each model writes to its own folder, `rl_explore_exploit_results/<model_dir>/`, named after the model id (e.g. `llama_3_1_70b_instruct`); set `--model_dir` to choose the name.

To run the full grid (4 conditions × 2 schemas × 2 visibilities), use `run_ablations.sh`. Pass it one worker's index, and give every worker the same `GPU_GROUPS`:

```
GPU_GROUPS="0,1,2,3" bash run_ablations.sh 0                  # one worker on GPUs 0-3
GPU_GROUPS="0,1,2,3 4,5,6,7" bash run_ablations.sh 0          # two workers, each in its own shell
GPU_GROUPS="0,1,2,3 4,5,6,7" bash run_ablations.sh 1
MODEL=meta-llama/Llama-3.1-8B-Instruct GPU_GROUPS="0" bash run_ablations.sh 0
```

- Each GPU group is one worker, and each worker takes a separate share of the users (a shard). If workers are given different `GPU_GROUPS`, their shards overlap and users are run twice.
- The number of GPUs in a group sets vLLM's tensor parallelism, which must be 1, 2, 4 or 8.
- Workers skip users that already have a `summary.json`, so an interrupted grid can be relaunched with the same command.
- Optional environment variables: `MODEL` (default `meta-llama/Llama-3.1-70B-Instruct`), `BATCH_SIZE` (users per worker at once, default 64), `PYTHON_BIN`, `HF_HUB_CACHE`, `LOG_DIR` (default `logs_<model name>`).

To shard a single configuration by hand, pass `--shard i --num_shards N` to `executor.py`.

#### Output

Every run writes one directory per user and run tag, under a folder per model:

```
rl_explore_exploit_results/<model>/user_<id>/<run_tag>/
    turn_log.json      # per-turn decisions, questions, answers and predictions
    predictions.csv    # exploit predictions against ground truth
    summary.json       # final budget, accuracy and decision counts
```

### Evaluation

1. Export trajectories for each model:

   ```
   python export_trajectories.py --results-root rl_explore_exploit_results/<model> \
       --model-name <model> --out trajectories/<file name>
   ```

   The data is stored in one row for each user at a given setting. Each row contains the decision sequence (`R` = exploit, `R!` = budget-forced exploit, a number = the utility of an explore slot) and the e_{l}@5 / e_{l}@10 numerator and denominator for each discount factor γ in `--gammas`.

   Steps 2 and 3 read trajectories from `trajectories/` in the repository root, using the file names listed in `TRAJECTORIES` in `bootstrap_overall.py`. Save each export there under its listed name.

2. Compute the metrics:

   ```
   python make_normalized_metrics.py                                 # seen, expensive, γ = 0.9
   python make_normalized_metrics.py --visibility unseen --cost cheap --gamma 0.5
   ```

   All metrics lie in [0, 1], and higher is better:

   - **Sens**: spread of the exploration rate across context levels.
   - **Qual@5**: mean Exp@5 across context levels.
   - **Align**: whether exploration falls as context grows while question quality stays independent of context.

   Outputs: per-metric CSVs and `model_ranks.csv` (which ranks models on each metric and on their mean) in `metrics/`, and figures in `figures/` (PDF and PNG).

3. Estimate uncertainty and test the ablations:

   ```
   python bootstrap_overall.py
   python bootstrap_overall.py --n-boot 2000 --correction holm --traj-dir path/to/trajectories
   ```

   Covers the base configuration (seen, expensive, γ = 0.9, K = 5) and one ablated cell per factor: unseen, cheap, γ = 0.7 / 0.5 / 0.3, and K = 10. It resamples users (5,000 replicates by default) with the same draws for every cell and model. It reports 95% CIs for Sens, Qual@K, Align and Overall, each model's rank with its interval, and pairwise model tests. For each ablation it also tests each model's change in Overall and rank against the baseline (Benjamini-Hochberg by default) and gives Kendall τ between the two rankings.

   Outputs, in `bootstrap_with_rank/`: per-cell `bootstrap_overall_*.csv`, `bootstrap_pairs_*.csv` and `bootstrap_draws_*.csv`, plus `ablation_models.csv`, `ablation_summary.csv` and `ablation_rank_tables.tex`.


