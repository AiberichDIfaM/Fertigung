# Fertigung – RL for Flexible Job-Shop Scheduling

[![CI](https://github.com/AiberichDafM/Fertigung/actions/workflows/ci.yml/badge.svg)](https://github.com/AiberichDafM/Fertigung/actions/workflows/ci.yml)

A reinforcement learning based controller for job-shop manufacturing on generic production graphs.
Plants are described declaratively (part types, transformations, machines, orders); a discrete-time
simulation core executes them, and a MaskablePPO agent learns when to start which transformation on which
machine. A web UI and an HTTP API cover the whole loop: model the plant, set orders and reward weights,
train, simulate and compare against heuristic baselines.

## Quickstart

With Docker:

```sh
docker run -p 8000:8000 -v fertigung-data:/data ghcr.io/aiberichdafm/fertigung:latest
# or, from a checkout: docker compose up -d
```

Open http://localhost:8000. The reference plant and a model pretrained on it are available right away.

Locally, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run fertigung serve                             # UI and API on http://127.0.0.1:8000, API docs at /docs
uv run fertigung validate                          # check the reference plant (or pass a YAML/JSON file)
uv run fertigung simulate --policy pull            # run it with a heuristic and print KPIs
uv run fertigung evaluate --model reference        # compare the bundled model with the heuristics
uv run fertigung train my_plant.yaml --out models/my_plant
```

## Concepts

- A factory (*Anlage*) consists of **machines**. Each machine has a **machine type** that defines how
  many jobs it can run in parallel (**slots**) and which **transformations** it can perform.
- A transformation turns a multiset of input parts into one output part and takes a fixed number of ticks.
- The part types follow from the production graph:
  - **raw materials** are never produced; they are bought on demand at their `cost`,
  - **final products** are never consumed; they are sold on completion at their `price`,
  - everything else is an **intermediate** and waits in a shared WIP buffer of limited capacity.
- The controller decides at every tick which transformations to start. Intermediate inputs are taken
  from the buffer; if the buffer is full when a job finishes, the job blocks its slot until space frees up.
- **Orders** request a quantity of a final product by a deadline. Finished products are allocated to the
  open order with the earliest deadline; products without an order are sold at list price.

Manufacturing is therefore the progression of parts along the production graph, and the scheduling
challenge is to keep the limited buffer filled with parts that can actually be combined.

## Architecture

```
 browser UI (/ui)          CLI (fertigung ...)
        │                        │
        ▼                        │
 FastAPI app (api/) ─────────────┤
   │  SQLite store + model dirs  │
   │  training worker ──► spawned training process
   ▼                             ▼
 evaluation.py, heuristics.py   rl/  (JobShopEnv, behavior cloning, MaskablePPO, TrainedModel)
        │                        │
        └──────────┬─────────────┘
                   ▼
 core/  config schema → Plant → Simulation → Reward, validation
```

The simulation core has no Gym or web dependencies; the Gym env, the heuristics and the API are thin layers on
top of it.

## Plant configuration

Plants are YAML or JSON files (or edited in the UI); see
[`src/fertigung/configs/reference.yaml`](src/fertigung/configs/reference.yaml).

```yaml
name: example
buffer_capacity: 10
part_types:
  - {name: steel, cost: 10}
  - {name: bracket}
  - {name: frame, price: 60}
transformations:
  - {name: cut, inputs: [steel], output: bracket, duration: 2}
  - {name: weld, inputs: [bracket, bracket, steel], output: frame, duration: 4}
machine_types:
  - {name: saw, slots: 2, transformations: [cut]}
  - {name: welder, slots: 1, transformations: [weld]}
machines:
  - {name: saw-1, type: saw}
  - {name: welder-1, type: welder}
orders:
  - {product: frame, quantity: 5, deadline: 60}
reward:
  holding_cost: 0.5
  lateness: 5
```

### Reward

The optional `reward` section weights the terms of the RL reward. All weights are non-negative;
costs are subtracted.

| Key | Default | Meaning |
|---|---|---|
| `revenue` | 1.0 | per unit of sales revenue |
| `material_cost` | 1.0 | per unit of raw material spend |
| `holding_cost` | 0.0 | per intermediate part in the buffer per tick |
| `lateness` | 0.0 | per missing unit of an overdue order per tick |
| `idle` | 0.0 | per idle machine slot per tick |
| `shaping` | 0.0 | weight of potential-based shaping, potential = WIP valued at raw material cost |
| `gamma` | 0.99 | discount factor for shaping; also used as the agent's gamma |
| `scale` | 1.0 | multiplies the total reward |

Shaping moves the credit for material spend closer to the moment the WIP is created and does not change
the optimal policy. Summed over an episode it adds roughly `-(1 - gamma)` times the average WIP value per step,
so compare policies on the unshaped reward (`reward_unshaped` in evaluations).

### Validation

`fertigung validate`, `POST /plants/validate` and the UI report structural problems, e.g. transformations that
no machine can perform, parts that are treated as raw material only because their transformation is unavailable,
products that cannot be produced, transformations that need more intermediates than the buffer holds, and final
products that sell below their raw material cost.

## Reinforcement learning

`JobShopEnv` (`fertigung.rl.env`) is a flat Gymnasium env over the simulation:

- **Actions**: `0` advances time by one tick, every other action starts one (machine, transformation) pair.
  Invalid actions are masked (`action_masks()`); decisions where waiting is the only legal action are skipped.
- **Observation**: buffer counts per intermediate, parts in progress per output, running jobs per pair,
  free and blocked slots per machine, outstanding quantity and deadline slack per final product,
  elapsed time and buffer fill, all scaled to [-1, 1].
- **Episode**: `horizon` ticks, by default 1.2 x the last order deadline.

Training first imitates the `pull` heuristic (behavior cloning on expert-labelled rollouts, including
states reached by random deviations, with value pretraining on the observed returns) and then fine-tunes
with MaskablePPO. The imitated policy is the initial best model, so fine-tuning can only replace it with a
better one. Without pretraining (`--no-pretrain`), PPO on the reference plant settles on producing nothing:
random exploration fills the buffer with parts that cannot be combined long before any product is sold.

`fertigung train` writes a model directory with `model.zip` (best policy seen during evaluation),
`meta.json` (plant, horizon, training settings, final evaluation), `progress.csv` and checkpoints.
A model only fits plants with the same machines, transformations and final products.

Baselines in `fertigung.heuristics`: `pull` (explodes the bill of materials of the next due product and
produces only net requirements), `fifo` (oldest buffered part first) and `random`.

### Reference plant and model

The package ships the reference plant (15 transformations, 10 machines, two final products, four orders) and a
model trained on it (`src/fertigung/models/reference`, usable as `--model reference`). The server copies both into
its data directory on first start. Evaluation over the default horizon of 360 ticks:

| Policy | Reward | Unshaped reward | Profit | Products | Orders on time | Avg. buffer |
|---|---|---|---|---|---|---|
| **reference model** | **-2.95** | **5.74** | 1280 | **19** | 4/4 | **3.9** |
| pull | -6.26 | 5.04 | 1380 | 17 | 4/4 | 4.9 |
| fifo | -113.83 | -104.30 | -370 | 0 | 0/4 | 9.8 |
| random (5 episodes) | -131.56 | -109.51 | -876 | 0 | 0/4 | 9.9 |

The model makes 13 x fp1 and 6 x fp2 where pull makes 5 x fp1 and 12 x fp2, and keeps less work in progress.
It was trained with `fertigung train --timesteps 300000 --seed 0`; the best policy appeared after 40k PPO steps.
PPO fine-tuning does not improve on the imitated policy reliably: with seeds 1 and 2 no evaluation beat it, so
those runs end up with a pull-equivalent model. Training several seeds and keeping the best is worthwhile.

## Web UI

`fertigung serve` serves a browser UI at `/` (static HTML with Alpine.js, Chart.js and Cytoscape.js,
vendored in `src/fertigung/ui/vendor`, no build step):

- **Plant**: edit part types, transformations, machine types and machines; live validation and production graph
  with material cost per part; save, duplicate, import/export JSON.
- **Orders & reward**: orders and reward weights (fields and defaults come from the API schema).
- **Training**: start jobs with hyperparameters, follow progress and the reward curve, cancel.
- **Simulation**: run the current (also unsaved) plant with a heuristic or a trained model; KPIs, orders,
  reward components, machine schedule (Gantt) and event log; compare against the baselines.
- **Models**: evaluation summary, simulate, download, delete.

## HTTP API

Plants, training jobs, models and simulation results are stored in `$FERTIGUNG_DATA_DIR` (default `./data`,
`/data` in the image): a SQLite database plus one directory per model. Interactive documentation is served at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| GET, POST | `/plants` | list / create plants (body: plant config) |
| GET, PUT, DELETE | `/plants/{id}` | read / replace / delete a plant |
| POST | `/plants/validate`, `/plants/{id}/validate` | issues, part classification, material costs, graph edges |
| GET, POST | `/training-jobs` | list / queue a training job (`plant_id`, optional `training` settings) |
| GET, DELETE | `/training-jobs/{id}` | status, progress and reward curve / cancel |
| GET, DELETE | `/models/{id}` | model metadata incl. evaluation / delete |
| GET | `/models`, `/models/{id}/download` | list models / download as zip |
| POST | `/simulations` | run one episode with `pull`, `fifo`, `random` or `model`; KPIs, orders, Gantt bars, events |
| GET | `/simulations/{id}` | stored simulation result |
| POST | `/evaluations` | compare the heuristics and optionally a model on a plant |

Training jobs run one at a time in a separate process; progress is reported every 2048 steps.

### Authentication

Set `FERTIGUNG_API_KEY` to require the header `X-API-Key: <key>` on every endpoint except `/health`, the UI
files and the API docs. The UI asks for the key once and keeps it in the browser's local storage; in `/docs`
use *Authorize*. Without the variable the API is open, and `fertigung serve` warns when it listens on a
public interface.

```sh
FERTIGUNG_API_KEY=change-me docker compose up -d
curl -H "X-API-Key: change-me" http://localhost:8000/plants
```

## Docker

The image (`python:3.12-slim`, CPU-only PyTorch, about 1.5 GB) runs as a non-root user, keeps all state in the
volume at `/data` and has a health check on `/health`. `docker-compose.yml` builds it, maps port 8000 and passes
`FERTIGUNG_API_KEY` through.

## Development

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
```

`.github/workflows/ci.yml` runs on pull requests, pushes to `main` and `v*` tags:

1. **lint**: `ruff check`, `ruff format --check`
2. **test**: `pytest` (includes a short training run and an API training job)
3. **docker**: builds the image, starts it and checks health, UI and a simulation; on pushes it publishes
   to `ghcr.io/aiberichdafm/fertigung` as `latest` (main), `<version>` and `<major>.<minor>` (tags) and `sha-<commit>`.

Dependabot keeps the uv lockfile, the GitHub Actions and the Docker base image up to date.

```
src/fertigung/
├── core/          # config schema, plant model, simulation, reward, validation
├── rl/            # Gymnasium env, behavior cloning, training, model loading
├── api/           # FastAPI app, SQLite store, training worker
├── ui/            # browser UI served at /
├── configs/       # bundled reference plant
├── models/        # bundled reference model
├── heuristics.py  # baseline dispatch policies
├── evaluation.py  # KPIs per policy
└── cli.py
```

## License

[0BSD](LICENSE)
