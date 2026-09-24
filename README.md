# Fertigung – RL for Flexible Job-Shop Scheduling

A reinforcement learning based controller for job-shop manufacturing on generic production graphs.
Plants are described declaratively (part types, transformations, machines, orders); a discrete-time
simulation core executes them, and RL agents learn when to start which transformation on which machine.

> Work in progress: the project is being converted from an experiment into a product
> (API, UI, Docker image). The previous hierarchical PPO prototype lives in `fertigung.legacy`
> until the new RL layer replaces it.

## Operating philosophy

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

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run fertigung validate                       # check the bundled reference plant
uv run fertigung simulate --ticks 300           # run it with the pull heuristic, print KPIs
uv run fertigung simulate my_plant.yaml --events
```

Development:

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
```

## Plant configuration

Plants are YAML or JSON files; see [`src/fertigung/configs/reference.yaml`](src/fertigung/configs/reference.yaml).

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
| `gamma` | 0.99 | discount factor for shaping; should match the agent's gamma |
| `scale` | 1.0 | multiplies the total reward |

Shaping moves the credit for material spend closer to the moment the WIP is created and does not change
the optimal policy. Summed over an episode it adds roughly `-(1 - gamma)` times the average WIP value per step,
so compare policies on the unshaped components. `fertigung simulate` prints the breakdown.

### Validation

`fertigung validate` reports structural problems, e.g. transformations that no machine can perform,
products that cannot be produced, transformations that need more intermediates than the buffer holds,
and final products that sell below their raw material cost.

## Layout

```
src/fertigung/
├── core/          # config schema, plant model, simulation, validation
├── configs/       # bundled reference plant
├── heuristics.py  # baseline dispatch policies
├── cli.py
└── legacy/        # previous hierarchical PPO prototype (to be replaced)
```

## License

[0BSD](LICENSE)
