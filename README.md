# Hierarchical RL for Flexible Job-Shop Scheduling

A modular framework combining low-level and high-level PPO agents to solve flexible job-shop scheduling problems via OpenAI Gymnasium environments.

## Overview
General purpose
- The project implements a reinforcement learning based solver for the job shop scheduling problem on generic manufacturing graphs.
- There are no restrictions on the shape of the manufacturing graph 
- Pretrained models help to quickly advance toward a working expert system

This repository implements:

1. **Data model** (`classes.py`):  
   - `PartType`, `Transformation`, `MachineType`, `Machine`, `Anlage` classes  
   - Fully configurable manufacturing graph (parts, transformations, machines)

2. **Low-Level Environment** (`flexible_jobshop_env.py`):  
   - Goal-conditioned Gymnasium env that exposes buffer & machine state  
   - Supports potential-based reward shaping to guide agent toward intermediate targets  
   - Action masking to prevent invalid moves

3. **High-Level Environment** (`hierarchical_env.py`):  
   - Wraps low-level envs to issue macro “subgoals”  
   - Observations identical to low-level (buffer + structure)  
   - Deadline-based reward shaping for user-defined product requests

4. **Training scripts**:  
   - `train_low_level.py`: Learns goal-conditioned low-level policies via MaskablePPO  
   - `train_high_level.py`: Trains high-level planner, with low-level agent fixed  
   - `train_joint.py`: Sequentially runs both training scripts  
   - `test_hierarchical.py`: Loads both policies to run a single hierarchical test episode  

5. **Orchestration** (`main.py`):  
   - Automates full pipeline: low-level → high-level → joint → test → full simulation  
   - Logs detailed event traces to `production_rl_event_log.txt`

## Repository Structure

├── classes.py
├── flexible_jobshop_env.py
├── hierarchical_env.py
├── manufacturing_structure.py # Example factory setup
├── train_low_level.py
├── train_high_level.py
├── train_joint.py
├── test_hierarchical.py
├── production_process_with_rl.py # Full hierarchical sim + logging
├── main.py
├── requirements.txt
└── README.md

## ⚙️ Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/AiberichDafM/Fertigung.git
   cd fertigung
2. Create a virtual environment and install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Key packages:
– gymnasium
– stable_baselines3 + sb3_contrib
– networkx, numpy

## Usage && Operation philosophy
1. Configure your factory
-  There is a default "factory" that the pretraining is based on.
You can either edit this factory (manufacturing_structure.py). Or replace it and define your own by replacing the file or importing a different file into the scripts. In the long run a generic API call for factory structures will be added to the project (maybe with a webinterface and -form).
2. The API follows a general philosophy:
-   a factory is an "anlage"
-   a anlage consists of "machines"
-   machines perform transformations
-   transformations transform a set of parts into a different part
3. In order to create a new factory structure the system requires definitions of part-types, transformations and machines
4. The manufacturing process is the progression along the manufacturing graph. Each machine has an input and output buffer and preferences, which kind of transformation it prefers to do. At each machine therefore some kind of parts are turned into a different one, put into the output buffer and from there transferred to the input buffer of a different machine.
5. The two agents perform different tasks: The low level agent fills input buffers. The high level agent assigns subgoals and by this determines the reward for the low level agent. The high level agent is turned rewarded according to the progression of products along the graph and finally meeting production schedules.
6. Primary task is to properly train the two agents. Once they are trained, you can give them orders with delivery dates and he tries to meet them according to their net worth (meaning more profitable products will be prioritised). 


📈 Customization
Subgoals & Deadlines
Define your desired product orders in hierarchical_env.py via the required_products list of dictionaries:

Reward Shaping
Adjust time penalties, potential-based rewards or deadline weights in the environments.

## Misc
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feat/YourFeature)

Commit your changes (git commit -am 'Add new feature')

Push to the branch (git push origin feat/YourFeature)

Open a Pull Request

📄 License
This project is released under the MIT License. See LICENSE for details.


Feel free to tweak any section to match your final file-names, deadlines, or example structure!
