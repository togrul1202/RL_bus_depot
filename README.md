# RL Bus Depot

A Reinforcement Learning project for optimizing the operation of hybrid bus depots using intelligent agents.

## Overview

This project develops and trains RL agents to efficiently manage hybrid bus depot operations. The agent makes decisions about bus scheduling, charging station allocation, and fuel station operations to optimize depot efficiency. The environment is built using the Gymnasium framework, and models are trained using Stable Baselines3 algorithms (PPO, A2C, or Masked PPO).

## Features

- **Hybrid Bus Simulation**: Models buses with both electric and fuel-based propulsion
- **Dynamic Scheduling**: RL agents learn optimal bus scheduling and resource allocation
- **Charging Infrastructure**: Configurable charging stations with different charge speeds (regular and fast-charging)
- **Fueling Stations**: Separate fuel management for non-electric vehicles
- **Employee Management**: Support for employee allocation and shift management
- **Action Masking**: Optional masking to constrain agent actions to valid operations
- **Multiple RL Algorithms**: Support for PPO, A2C, and Masked PPO from Stable Baselines3

## Project Structure

```
RL_bus_depot/
├── gym_depot/              # Custom Gymnasium environment
│   ├── config.yaml        # Configuration parameters for depot simulation
│   ├── utils.py           # Utility functions
│   ├── __init__.py
│   └── envs/
│       ├── depot_env.py   # Main environment implementation
│       └── __init__.py
├── mppo_model/            # Trained model storage
├── results/               # Training results and logs
├── train.py               # Script to train RL agents
├── evaluate.py            # Script to evaluate trained models
├── check_env.py           # Environment validation utility
├── test.py                # Testing scripts
├── manual.py              # Manual control interface
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration

All simulation parameters are defined in `gym_depot/config.yaml`:

- **Bus Configuration**: Number of buses, initial state randomization
- **Charging Stations**: Number, charging speeds, charge times
- **Fuel Stations**: Number, fueling speed, fuel times
- **Energy Levels**: Number of discrete fuel/charge levels
- **Employees**: Number of depot staff
- **Time**: Conversion between minutes and timesteps

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RL_bus_depot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

Train a new RL agent using:
```bash
python train.py
```

Configuration for training (algorithm, learning parameters, etc.) is specified in the config files.

### Evaluating a Model

Evaluate a trained model:
```bash
python evaluate.py
```

### Manual Control

Test the environment with manual control:
```bash
python manual.py
```

### Environment Validation

Check the environment setup:
```bash
python check_env.py
```

## Dependencies

- **gymnasium**: RL environment framework
- **stable-baselines3**: RL algorithms (PPO, A2C)
- **sb3-contrib**: Additional algorithms (Masked PPO)
- **numpy**: Numerical computing
- **PyYAML**: Configuration file parsing
- **matplotlib**: Visualization
- **pygame**: Rendering
- **tensorboard**: Training visualization

## Results

Trained models and training logs are stored in the `results/` directory, organized by training run with timestamps.



