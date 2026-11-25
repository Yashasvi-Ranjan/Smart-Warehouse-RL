# Smart Warehouse Robot (Reinforcement Learning)

A robot learns to pick up a package, deliver it to a dock and recharge its battery in a
procedurally generated warehouse. The project compares Q-learning, DQN and PPO against a random policy
and a privileged A* planner, and ships a pygame dashboard for simulation, training, evaluation and analytics.


Quick start (Python 3.15 virtual environment in `.venv`):

```bash
python3.15 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --smoke     # headless self-test
.venv/bin/python main.py             # open the dashboard
.venv/bin/python -m pytest tests     # run the test suite
```
