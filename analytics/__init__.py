"""Analytics: evaluation metrics, agent comparison, plots and result export."""
from analytics.metrics import (  # noqa: F401
    EPISODE_COLUMNS, aggregate_over_seeds, bootstrap_ci, evaluate_agent, evaluate_generalization,
    load_train_log, moving_average, run_episode, summarize, wilson_interval,
)
