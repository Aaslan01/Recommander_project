from plot_results import plot_rmse_comparison
from experiment_runner import run_latent_factor_experiment

factors, mf_rmses, hybrid_rmses = run_latent_factor_experiment()
plot_rmse_comparison(mf_rmses, hybrid_rmses, factors)
