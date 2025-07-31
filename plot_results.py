
import matplotlib.pyplot as plt

def plot_rmse_comparison(mf_rmses, hybrid_rmses, factors):
    plt.plot(factors, mf_rmses, label="Matrix Factorization", marker='o')
    plt.plot(factors, hybrid_rmses, label="Hybrid Recommender", marker='o')
    plt.xlabel("Latent Factors")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Latent Factors")
    plt.legend()
    plt.grid(True)
    plt.savefig("rmse_comparison.png")
    plt.show()
