"""Persist epoch metrics and loss plots."""
import csv

def save_history(run_dir, history, model_name="efficientphys_c"):
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "valid_loss", "learning_rate", "seconds"])
        writer.writeheader()
        writer.writerows(history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for name in ("train_loss", "valid_loss"):
        ax.plot([row["epoch"] for row in history], [row[name] for row in history], marker="o", label=name)
    ax.set(xlabel="Epoch", ylabel="MSE loss", title=f"{model_name} / UBFC subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss.png", dpi=150)
    plt.close(fig)
