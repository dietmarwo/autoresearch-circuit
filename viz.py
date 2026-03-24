"""
Visualisation — plot best topologies, phenotype traces, and search progress.

Produces publication-ready matplotlib figures showing:
  1. Best topology traces (SSA time-series for each gene)
  2. Score progression over iterations
  3. Motif summary bar chart
  4. Network diagrams for top topologies
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

from grammar import Topology
from model_builder import build_model
from archive import Archive
import config as cfg


def plot_topology_trace(topology: Topology, params: np.ndarray,
                        seed: int = 42, t_end: float = cfg.SIM_T_END,
                        save_path: str = None, title: str = None):
    """
    Simulate one SSA trajectory and plot gene copy numbers over time.
    """
    model = build_model(topology, params, t_end=t_end)
    result = model.run(solver=cfg.SIM_SOLVER, seed=seed)

    fig, ax = plt.subplots(figsize=(10, 4))
    time_arr = np.array(result["time"])
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for gene, color in zip(cfg.GENES, colors):
        vals = np.array(result[gene])
        ax.plot(time_arr, vals, label=gene, color=color, linewidth=1.0, alpha=0.85)

    ax.set_xlabel("Time")
    ax.set_ylabel("Copy number")
    ax.set_title(title or f"SSA trace: {topology.to_label()}")
    ax.legend(loc="upper right")
    ax.set_xlim(0, t_end)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved trace plot: {save_path}")
    else:
        plt.close(fig)
    return fig


def plot_score_progression(archive: Archive, save_path: str = None):
    """Plot score vs. iteration for all evaluated topologies."""
    if not archive.results:
        return

    iters = [r.iteration for r in archive.results]
    scores = [r.score for r in archive.results]

    # Running best
    running_best = []
    best_so_far = 0.0
    for s in scores:
        best_so_far = max(best_so_far, s)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(iters, scores, alpha=0.5, s=20, color="#90A4AE", label="individual")
    ax.plot(range(len(running_best)), running_best, color="#E91E63",
            linewidth=2, label="running best")

    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Score")
    ax.set_title("Search Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved progression plot: {save_path}")
    else:
        plt.close(fig)
    return fig


def plot_top_scores(archive: Archive, top_k: int = 10, save_path: str = None):
    """Horizontal bar chart of the top-K topology scores."""
    results = archive.top_k(top_k)
    if not results:
        return

    labels = [r.topology.to_label()[:50] for r in results]
    scores = [r.score for r in results]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(results))))
    y_pos = range(len(results))
    bars = ax.barh(y_pos, scores, color="#42A5F5", edgecolor="#1565C0", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title(f"Top {len(results)} Topologies")
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved top-scores plot: {save_path}")
    else:
        plt.close(fig)
    return fig


def plot_network_diagram(topology: Topology, save_path: str = None,
                         title: str = None):
    """
    Draw a simple 3-node regulatory network diagram.
    Nodes are placed in a triangle; edges are coloured arrows.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Node positions (equilateral triangle)
    positions = {
        "A": np.array([0.0, 1.0]),
        "B": np.array([-0.87, -0.5]),
        "C": np.array([0.87, -0.5]),
    }

    # Draw nodes
    for gene, pos in positions.items():
        circle = plt.Circle(pos, 0.22, color="#BBDEFB", ec="#1565C0",
                            linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], gene, ha="center", va="center",
                fontsize=16, fontweight="bold", zorder=11)

    # Draw edges
    for idx, val in enumerate(topology.edges):
        if val == 0:
            continue
        src_idx, tgt_idx = cfg.EDGE_INDEX_MAP[idx]
        src_gene = cfg.GENES[src_idx]
        tgt_gene = cfg.GENES[tgt_idx]

        if src_idx == tgt_idx:
            # Self-loop: draw a small arc above the node
            pos = positions[src_gene]
            color = "#4CAF50" if val == 1 else "#F44336"
            symbol = "→" if val == 1 else "⊣"
            ax.annotate(
                symbol, xy=(pos[0] + 0.15, pos[1] + 0.25),
                fontsize=14, color=color, fontweight="bold",
            )
            continue

        src_pos = positions[src_gene]
        tgt_pos = positions[tgt_gene]

        # Shorten arrow to not overlap with circles
        direction = tgt_pos - src_pos
        dist = np.linalg.norm(direction)
        unit = direction / dist
        start = src_pos + unit * 0.28
        end = tgt_pos - unit * 0.28

        color = "#4CAF50" if val == 1 else "#F44336"
        head = ">" if val == 1 else "|"
        style = f"-{head}"

        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="->" if val == 1 else "-|>",
                color=color, lw=2.0,
                connectionstyle="arc3,rad=0.15",
            ),
        )

    ax.set_title(title or topology.to_label(), fontsize=10, pad=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved network diagram: {save_path}")
    else:
        plt.close(fig)
    return fig


def generate_all_plots(archive: Archive, output_dir: str = "results"):
    """Generate all standard plots for a completed search."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("Generating plots...")

    # Score progression
    plot_score_progression(archive, save_path=str(out / "score_progression.png"))

    # Top scores bar chart
    plot_top_scores(archive, top_k=10, save_path=str(out / "top_scores.png"))

    # Traces and network diagrams for top-5
    for rank, r in enumerate(archive.top_k(5), 1):
        if r.params is not None and r.score > 0.01:
            plot_topology_trace(
                r.topology, r.params,
                save_path=str(out / f"trace_top{rank}.png"),
                title=f"#{rank} (score={r.score:.3f}): {r.topology.to_label()}",
            )
            plot_network_diagram(
                r.topology,
                save_path=str(out / f"network_top{rank}.png"),
                title=f"#{rank}: {r.topology.to_label()}",
            )

    print(f"All plots saved to {out}/")
