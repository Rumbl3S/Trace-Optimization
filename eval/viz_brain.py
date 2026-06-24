"""viz_brain.py — Live brain state visualization. Four-panel dashboard.

  Panel 1 — Neuron Graph:    Markov thought-state nodes (colored by failure rate),
                              transition edges (thickness = frequency). Once the
                              Markov chain activates (≥30 chunks stored), each node
                              is a k-means cluster of semantically similar reasoning
                              chunks. Raw scatter shown before Markov activates.

  Panel 2 — Trajectory Map:  PCA-2D of all stored chunk embeddings. Each run is
                              a connected polyline: ● = start, × = end.
                              Green = pass, red = fail.

  Panel 3 — Score Timeline:  Pass/fail bars per task + cumulative accuracy line.

  Panel 4 — Fire Report:     Brain fires per task + cumulative fire rate vs 30% target.

Usage::
    from viz_brain import BrainViz
    viz = BrainViz()
    viz.update(brain, results, fire_counts)
    viz.save(Path("eval/results/brain_overview.png"))
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


_DARK   = "#0d0d1a"
_PANEL  = "#16213e"
_GRID   = "#2a2a4a"
_PASS   = "#00d26a"
_FAIL   = "#ff4444"
_FIRE   = "#ffaa00"
_TEXT   = "#e0e0ff"
_DIM    = "#55557a"


class BrainViz:
    """Live-updating 4-panel brain dashboard. Call update() + save() after each task."""

    def __init__(self):
        self.fig = plt.figure(figsize=(16, 11), facecolor=_DARK)
        gs = self.fig.add_gridspec(
            2, 2, hspace=0.38, wspace=0.30,
            left=0.06, right=0.97, top=0.93, bottom=0.07,
        )
        self.ax_neurons = self.fig.add_subplot(gs[0, 0])
        self.ax_traj    = self.fig.add_subplot(gs[0, 1])
        self.ax_scores  = self.fig.add_subplot(gs[1, 0])
        self.ax_fires   = self.fig.add_subplot(gs[1, 1])
        for ax in self._axes():
            ax.set_facecolor(_PANEL)
        self.fig.suptitle(
            "Brain — Live State Dashboard",
            color=_TEXT, fontsize=14, fontweight='bold', y=0.97,
        )

    def _axes(self):
        return (self.ax_neurons, self.ax_traj, self.ax_scores, self.ax_fires)

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, brain, results: list[dict], fire_counts: list[int]) -> None:
        """Redraw all four panels with current brain state + results."""
        for ax in self._axes():
            ax.cla()
            ax.set_facecolor(_PANEL)
        self._draw_neurons(brain)
        self._draw_trajectories(brain)
        self._draw_scores(results)
        self._draw_fires(fire_counts, len(results))

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=_DARK)

    # ── Panel 1: Neuron graph ─────────────────────────────────────────────────

    def _draw_neurons(self, brain) -> None:
        ax = self.ax_neurons
        ax.set_title("Neuron Graph  (Markov thought states)", color=_TEXT,
                     fontsize=9, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(_GRID)

        store = brain._traj_store
        with store._lock:
            km        = store._km
            fail_rate = store._fail_rate
            all_vecs  = list(store._all_vecs)
            run_ids   = list(store._all_run_ids)
            runs      = list(store._runs)

        if not all_vecs:
            _no_data(ax, "No trajectories stored yet")
            return

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            _no_data(ax, "scikit-learn required for visualization")
            return

        all_mat = np.stack(all_vecs)

        if km is None or fail_rate is None:
            # Markov not yet fitted — show raw scatter
            _raw_scatter(ax, all_mat, runs, run_ids)
            n = len(all_vecs)
            ax.text(0.02, 0.02, f"Markov: {n} chunks  (need ≥30 to activate)",
                    color=_DIM, fontsize=7, transform=ax.transAxes)
            return

        # Project all chunks + cluster centers to 2D
        pca = PCA(n_components=2, random_state=42)
        pca.fit(all_mat)
        centers_2d = pca.transform(km.cluster_centers_)

        # Visit counts per cluster
        states      = km.predict(all_mat)
        k           = len(km.cluster_centers_)
        visit_count = np.bincount(states, minlength=k).astype(float)
        max_visit   = max(visit_count.max(), 1.0)

        # Transition counts
        trans = np.zeros((k, k))
        for run in runs:
            if len(run.vecs) < 2:
                continue
            run_states = km.predict(run.vecs)
            for s_i, s_j in zip(run_states[:-1], run_states[1:]):
                trans[s_i, s_j] += 1
        max_trans = max(trans.max(), 1.0)

        # Draw transition edges (threshold to avoid clutter)
        for i in range(k):
            for j in range(k):
                if i == j or trans[i, j] == 0:
                    continue
                w = float(trans[i, j]) / max_trans
                if w < 0.12:
                    continue
                x0, y0 = centers_2d[i]
                x1, y1 = centers_2d[j]
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=_DIM,
                        alpha=min(0.85, w * 0.7),
                        lw=1.2 + w * 1.5,
                        mutation_scale=10,
                        connectionstyle="arc3,rad=0.08",
                    ),
                    zorder=1,
                )

        # Draw nodes
        cmap = plt.get_cmap("RdYlGn_r")
        for idx, (x, y) in enumerate(centers_2d):
            fr    = float(fail_rate[idx])
            size  = 90 + 380 * (visit_count[idx] / max_visit)
            color = cmap(fr)
            ax.scatter(x, y, s=size, color=color, zorder=4,
                       edgecolors='white', linewidths=0.6, alpha=0.92)
            ax.text(x, y, str(idx), ha='center', va='center',
                    fontsize=6, color='black', zorder=5, fontweight='bold')

        ax.text(0.02, 0.97,
                f"{k} states · {len(runs)} runs · {len(all_vecs)} chunks",
                color=_DIM, fontsize=7, transform=ax.transAxes, va='top')

        handles = [
            mpatches.Patch(color=cmap(0.05), label='Low fail'),
            mpatches.Patch(color=cmap(0.95), label='High fail'),
        ]
        ax.legend(handles=handles, loc='lower right', fontsize=7,
                  facecolor=_PANEL, labelcolor=_TEXT, framealpha=0.6)

    # ── Panel 2: Trajectory map ───────────────────────────────────────────────

    def _draw_trajectories(self, brain) -> None:
        ax = self.ax_traj
        ax.set_title("Trajectory Map  (PCA — each run = one polyline)",
                     color=_TEXT, fontsize=9, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(_GRID)

        store = brain._traj_store
        with store._lock:
            runs = list(store._runs)

        if not runs:
            _no_data(ax, "No runs stored yet")
            return

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            _no_data(ax, "scikit-learn required")
            return

        # Stack all vecs; track boundaries
        all_vecs   = np.vstack([r.vecs for r in runs])
        boundaries = []
        offset = 0
        for r in runs:
            n = len(r.vecs)
            boundaries.append((offset, offset + n, r.label))
            offset += n

        n_comp = min(2, all_vecs.shape[0], all_vecs.shape[1])
        pca    = PCA(n_components=n_comp, random_state=42)
        pts    = pca.fit_transform(all_vecs)
        if pts.ndim == 1 or pts.shape[1] < 2:
            pts = np.column_stack([pts.ravel(), np.zeros(len(pts))])

        for start, end, label in boundaries:
            color = _PASS if label == 1 else _FAIL
            xs = pts[start:end, 0]
            ys = pts[start:end, 1]
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9, zorder=1)
            ax.scatter(xs[:1], ys[:1], color=color, s=22, zorder=3, alpha=0.8)
            ax.scatter(xs[-1:], ys[-1:], marker='x', color=color, s=28,
                       zorder=3, linewidths=1.5, alpha=0.85)

        n_pass = sum(1 for r in runs if r.label == 1)
        n_fail = len(runs) - n_pass
        ax.legend(
            handles=[
                mpatches.Patch(color=_PASS, label=f'Pass ({n_pass})'),
                mpatches.Patch(color=_FAIL, label=f'Fail ({n_fail})'),
            ],
            loc='lower right', fontsize=7,
            facecolor=_PANEL, labelcolor=_TEXT, framealpha=0.6,
        )

    # ── Panel 3: Score timeline ───────────────────────────────────────────────

    def _draw_scores(self, results: list[dict]) -> None:
        ax = self.ax_scores
        ax.set_title("Score Timeline", color=_TEXT, fontsize=9, pad=6)
        _style_ax(ax)

        if not results:
            _no_data(ax, "No results yet")
            return

        n      = len(results)
        xs     = list(range(1, n + 1))
        passed = [int(r["passed"]) for r in results]
        colors = [_PASS if p else _FAIL for p in passed]

        ax.bar(xs, [1] * n, color=colors, alpha=0.72, width=0.8, zorder=2)

        # Cumulative accuracy overlay
        ax2 = ax.twinx()
        cum = [sum(passed[:i + 1]) / (i + 1) for i in range(n)]
        ax2.plot(xs, cum, color='white', linewidth=1.8, alpha=0.9, zorder=4)
        ax2.axhline(1.0, color=_DIM, linewidth=0.6, linestyle='--')
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_ylabel("Accuracy", color=_TEXT, fontsize=7)
        ax2.tick_params(colors=_TEXT, labelsize=7)
        for sp in ax2.spines.values():
            sp.set_color(_GRID)

        ax.set_xlim(0.3, n + 0.7)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("Task #", color=_TEXT, fontsize=7)
        ax.set_yticks([])
        ax.text(0.02, 0.95,
                f"{sum(passed)}/{n}  ({sum(passed)/n:.0%})",
                color=_TEXT, fontsize=10, fontweight='bold',
                transform=ax.transAxes, va='top')

    # ── Panel 4: Fire histogram ───────────────────────────────────────────────

    def _draw_fires(self, fire_counts: list[int], n_tasks: int) -> None:
        ax = self.ax_fires
        ax.set_title("Brain Fires per Task  (target ≈ 30%)", color=_TEXT,
                     fontsize=9, pad=6)
        _style_ax(ax)

        if not fire_counts:
            _no_data(ax, "No fire data yet")
            return

        n   = len(fire_counts)
        xs  = list(range(1, n + 1))
        colors = [_FIRE if c > 0 else _DIM for c in fire_counts]
        ax.bar(xs, [min(c, 3) for c in fire_counts], color=colors,
               alpha=0.85, width=0.8, zorder=2)

        # Cumulative fire rate
        n_fired = sum(1 for c in fire_counts if c > 0)
        cum_fr  = [
            sum(1 for c in fire_counts[:i + 1] if c > 0) / (i + 1)
            for i in range(n)
        ]
        ax2 = ax.twinx()
        ax2.plot(xs, cum_fr, color=_FIRE, linewidth=1.8, alpha=0.9, zorder=4)
        ax2.axhline(0.30, color='white', linewidth=0.9, linestyle='--', alpha=0.55)
        ax2.text(n + 0.25, 0.30, "30%", color=_TEXT, fontsize=7, va='center')
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_ylabel("Fire rate", color=_TEXT, fontsize=7)
        ax2.tick_params(colors=_TEXT, labelsize=7)
        for sp in ax2.spines.values():
            sp.set_color(_GRID)

        ax.set_xlim(0.3, n + 0.7)
        ax.set_xlabel("Task #", color=_TEXT, fontsize=7)
        fire_rate = n_fired / n
        ok = abs(fire_rate - 0.30) < 0.12
        ax.text(
            0.02, 0.95,
            f"Fire rate: {fire_rate:.0%}  ({n_fired}/{n} tasks)",
            color=_FIRE if ok else _FAIL,
            fontsize=9, fontweight='bold',
            transform=ax.transAxes, va='top',
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_data(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            color=_DIM, fontsize=9, transform=ax.transAxes)


def _style_ax(ax) -> None:
    ax.tick_params(colors=_TEXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(_GRID)


def _raw_scatter(ax, all_mat, runs, run_ids) -> None:
    """Scatter raw embeddings colored by run label (pre-Markov)."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    if len(all_mat) < 2:
        return
    n_comp = min(2, len(all_mat), all_mat.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    pts    = pca.fit_transform(all_mat)
    if pts.ndim == 1 or pts.shape[1] < 2:
        pts = np.column_stack([pts.ravel(), np.zeros(len(pts))])
    labels = [r.label for r in runs]
    colors = [_PASS if labels[run_ids[i]] == 1 else _FAIL
              for i in range(len(all_mat))]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=14, alpha=0.55, zorder=2)
