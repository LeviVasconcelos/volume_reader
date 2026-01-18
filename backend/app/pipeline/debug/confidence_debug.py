"""Debug visualizations for the confidence scoring stage."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.pipeline.debug import DebugContext


def create_score_breakdown_plot(
    component_scores: dict[str, float],
    weights: dict[str, float],
    final_score: float,
) -> np.ndarray:
    """
    Create bar chart showing component score breakdown.

    Args:
        component_scores: Individual component scores
        weights: Weight for each component
        final_score: Final weighted score

    Returns:
        Plot image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Raw scores
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        component_weights = [weights.get(c, 0) for c in components]

        # Color by score (green=good, red=bad)
        colors = [(1 - s, s, 0.2) for s in scores]

        bars = ax1.barh(components, scores, color=colors, alpha=0.7)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Score', fontsize=12)
        ax1.set_title('Component Scores', fontsize=14)
        ax1.axvline(0.6, color='orange', linestyle='--', linewidth=2, label='Threshold (0.6)')
        ax1.legend()

        # Add value labels
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{score:.2f}', va='center', fontsize=10)

        # Weighted contribution
        weighted_scores = [s * weights.get(c, 0) for c, s in zip(components, scores)]
        bars2 = ax2.barh(components, weighted_scores, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Weighted Contribution', fontsize=12)
        ax2.set_title(f'Weighted Contributions (Final: {final_score:.2f})', fontsize=14)

        # Add weight labels
        for bar, w in zip(bars2, component_weights):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'(w={w:.2f})', va='center', fontsize=9, color='gray')

        fig.tight_layout()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((400, 800, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def create_radar_chart(
    component_scores: dict[str, float],
    final_score: float,
) -> np.ndarray:
    """
    Create radar/spider chart of component scores.

    Args:
        component_scores: Individual component scores
        final_score: Final score (shown in center)

    Returns:
        Radar chart image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        # Prepare data
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        n = len(components)

        if n < 3:
            # Need at least 3 components for radar chart
            return create_score_breakdown_plot(component_scores, {}, final_score)

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        scores_plot = scores + scores[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, scores_plot, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, scores_plot, alpha=0.25, color='steelblue')

        # Add threshold circle
        threshold = [0.6] * (n + 1)
        ax.plot(angles, threshold, '--', linewidth=1, color='orange', alpha=0.7)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(components, fontsize=10)
        ax.set_ylim(0, 1)

        # Add center text
        ax.text(0, 0, f'{final_score:.2f}', ha='center', va='center',
               fontsize=24, fontweight='bold',
               color='green' if final_score >= 0.6 else 'red')

        ax.set_title('Confidence Score Components', fontsize=14, y=1.08)

        fig.tight_layout()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((500, 500, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img


def create_warnings_impact_plot(
    warnings: list[str],
    base_score: float,
    final_score: float,
) -> np.ndarray:
    """
    Create visualization showing warning impact on score.

    Args:
        warnings: List of warning messages
        base_score: Score before warning penalties
        final_score: Score after penalties

    Returns:
        Impact plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        if not warnings:
            ax.text(0.5, 0.5, 'No warnings generated',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
            ax.set_title('Warning Impact on Score', fontsize=14)
        else:
            # Calculate penalty
            penalty = base_score - final_score

            # Show waterfall
            labels = ['Base Score'] + [f'W{i+1}' for i in range(len(warnings))] + ['Final']
            values = [base_score]

            # Distribute penalty across warnings
            penalty_per = penalty / len(warnings) if warnings else 0
            for _ in warnings:
                values.append(-penalty_per)
            values.append(final_score)

            # Waterfall chart
            running = 0
            for i, (label, val) in enumerate(zip(labels, values)):
                if i == 0:
                    ax.bar(label, val, color='steelblue', alpha=0.7)
                    running = val
                elif i == len(labels) - 1:
                    ax.bar(label, val, color='green' if val >= 0.6 else 'red', alpha=0.7)
                else:
                    ax.bar(label, val, bottom=running + val, color='coral', alpha=0.7)
                    running += val

            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Warning Impact on Confidence Score', fontsize=14)
            ax.axhline(0.6, color='orange', linestyle='--', linewidth=2, label='Threshold')
            ax.legend()

            # Add warning text
            for i, warning in enumerate(warnings[:5]):  # Limit to 5
                ax.text(0.02, 0.95 - i * 0.05, f'W{i+1}: {warning[:60]}...' if len(warning) > 60 else f'W{i+1}: {warning}',
                       transform=ax.transAxes, fontsize=8, color='gray')

        fig.tight_layout()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    except ImportError:
        img = np.full((400, 800, 3), 255, dtype=np.uint8)
        cv2.putText(img, "matplotlib required for plots", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img


def debug_confidence(
    ctx: "DebugContext",
    component_scores: dict[str, float],
    weights: dict[str, float],
    final_score: float,
    warnings: list[str],
) -> None:
    """
    Generate debug visualizations for the confidence scoring stage.

    Outputs:
        - score_breakdown.png: Bar chart of component scores
        - radar_chart.png: Spider plot of all factors
        - warnings_impact.png: Warning effect on score
        - confidence_summary.json: Full breakdown

    Args:
        ctx: Debug context
        component_scores: Individual component scores
        weights: Weights for each component
        final_score: Final confidence score
        warnings: Warning messages generated
    """
    if not ctx.enabled:
        return

    # Score breakdown
    breakdown_plot = create_score_breakdown_plot(component_scores, weights, final_score)
    ctx.save_image("confidence", "score_breakdown.png", breakdown_plot)

    # Radar chart
    radar = create_radar_chart(component_scores, final_score)
    ctx.save_image("confidence", "radar_chart.png", radar)

    # Warnings impact
    # Calculate base score (before warning penalty)
    warning_weight = weights.get("warnings", 0.1)
    if warning_weight > 0:
        warning_score = component_scores.get("warnings", 1.0)
        # Estimate base score
        other_weighted = sum(
            s * weights.get(c, 0)
            for c, s in component_scores.items()
            if c != "warnings"
        )
        total_other_weight = sum(w for c, w in weights.items() if c != "warnings")
        base_score = other_weighted / total_other_weight if total_other_weight > 0 else final_score
    else:
        base_score = final_score

    warnings_plot = create_warnings_impact_plot(warnings, base_score, final_score)
    ctx.save_image("confidence", "warnings_impact.png", warnings_plot)

    # Summary JSON
    summary = {
        "final_score": final_score,
        "threshold": 0.6,
        "status": "good" if final_score >= 0.6 else "low",
        "component_scores": component_scores,
        "weights": weights,
        "weighted_contributions": {
            c: s * weights.get(c, 0)
            for c, s in component_scores.items()
        },
        "num_warnings": len(warnings),
        "warnings": warnings,
    }
    ctx.save_json("confidence", "confidence_summary", summary)
