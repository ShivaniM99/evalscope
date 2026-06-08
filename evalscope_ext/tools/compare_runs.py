"""
compare_runs.py — compare full vs pruned evalscope output directories.

Usage:
    python -m evalscope.ext.tools.compare_runs \
        --full  ./results_full \
        --pruned ./results_pruned
"""
import argparse, json, os
from pathlib import Path
from scipy.stats import spearmanr


def load_results(results_dir: str) -> dict:
    """Load per-model mean acc from an evalscope output directory."""
    results = {}
    for f in Path(results_dir).rglob("*.json"):
        try:
            data = json.loads(f.read_text())
            model = data.get("model", f.stem)
            acc   = data.get("acc") or data.get("weighted_avg_acc")
            if acc is not None:
                results[model] = float(acc)
        except Exception:
            continue
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   required=True, help="Full run results dir")
    parser.add_argument("--pruned", required=True, help="Pruned run results dir")
    args = parser.parse_args()

    full   = load_results(args.full)
    pruned = load_results(args.pruned)

    models = sorted(set(full) & set(pruned))
    if not models:
        print("No overlapping models found between full and pruned results.")
        return

    print(f"\n{'Model':<20} {'Full acc':>10} {'Pruned acc':>12} "
          f"{'Rank full':>10} {'Rank pruned':>12} {'Match':>7}")
    print("-" * 72)

    full_accs   = [full[m]   for m in models]
    pruned_accs = [pruned[m] for m in models]
    full_ranks  = sorted(range(len(models)), key=lambda i: -full_accs[i])
    pruned_ranks= sorted(range(len(models)), key=lambda i: -pruned_accs[i])

    for i, m in enumerate(models):
        rf = full_ranks.index(i) + 1
        rp = pruned_ranks.index(i) + 1
        match = "✅" if rf == rp else "❌"
        print(f"{m:<20} {full_accs[i]:>10.3f} {pruned_accs[i]:>12.3f} "
              f"{rf:>10} {rp:>12} {match:>7}")

    rho, pval = spearmanr(full_accs, pruned_accs)
    print(f"\nSpearman ρ = {rho:.4f}  (p = {pval:.4f})")
    print(f"Compression: {len(models)} models compared")


if __name__ == "__main__":
    main()