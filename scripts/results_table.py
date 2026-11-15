"""Print the experiment as a Markdown table:  python scripts/results_table.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "experiment.json")) as f:
    data = json.load(f)

NAMES = {"reflex": "Reflex (11 inputs)", "board": "Full board", "search-only": "Search only"}
rank = lambda r: 0 if r["agent"] == "reflex" else 3 if r["agent"] == "search-only" else 2 if r["budget"] else 1

print(f"Mean score over {data['games']} held-out games on a {data['size']}x{data['size']} board "
      f"(score = food eaten, {data['size'] ** 2 - 3} fills the board):\n")
print("| Agent | Budget | Mean score | 95% CI | Median | Best |")
print("|---|---:|---:|---:|---:|---:|")
for r in sorted(data["results"], key=lambda r: (rank(r), r["budget"])):
    name = "Full board + search" if r["agent"] == "board" and r["budget"] else NAMES[r["agent"]]
    print(f"| {name} | {r['budget'] or '-'} | **{r['mean']:.1f}** | {r['ci95'][0]:.1f} to {r['ci95'][1]:.1f} "
          f"| {r['median']:.0f} | {r['max']} |")
