"""
    Data Integrity Analysis
    ================================
    Checks:
        - Missing values
        - Negative/zero case counts 
        - Region name mismatches between labels and graph node names
        - Missing graph date files (expected 61 per country)
        - Zero/negative edge weights in graph files

    Outputs:
        - figures/integrity/integrity_report.csv   — machine-readable summary
        - (stdout)                                 — human-readable table
"""

# === IMPORTS === 

import sys
import csv
import pandas as pd
from pathlib import Path

# Resolve project root regardless of where the script is called from 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "figures" / "integrity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Country registry
COUNTRIES = {
    "England": ("data/England [COVID-19]", "england_labels.csv", "EN"),
    "France":  ("data/France [COVID-19]",  "france_labels.csv",  "FR"),
    "Italy":   ("data/Italy [COVID-19]",   "italy_labels.csv",   "IT"),
    "Spain":   ("data/Spain [COVID-19]",   "spain_labels.csv",   "ES"),
}



# Helper Functions
def load_labels(folder, label_file):
    path = PROJECT_ROOT / folder / label_file
    df = pd.read_csv(path, index_col=0)
    if "name" in df.columns:          # Italy has extra 'name' and 'id' columns
        df = df.set_index("name")
    date_cols = [c for c in df.columns if str(c).startswith("20")]
    return df[date_cols]   # index = region, columns = string dates


def _graph_path(folder, prefix, date_str):
    return PROJECT_ROOT / folder / "graphs" / f"{prefix}_{date_str}.csv"


def graph_nodes_for_date(folder, prefix, date_str):
    """Return the set of node names that appear in one graph file."""
    p = _graph_path(folder, prefix, date_str)
    nodes = set()
    with open(p, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                nodes.add(row[0].strip())
                nodes.add(row[1].strip())
    return nodes


def all_graph_edges(folder, prefix, date_str):
    """Return list of (src, dst, weight) tuples for one graph file."""
    p = _graph_path(folder, prefix, date_str)
    edges = []
    with open(p, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                try:
                    edges.append((row[0].strip(), row[1].strip(), float(row[2])))
                except ValueError:
                    pass
    return edges


# Main analysis 
rows = []

for country, (folder, label_file, prefix) in COUNTRIES.items():
    print(f"\n{'='*60}")
    print(f"  {country.upper()}")
    print(f"{'='*60}")

    # 1. Label CSV: missing values
    df = load_labels(folder, label_file)
    n_missing = int(df.isna().sum().sum())
    print(f"  Missing values in label CSV       : {n_missing}")

    # 2. Negative case counts
    neg_mask = df < 0
    n_negative = int(neg_mask.sum().sum())
    neg_examples = []
    if n_negative:
        for reg in df.index:
            bad_dates = df.columns[df.loc[reg] < 0].tolist()
            for bd in bad_dates:
                neg_examples.append(f"{reg}@{bd}")
    print(f"  Negative case counts              : {n_negative}" + (f"  → {neg_examples[:5]}" if neg_examples else ""))

    # 3. Zero case counts 
    zero_mask = df == 0
    n_zero = int(zero_mask.sum().sum())
    print(f"  Zero case counts (days×regions)   : {n_zero}")

    # Derive expected dates from the label CSV (per-country)
    expected_dates = list(df.columns)

    # 4. Missing graph date files 
    graphs_dir = PROJECT_ROOT / folder / "graphs"
    present = {
        f.stem.replace(f"{prefix}_", "")
        for f in graphs_dir.glob(f"{prefix}_*.csv")
    }
    missing_dates = [d for d in expected_dates if d not in present]
    extra_dates   = [d for d in sorted(present) if d not in expected_dates]
    print(f"  Missing graph files               : {len(missing_dates)}" + (f"  → {missing_dates}" if missing_dates else ""))
    print(f"  Unexpected extra graph files      : {len(extra_dates)}" + (f"  → {extra_dates}" if extra_dates else ""))

    # 5. Region name mismatches 
    label_regions = set(df.index.str.strip())
    
    # Collect all node names across all available graph files
    all_graph_nodes: set = set()
    for d in expected_dates:
        if d in present:
            all_graph_nodes |= graph_nodes_for_date(folder, prefix, d)
            
    # Remove self-loop nodes (they equal region names) vs. label names
    in_graphs_not_labels = all_graph_nodes - label_regions
    in_labels_not_graphs = label_regions  - all_graph_nodes
    print(f"  Graph nodes not in label CSV      : {len(in_graphs_not_labels)}" + (f"  → {sorted(in_graphs_not_labels)[:5]}" if in_graphs_not_labels else ""))
    print(f"  Label regions not in any graph    : {len(in_labels_not_graphs)}" + (f"  → {sorted(in_labels_not_graphs)[:5]}" if in_labels_not_graphs else ""))

    # 6. Zero/negative edge weights 
    n_zero_w = 0
    n_neg_w  = 0
    for d in expected_dates:
        if d in present:
            for src, dst, w in all_graph_edges(folder, prefix, d):
                if w < 0:
                    n_neg_w += 1
                elif w == 0:
                    n_zero_w += 1
    print(f"  Zero edge weights (across all files): {n_zero_w}")
    print(f"  Negative edge weights               : {n_neg_w}")

    rows.append({
        "country":                country,
        "label_missing_values":   n_missing,
        "label_negative_counts":  n_negative,
        "label_zero_counts":      n_zero,
        "missing_graph_files":    len(missing_dates),
        "missing_graph_dates":    ";".join(missing_dates),
        "extra_graph_files":      len(extra_dates),
        "graph_nodes_not_in_label": len(in_graphs_not_labels),
        "label_regions_not_in_graph": len(in_labels_not_graphs),
        "zero_edge_weights":      n_zero_w,
        "negative_edge_weights":  n_neg_w,
    })

# Save report
report = pd.DataFrame(rows)
report_path = OUT_DIR / "integrity_report.csv"
report.to_csv(report_path, index=False)
print(f"\n\nIntegrity report saved → {report_path}")
print("\nSummary table:")
print(report.to_string(index=False))
