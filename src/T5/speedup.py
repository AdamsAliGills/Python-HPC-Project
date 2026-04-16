import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import os


def parse_time(time_str):
    """Converts h:mm:ss or m:ss to total seconds."""
    parts = [float(p) for p in time_str.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0] * 60 + parts[1]


def extract_scaling_data(filename):
    """Parses core count and wall clock time from log."""
    with open(filename, "r") as f:
        content = f.read()
    cores = re.findall(r'Command being timed: ".*? (\d+)"', content)
    times = re.findall(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", content
    )
    if not cores or not times:
        return pd.DataFrame()
    return pd.DataFrame(
        {"Cores": [int(c) for c in cores], "Time_Sec": [parse_time(t) for t in times]}
    )


def plot_parallel_speedup(pattern):
    """Generates scaling metrics, plots results, and provides final Amdahl analysis."""
    files = glob.glob(pattern)
    if not files:
        return

    df_list = [extract_scaling_data(f) for f in files]
    all_data = pd.concat([df for df in df_list if not df.empty], ignore_index=True)
    all_data = all_data.sort_values("Cores").drop_duplicates("Cores")

    t1_val = all_data[all_data["Cores"] == 1]["Time_Sec"].iloc[0]
    all_data["Speedup"] = t1_val / all_data["Time_Sec"]
    all_data["Efficiency"] = all_data["Speedup"] / all_data["Cores"]

    all_data["Parallel_Fraction"] = all_data.apply(
        lambda r: (
            ((1 / r["Speedup"]) - 1) / ((1 / r["Cores"]) - 1)
            if r["Cores"] > 1
            else None
        ),
        axis=1,
    )

    all_data["Theoretical_Max_Speedup"] = 1 / (1 - all_data["Parallel_Fraction"])

    all_data.to_csv("data/scaling_results.csv", index=False)

    best_run = all_data.loc[all_data["Speedup"].idxmax()]
    p_est = all_data["Parallel_Fraction"].iloc[-3:].mean()
    s_max = 1 / (1 - p_est)
    pct_achieved = (best_run["Speedup"] / s_max) * 100

    print(f"Parallelized Fraction: {p_est * 100:.2f}%")
    print(f"Theoretical Max Speedup: {s_max:.2f}x")
    print(
        f"Max Speedup Achieved: {best_run['Speedup']:.2f}x ({pct_achieved:.1f}% of theoretical limit)"
    )
    print(f"Cores used for max: {int(best_run['Cores'])}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        all_data["Cores"].values, all_data["Speedup"].values, "o-", label="Observed"
    )
    plt.plot(
        all_data["Cores"].values,
        all_data["Cores"].values,
        "--",
        color="gray",
        label="Ideal",
    )
    plt.legend()
    plt.savefig("plots/parallel_speedup_plot.png")


plot_parallel_speedup("required_logs/T5/plans_10/*.log")

