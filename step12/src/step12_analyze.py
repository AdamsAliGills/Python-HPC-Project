import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Analyze step 12 CSV results.")
    parser.add_argument("--input", default="output/step12/all_buildings_results.csv")
    parser.add_argument("--plot", default="output/step12/mean_temperature_histogram.png")
    parser.add_argument("--summary", default="output/step12/step12_summary.txt")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.plot), exist_ok=True)

    df = pd.read_csv(args.input)

    avg_mean_temp = df["mean_temp"].mean()
    avg_std_temp = df["std_temp"].mean()
    num_above_18 = int((df["pct_above_18"] >= 50.0).sum())
    num_below_15 = int((df["pct_below_15"] >= 50.0).sum())

    plt.figure(figsize=(8, 5))
    plt.hist(df["mean_temp"], bins=args.bins)
    plt.xlabel("Mean temperature (°C)")
    plt.ylabel("Number of buildings")
    plt.title("Distribution of mean temperatures across buildings")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)

    summary_lines = [
        f"Number of buildings: {len(df)}",
        f"Average mean temperature: {avg_mean_temp}",
        f"Average temperature standard deviation: {avg_std_temp}",
        f"Buildings with at least 50% area above 18C: {num_above_18}",
        f"Buildings with at least 50% area below 15C: {num_below_15}",
        f"Histogram saved to: {args.plot}",
    ]

    with open(args.summary, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    for line in summary_lines:
        print(line)
    print(f"Summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
