import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_run", type=str, help="name of run to load, ex: 2024-06-21_17-46-50_6_thrusters_indep", required=True)
    parser.add_argument("--checkpoint", type=str, help="name of checkpoint, ex: model_100.pt", required=True)
    parser.add_argument("--experiment_name", type=str, help="name of experiment, ex: warpauv_direct", required=True)
    parser.add_argument("--min", type=int, help="min step value to view data from, ex: 5", default=None)
    parser.add_argument("--max", type=int, help="max step value to view data from, ex: 10", default=None)
    args = parser.parse_args()

    dir_path = os.path.join("source", "results", "rsl_rl", args.experiment_name, args.load_run, args.checkpoint[:-3] + "_play")
    load_path = os.path.join(dir_path, "output.csv")
    save_path = os.path.join(dir_path, "output.png")

    df = pd.read_csv(load_path)

    if (((args.min != None) and (args.max != None)) and (args.min <= args.max)):
        df = df.iloc[args.min:args.max, :]

    ixs = list(range(df.shape[0]))

    fig, axes = plt.subplots(2)

    axes[0].plot(ixs, df.iloc[:, 0])
    axes[0].set_title("Reward")
    axes[1].plot(ixs, df.iloc[:, 1])
    axes[1].set_title("L2 Distance from Goal")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(save_path)

if __name__ == "__main__":
    main()