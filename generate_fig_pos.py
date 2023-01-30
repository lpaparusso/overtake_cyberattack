import numpy as np
import matplotlib.pyplot as plt
import csv, os
from models.ego_vehicle import EgoVehicle
from models.other_vehicle import OtherVehicle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import json
import argparse
import pickle
import matplotlib as mpl

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c1", "--config1", help="Config file with dataset parameters", required=True)
parser.add_argument("-c2", "--config2", help="Config file with dataset parameters", required=True)
parser.add_argument("-f1", "--file1", help="File containing scene", required=True)
parser.add_argument("-f2", "--file2", help="File containing scene", required=True)
args = parser.parse_args()

def main():
    with open(args.config1, 'r', encoding='utf-8') as conf_json:
        hyperparams1 = json.load(conf_json)
    with open(args.config2, 'r', encoding='utf-8') as conf_json:
        hyperparams2 = json.load(conf_json)
    with open(args.file1, 'rb') as handle:
        save_dict1 = pickle.load(handle)
    with open(args.file2, 'rb') as handle:
        save_dict2 = pickle.load(handle)

    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    fig.tight_layout(pad=3.0)
    plt.cla()
    time1 = np.arange(save_dict1["ego"]["x"].shape[0]) * 0.01
    time2 = np.arange(save_dict2["ego"]["x"].shape[0]) * 0.01
    ax[0].plot(time1, save_dict1["other"]["x"] - save_dict1["ego"]["x"], color="k")
    ax[1].plot(time2, save_dict2["other"]["x"] - save_dict2["ego"]["x"], color="k")

    ax[0].plot(time1, -6 + time1 * 0.0, color='orange')
    ax[0].plot(time1, 6 + time1 * 0.0, color='orange')
    ax[1].plot(time2, -6 + time2 * 0.0, color='orange')
    ax[1].plot(time2, 6 + time2 * 0.0, color='orange')

    ax[0].set_title(r"Interaction-unaware cyberattack", fontdict={'fontsize': 10, 'fontweight': 'medium'})
    ax[1].set_title(r"Interaction-aware cyberattack", fontdict={'fontsize': 10, 'fontweight': 'medium'})

    ax[0].set_xlabel(r"time [s]", fontsize=8)
    ax[0].set_ylabel(r"$x_{\mathrm{B}} - x_{\mathrm{A}}$ [m]", fontsize=8)
    ax[1].set_xlabel(r"time [s]", fontsize=8)
    ax[1].set_ylabel(r"$x_{\mathrm{B}} - x_{\mathrm{A}}$ [m]", fontsize=8)
    ax[0].set_xlim([0.0, time1[-1]])
    ax[1].set_xlim([0.0, time2[-1]])
    ax[0].set_ylim([-10.0, 10.0])
    ax[1].set_ylim([-10.0, 10.0])
    plt.show()
    fig.savefig("figs/pos_comparison.svg", bbox_inches='tight',
               transparent=True, dpi=200)


if __name__ == '__main__':
    main()
