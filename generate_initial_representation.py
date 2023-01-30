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
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-f", "--file", help="File containing initial scene", required=True)
args = parser.parse_args()

def main():
    with open(args.config, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)
    with open(args.file, 'rb') as handle:
        save_dict = pickle.load(handle)

    x_ego = save_dict["ego"]["x"][0]
    y_ego = save_dict["ego"]["y"][0]
    yaw_ego = save_dict["ego"]["yaw"][0]
    x_other = save_dict["other"]["x"][0]
    y_other = save_dict["other"]["y"][0]
    yaw_other = save_dict["other"]["yaw"][0]

    L_ego = hyperparams["ego"]["params"]["L"]
    L_other = hyperparams["other"]["params"]["L"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    plt.cla()
    road = Rectangle((-12.0, -1.25), 1000.0, 5.0, edgecolor='k', facecolor='grey', fill=True, zorder=0)
    ax.add_patch(road)
    plt.plot([-12.0, 1000.0], [-1.25, -1.25], 'w-', linewidth=2, zorder=5)
    plt.plot([-12.0, 1000.0], [1.25, 1.25], 'w--', linewidth=2, zorder=5)
    plt.plot([-12.0, 1000.0], [3.75, 3.75], 'w-', linewidth=2, zorder=5)
    ego = Rectangle((x_ego - L_ego / 2, y_ego - 1.5 / 2), L_ego, 1.5, edgecolor='k', facecolor='blue', fill=True, zorder=10)
    ax.add_patch(ego)
    other = Rectangle((x_other - L_other / 2, y_other - 1.5 / 2), L_other, 1.5, angle=np.rad2deg(yaw_other), edgecolor='k',
                      facecolor='red', fill=True, zorder=10)
    ax.add_patch(other)

    plt.xlim(x_ego - 12.0, x_ego + 10.0)
    plt.ylim(-1.75, 4.25)
    ax.set_xlabel(r"$x$ [m]", fontsize=10)
    ax.set_ylabel(r"$y$ [m]", fontsize=10)
    plt.show()
    fig.savefig("figs/initial_representation.svg", bbox_inches='tight',
               transparent=True, dpi=200)


if __name__ == '__main__':
    main()
