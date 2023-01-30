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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
args = parser.parse_args()

def main():

    # Load configs
    with open(args.config, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Initialize vehicles
    params_ego = hyperparams["ego"]["params"]
    x_ego = hyperparams["ego"]["initial_conditions"]["x"]
    y_ego = hyperparams["ego"]["initial_conditions"]["y"]
    v_ego = hyperparams["ego"]["initial_conditions"]["v"]
    yaw_ego = hyperparams["ego"]["initial_conditions"]["yaw"]
    params_other = hyperparams["other"]["params"]
    x_other = hyperparams["other"]["initial_conditions"]["x"]
    y_other = hyperparams["other"]["initial_conditions"]["y"]
    v_other = hyperparams["other"]["initial_conditions"]["v"]
    yaw_other = hyperparams["other"]["initial_conditions"]["yaw"]

    ego_vehicle = EgoVehicle(params_ego, x_ego, y_ego, yaw_ego, v_ego)
    other_vehicle = OtherVehicle(params_other, x_other, y_other, yaw_other, v_other)
    other_vehicle_vel = v_other
    other_vehicle_knowledge = hyperparams["other_vehicle_knowledge"]

    # Initialize dictionary to save vehicles quantities
    save_dict = initialize_save_dict()

    # Start simulation
    iteration = 0

    if hyperparams["show_animation"]:
        fig, ax = plt.subplots()
        plt.cla()
        plot_vehicles(ego_vehicle, other_vehicle, iteration, ax)

    store_vehicles_quantities(ego_vehicle, other_vehicle, iteration, save_dict, hyperparams["save_dir"])

    while True:

        dt = hyperparams["ego"]["params"]["dt"]
        if iteration > int(20.0/dt):
            print("crashed")
            break

        if other_vehicle_knowledge is False:
            other_vehicle_acc = (other_vehicle.v - other_vehicle_vel) / dt
            other_vehicle_vel = other_vehicle.v
        escaped = other_vehicle.next(ego_vehicle.x, ego_vehicle.y, ego_vehicle.yaw, ego_vehicle.v)
        if other_vehicle_knowledge is True:
            if other_vehicle.cumulative_overtake_time + 100 * dt >= 4.0:
                other_vehicle_acc = 0.0
            else:
                other_vehicle_acc = other_vehicle.throttle
        ego_vehicle.next(other_vehicle.x, other_vehicle.y, other_vehicle.yaw, other_vehicle.v, other_vehicle_acc)
        if escaped is not None:
            print("escaped")
            break

        iteration += 1

        if (iteration % int(1/(4*dt)) == 0) and hyperparams["show_animation"]:
            plot_vehicles(ego_vehicle, other_vehicle, iteration, ax)

        store_vehicles_quantities(ego_vehicle, other_vehicle, iteration, save_dict, hyperparams["save_dir"])
    save_vehicles_quantities(save_dict, directory=hyperparams["save_dir"])

def initialize_save_dict():
    save_dict = dict()
    save_dict["ego"] = dict()
    save_dict["other"] = dict()

    save_dict["ego"]["x"] = list()
    save_dict["ego"]["y"] = list()
    save_dict["ego"]["yaw"] = list()
    save_dict["ego"]["v"] = list()
    save_dict["ego"]["throttle"] = list()
    save_dict["ego"]["delta"] = list()
    save_dict["ego"]["mode"] = list()

    save_dict["other"]["x"] = list()
    save_dict["other"]["y"] = list()
    save_dict["other"]["yaw"] = list()
    save_dict["other"]["v"] = list()
    save_dict["other"]["throttle"] = list()
    save_dict["other"]["delta"] = list()
    save_dict["other"]["mode"] = list()
    save_dict["other"]["change_lane_complete"] = list()
    save_dict["other"]["cumulative_overtake_time"] = list()
    save_dict["other"]["case_overtake"] = list()
    save_dict["other"]["overtake_tried"] = list()

    return save_dict

def store_vehicles_quantities(ego_vehicle, other_vehicle, iteration, save_dict, directory=None):

    if directory is not None:
        save_dict["ego"]["x"].append(ego_vehicle.x)
        save_dict["ego"]["y"].append(ego_vehicle.y)
        save_dict["ego"]["yaw"].append(ego_vehicle.yaw)
        save_dict["ego"]["v"].append(ego_vehicle.v)
        save_dict["ego"]["throttle"].append(ego_vehicle.throttle)
        save_dict["ego"]["delta"].append(ego_vehicle.delta)
        save_dict["ego"]["mode"].append(ego_vehicle.mode)

        save_dict["other"]["x"].append(other_vehicle.x)
        save_dict["other"]["y"].append(other_vehicle.y)
        save_dict["other"]["yaw"].append(other_vehicle.yaw)
        save_dict["other"]["v"].append(other_vehicle.v)
        save_dict["other"]["throttle"].append(other_vehicle.throttle)
        save_dict["other"]["delta"].append(other_vehicle.delta)
        save_dict["other"]["mode"].append(other_vehicle.mode)
        save_dict["other"]["change_lane_complete"].append(other_vehicle.change_lane_complete)
        save_dict["other"]["cumulative_overtake_time"].append(other_vehicle.cumulative_overtake_time)
        save_dict["other"]["case_overtake"].append(other_vehicle.case_overtake)
        save_dict["other"]["overtake_tried"].append(other_vehicle.overtake_tried)

def save_vehicles_quantities(save_dict, directory=None):

    if directory is not None:
        for agent_name, agent in save_dict.items():
            for variable_name, variable in agent.items():
                save_dict[agent_name][variable_name] = np.array(variable)
        with open(directory, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_vehicles(ego_vehicle, other_vehicle, frame, ax):
    x_ego = ego_vehicle.x
    y_ego = ego_vehicle.y
    yaw_ego = ego_vehicle.yaw
    x_other = other_vehicle.x
    y_other = other_vehicle.y
    yaw_other = other_vehicle.yaw

    L_ego = ego_vehicle.params["L"]
    L_other = other_vehicle.params["L"]

    plt.cla()
    road = Rectangle((-10.0, -1.25), 1000.0, 5.0, edgecolor='k', facecolor='grey', fill=True, zorder=0)
    ax.add_patch(road)
    plt.plot([-10.0, 1000.0], [-1.25, -1.25], 'w-', linewidth=2, zorder=5)
    plt.plot([-10.0, 1000.0], [1.25, 1.25], 'w--', linewidth=2, zorder=5)
    plt.plot([-10.0, 1000.0], [3.75, 3.75], 'w-', linewidth=2, zorder=5)
    ego = Rectangle((x_ego - L_ego / 2, y_ego - 1.5 / 2), L_ego, 1.5, edgecolor='k', facecolor='blue', fill=True, zorder=10)
    ax.add_patch(ego)
    other = Rectangle((x_other - L_other / 2, y_other - 1.5 / 2), L_other, 1.5, angle=np.rad2deg(yaw_other), edgecolor='k',
                      facecolor='red', fill=True, zorder=10)
    ax.add_patch(other)

    plt.xlim(x_ego - 10.0, x_ego + 10.0)
    plt.ylim(-8.75, 11.25)
    plt.title('frame={}'.format(frame))

    plt.pause(0.05)

if __name__ == '__main__':
    main()
