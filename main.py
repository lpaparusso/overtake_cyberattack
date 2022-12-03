import numpy as np
import matplotlib.pyplot as plt
import csv, os
from models.ego_vehicle import EgoVehicle
from models.other_vehicle import OtherVehicle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# show the animation plot during the simulation, default is False.
show_animation = True

def main():

    L = 2.9 # m
    dt = 0.01

    iteration = 0

    x_ego = 0.0
    y_ego = 0.0
    yaw_ego = 0.0
    v_ego = 20.0
    x_other = -10.0
    y_other = 0.0
    yaw_other = 0.0
    v_other = 22.0

    ego_vehicle = EgoVehicle(x_ego, y_ego, yaw_ego, v_ego)
    other_vehicle = OtherVehicle(x_other, y_other, yaw_other, v_other)
    other_vehicle_knowledge = False
    other_vehicle_vel = v_other

    if show_animation:
        fig, ax = plt.subplots()
        plt.cla()
        plot_vehicles(ego_vehicle, other_vehicle, L, iteration, ax)

    while True:
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

        if (iteration % int(1/(4*dt)) == 0) and show_animation:
            plot_vehicles(ego_vehicle, other_vehicle, L, iteration, ax)


def plot_vehicles(ego_vehicle, other_vehicle, L, frame, ax):
    x_ego = ego_vehicle.x
    y_ego = ego_vehicle.y
    yaw_ego = ego_vehicle.yaw
    x_other = other_vehicle.x
    y_other = other_vehicle.y
    yaw_other = other_vehicle.yaw

    plt.cla()
    road = Rectangle((-10.0, -1.25), 1000.0, 5.0, edgecolor='k', facecolor='grey', fill=True, zorder=0)
    ax.add_patch(road)
    plt.plot([-10.0, 1000.0], [-1.25, -1.25], 'w-', linewidth=2, zorder=5)
    plt.plot([-10.0, 1000.0], [1.25, 1.25], 'w--', linewidth=2, zorder=5)
    plt.plot([-10.0, 1000.0], [3.75, 3.75], 'w-', linewidth=2, zorder=5)
    ego = Rectangle((x_ego - L / 2, y_ego - 1.5 / 2), L, 1.5, edgecolor='k', facecolor='blue', fill=True, zorder=10)
    ax.add_patch(ego)
    other = Rectangle((x_other - L / 2, y_other - 1.5 / 2), L, 1.5, angle=np.rad2deg(yaw_other), edgecolor='k',
                      facecolor='red', fill=True, zorder=10)
    ax.add_patch(other)

    plt.xlim(x_ego - 10.0, x_ego + 10.0)
    plt.ylim(-8.75, 11.25)
    plt.title('frame={}'.format(frame))

    plt.pause(0.1)

if __name__ == '__main__':
    main()
