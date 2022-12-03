import numpy as np
import matplotlib.pyplot as plt
import csv, os
from models.ego_vehicle import EgoVehicle
from models.other_vehicle import OtherVehicle
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
    v_ego = 15.0
    x_other = -8.0
    y_other = 0.0
    yaw_other = 0.0
    v_other = 15.0

    ego_vehicle = EgoVehicle(x_ego, y_ego, yaw_ego, v_ego)
    other_vehicle = OtherVehicle(x_other, y_other, yaw_other, v_other)

    if show_animation:
        fig, ax = plt.subplots()
        plt.cla()
        plot_vehicles(ego_vehicle, other_vehicle, L, iteration, ax)

    while True:
        if iteration > int(20.0/dt):
            print("crashed")
            break

        ego_vehicle.next(other_vehicle.x, other_vehicle.y, other_vehicle.yaw, other_vehicle.v)
        escaped = other_vehicle.next(ego_vehicle.x, ego_vehicle.y, ego_vehicle.yaw, ego_vehicle.v)
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

    ax.add_patch(Rectangle((-10.0, -1.25), 1000.0, 5.0, edgecolor='k', facecolor='grey', fill=True))
    plt.plot([-10.0, 1000.0], [-1.25, -1.25], 'w-', linewidth=2)
    plt.plot([-10.0, 1000.0], [1.25, 1.25], 'w--', linewidth=2)
    plt.plot([-10.0, 1000.0], [3.75, 3.75], 'w-', linewidth=2)
    ax.add_patch(Rectangle((x_ego - L / 2, y_ego - 1.5 / 2), L, 1.5, edgecolor='k', facecolor='blue', fill=True))
    ax.add_patch(Rectangle((x_other - L / 2, y_other - 1.5 / 2), L, 1.5, angle=np.rad2deg(yaw_other), edgecolor='k', facecolor='red', fill=True))

    plt.xlim(x_ego - 10.0, x_ego + 10.0)
    plt.ylim(-8.75, 11.25)
    plt.title('frame={}'.format(frame))

    plt.pause(0.1)

if __name__ == '__main__':
    main()
