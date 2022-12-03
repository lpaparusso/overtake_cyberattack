import numpy as np
from models.bicyclemodel import ConstrainedLinearBicycleModel

class OtherVehicle(ConstrainedLinearBicycleModel):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super().__init__(x, y, yaw, v)
        self.x_other = None
        self.y_other = None
        self.yaw_other = None
        self.v_other = None
        self.mode = None
        self.start_overtake_complete = False
        self.cumulative_overtake_time = 0.0
        self.case_overtake = 1
        self.overtake_tried = False

        self.max_steer = np.radians(30.0)
        self.min_v = 10.0
        self.max_v = 20.0
        self.min_a = -7.0
        self.max_a = 4.0
        self.min_jerk = -100.0
        self.max_jerk = 20.0
        self.dt = 0.01

    def next(self, x_other, y_other, yaw_other, v_other):
        """
        Computes next pose using a state machine
        :param x_other: global x of the other agent
        :param y_other: global y of the other agent
        :param yaw_other: global yaw of the other agent
        :param v_other: global speed of the other agent
        """

        distance_other = np.sqrt((x_other - self.x) ** 2 + (y_other - self.y) ** 2)

        if self.mode is None:
            self.steady()
            return None
        if self.mode == 'steady':
            if distance_other > 8.0:
                self.steady()
                return None
            else:
                self.start_overtake()
                return None
        if self.mode == 'start_overtake' and (self.start_overtake_complete is False):
            self.start_overtake()
            return None
        elif self.mode == 'start_overtake' and (self.start_overtake_complete is True):
            self.perform_overtake(x_other, y_other, yaw_other, v_other)
            return None
        if self.mode == 'perform_overtake' and not self.overtake_tried:
            self.perform_overtake(x_other, y_other, yaw_other, v_other)
            return None
        if self.mode == 'perform_overtake' and self.overtake_tried:
            self.perform_overtake(x_other, y_other, yaw_other, v_other)
            if distance_other > 6.0:
                return 0

    def steady(self):
        """
        Behavioral mode: steady. Computes next pose
        """

        self.mode = 'steady'
        throttle = 0.0
        delta = 0.0
        self.update(throttle, delta)

    def start_overtake(self):
        """
        Behavioral mode: start_overtake. Computes next pose
        """

        self.mode = 'start_overtake'
        throttle = 0.0
        lateral_error = 2.5 - self.y
        orientation_error = - self.yaw
        delta = self.stanley_controller(lateral_error, orientation_error)
        self.update(throttle, delta)
        if lateral_error < 0.05 and orientation_error < 0.01:
            self.start_overtake_complete = True

    def stanley_controller(self, lateral_error, orientation_error):
        """
        Computes steering given by Stanley controller
        :param lateral_error: the vehicle lateral error with respect to the reference lane
        :param orientation_error: the vehicle orientation error with respect to the reference lane
        """
        k_e = 1.5
        k_v = 20.0

        yaw_diff_crosstrack = np.arctan(k_e * lateral_error / (k_v + self.v))
        delta = orientation_error + yaw_diff_crosstrack
        if delta > np.pi:
            delta -= 2 * np.pi
        if delta < - np.pi:
            delta += 2 * np.pi
        return delta



    def perform_overtake(self, x_other, y_other, yaw_other, v_other):
        """
        Behavioral mode: start_overtake. Computes next pose
        """

        self.mode = 'perform_overtake'
        if self.cumulative_overtake_time < 4.0:
            self.cumulative_overtake_time += self.dt

        else:
            self.overtake_tried = True
            self.case_overtake = - self.case_overtake
            self.cumulative_overtake_time = 0.0

        if self.case_overtake == 1:
            throttle = self.max_a
        else:
            throttle = self.min_a

        delta = 0.0
        self.update(throttle, delta)