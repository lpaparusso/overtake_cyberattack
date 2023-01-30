import numpy as np
from models.bicyclemodel import ConstrainedLinearBicycleModel

class OtherVehicle(ConstrainedLinearBicycleModel):
    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super().__init__(params, x, y, yaw, v)
        self.mode = None
        self.change_lane_complete = False
        self.cumulative_overtake_time = 0.0
        self.case_overtake = 1
        self.overtake_tried = False

    def next(self, x_other, y_other, yaw_other, v_other):
        """
        Computes next pose using a state machine
        :param x_other: global x of the other agent
        :param y_other: global y of the other agent
        :param yaw_other: global yaw of the other agent
        :param v_other: global speed of the other agent
        """

        long_distance_other = self.x - x_other

        if self.mode is None:
            self.steady()
            return None
        if self.mode == 'steady':
            if np.abs(long_distance_other) > 7.0:
                self.steady()
                return None
            else:
                self.change_lane()
                return None
        if self.mode == 'change_lane':
            if self.change_lane_complete is False:
                self.change_lane()
                return None
            else:
                self.perform_overtake(x_other, y_other, yaw_other, v_other)
                return None
        if self.mode == 'perform_overtake':
            if self.overtake_tried is False:
                self.perform_overtake(x_other, y_other, yaw_other, v_other)
                if long_distance_other > 6.0:
                    return 0
                else:
                    return None
            else:
                self.perform_overtake(x_other, y_other, yaw_other, v_other)
                if np.abs(long_distance_other) > 6.0:
                    return 0
                else:
                    return None

    def steady(self):
        """
        Behavioral mode: steady. Computes next pose
        """

        self.mode = 'steady'
        throttle = 0.0
        delta = 0.0
        self.update(throttle, delta)

    def change_lane(self):
        """
        Behavioral mode: change_lane. Computes next pose
        """

        self.mode = 'change_lane'
        throttle = 0.0
        lateral_error = 2.5 - self.y
        orientation_error = - self.yaw
        delta = self.stanley_controller(lateral_error, orientation_error)
        self.update(throttle, delta)
        if lateral_error < 0.01 and orientation_error < 0.01:
            self.change_lane_complete = True

    def stanley_controller(self, lateral_error, orientation_error):
        """
        Computes steering given by Stanley controller
        :param lateral_error: the vehicle lateral error with respect to the reference lane
        :param orientation_error: the vehicle orientation error with respect to the reference lane
        """
        k_e = 2.0
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
        dt = self.params['dt']
        max_a = self.params['max_a']
        min_a = self.params['min_a']

        self.mode = 'perform_overtake'
        if self.cumulative_overtake_time < 4.0:
            self.cumulative_overtake_time += dt

        else:
            self.overtake_tried = True
            self.case_overtake = - self.case_overtake
            self.cumulative_overtake_time = 0.0

        if self.case_overtake == 1:
            throttle = max_a
        else:
            throttle = min_a

        lateral_error = 2.5 - self.y
        orientation_error = - self.yaw
        delta = self.stanley_controller(lateral_error, orientation_error)
        self.update(throttle, delta)