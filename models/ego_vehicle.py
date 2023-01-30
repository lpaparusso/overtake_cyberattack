import numpy as np
from models.bicyclemodel import ConstrainedLinearBicycleModel

class EgoVehicle(ConstrainedLinearBicycleModel):
    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super().__init__(params, x, y, yaw, v)
        self.mode = None

    def next(self, x_other, y_other, yaw_other, v_other, a_other):
        """
        Computes next pose using a state machine
        :param x_other: global x of the other agent
        :param y_other: global y of the other agent
        :param yaw_other: global yaw of the other agent
        :param v_other: global speed of the other agent
        """

        long_distance_other = np.abs(x_other - self.x)

        if self.mode is None:
            self.steady()
            return None
        if self.mode == 'steady':
            if long_distance_other > 2.0:
                self.steady()
                return None
            else:
                self.danger(x_other, y_other, yaw_other, v_other, a_other)
                return None
        if self.mode == 'danger':
            self.danger(x_other, y_other, yaw_other, v_other, a_other)
            return None

    def steady(self):
        """
        Behavioral mode: steady. Computes next pose
        """

        self.mode = 'steady'
        throttle = 0.0
        delta = 0.0
        self.update(throttle, delta)

    def danger(self, x_other, y_other, yaw_other, v_other, a_other):
        """
        Behavioral mode: danger. Computes next pose given the current surrounding agent state
        :param x_other: global x of the other agent
        :param y_other: global y of the other agent
        :param yaw_other: global yaw of the other agent
        :param v_other: global speed of the other agent
        :param a_other: global acceleration of the other agent
        """

        self.mode = 'danger'
        pos_error = x_other - self.x
        vel_error = v_other - self.v
        throttle = self.pid(pos_error, vel_error, a_other)
        delta = 0.0
        self.update(throttle, delta)

    def pid(self, pos_error, vel_error, a_other):
        throttle = a_other + 0.25 * pos_error + 0.5 * vel_error
        return throttle
