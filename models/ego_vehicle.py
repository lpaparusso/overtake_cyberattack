import numpy as np
from models.bicyclemodel import ConstrainedLinearBicycleModel

class EgoVehicle(ConstrainedLinearBicycleModel):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super().__init__(x, y, yaw, v)
        self.x_other = None
        self.y_other = None
        self.yaw_other = None
        self.v_other = None
        self.mode = None

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

        long_distance_other = np.abs(x_other - self.x)

        if self.mode is None:
            self.steady()
            return None
        if self.mode == 'steady':
            if long_distance_other > 2.0:
                self.brake()
                return None
            else:
                self.steady()
                return None
        if self.mode == 'brake':
            if long_distance_other > 0.1:
                self.brake()
                return None
            else:
                self.danger(x_other, y_other, yaw_other, v_other)
                return None
        if self.mode == 'danger':
            self.danger(x_other, y_other, yaw_other, v_other)
            print(self.throttle)
            return None

    def steady(self):
        """
        Behavioral mode: steady. Computes next pose
        """

        self.mode = 'steady'
        throttle = 0.0
        delta = 0.0
        self.update(throttle, delta)

    def brake(self):
        """
        Behavioral mode: brake. Computes next pose
        """

        self.mode = 'brake'
        throttle = - 0.1
        delta = 0.0
        self.update(throttle, delta)

    def danger(self, x_other, y_other, yaw_other, v_other):
        """
        Behavioral mode: danger. Computes next pose given the current surrounding agent state
        :param x_other: global x of the other agent
        :param y_other: global y of the other agent
        :param yaw_other: global yaw of the other agent
        :param v_other: global speed of the other agent
        """

        self.mode = 'danger'
        pos_error = x_other - self.x
        vel_error = v_other - self.v
        throttle = self.pid(pos_error, vel_error)
        delta = 0.0
        self.update(throttle, delta)

    def pid(self, pos_error, vel_error):
        throttle = 1000 * pos_error# + 10 * vel_error
        return throttle
