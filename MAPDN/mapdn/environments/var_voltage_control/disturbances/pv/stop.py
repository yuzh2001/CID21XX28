# from mapdn.environments.var_voltage_control.disturbances import DisturbanceBase
# from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl


class PVStop:
    """
    Stop one or multiple PVs

    disturbance_args: dict = {"pv_id": [1, 2]}
    """

    def __init__(self, env, disturbance_args: dict):
        # super().__init__(env, disturbance_args)
        self.type = "pv_stop"
        self.env = env
        self.disturbance_args = disturbance_args

    def start(self):
        self.env = self.env  # activate python type inference

        # update the record in the pandapower
        # self.env.powergrid.sgen["p_mw"] = self.env.powergrid.sgen["p_mw"] * self.disturbance_args["multiplier"]
        # self.env.powergrid.load["p_mw"] = (
        #     self.env.powergrid.load["p_mw"] * self.disturbance_args["multiplier"]
        # )
        # self.env.powergrid.load["q_mvar"] = (
        #     self.env.powergrid.load["q_mvar"] * self.disturbance_args["multiplier"]
        # )
        # rich.print(self.env.powergrid.load["q_mvar"])

    def end(self):
        self.env = self.env
        # pass
