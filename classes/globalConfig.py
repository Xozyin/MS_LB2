class GlobalConfig():
    def __init__(self, x0: list[float], T_interval, c: float, u: float, h_tv: float, g: float):
        self.x0: list[float] = x0
        self.T_interval = T_interval
        self.c: float = c
        self.u: float = u
        self.h_tv: float = h_tv
        self.g: float = g

    def return_params_str(self) -> str:
        return f"X0 = {self.x0}\nT = {self.T_interval}\nc = {self.c}\nu = {self.u}\nh_tv = {self.h_tv}\ng = {self.g}"