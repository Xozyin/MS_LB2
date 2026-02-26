from classes import GlobalConfig, PointInTime
import math
import numpy as np

def model4(globConf: GlobalConfig, x_vect: list[float]):
    c = globConf.c
    u = globConf.u
    h_tv = globConf.h_tv
    g = globConf.g

    # Параметр r зависит от x1
    r = 0.1 * np.exp(-x_vect[0] / h_tv)
    
    dx = []
    dx.append(x_vect[1])  # dx1/dt = x2
    # dx2/dt = (c*u)/x3 - g - (r * x2^2)/x3
    dx.append(((c * u) / x_vect[2]) - g - ((r * (x_vect[1])**2) / x_vect[2]))
    dx.append(-u)  # dx3/dt = -u
    
    return dx

def runge_kut(globConf: GlobalConfig, func, h: float):
    x0 = globConf.x0
    t0, T = globConf.T_interval

    len_x0 = len(x0)
    
    # Количество шагов
    N = int(math.ceil((T - t0) / h))
    h = (T - t0) / N  # корректируем шаг
    
    # Массив для хранения результатов
    final_res: list[PointInTime] = []
    
    # Начальное состояние
    final_res.append(PointInTime(x_list=x0.copy(), time=t0))
    
    # Основной цикл интегрирования
    for i in range(N):
        x_cur = final_res[i].x_list
        t_cur = final_res[i].time
        
        k1 = func(globConf, x_cur)
        
        # Для k2 используем x_cur + k1/2
        x_temp = [x_cur[j] + (h/2) * k1[j] for j in range(len_x0)]
        k2 = func(globConf, x_temp)
        
        # Для k3 используем x_cur + k2/2
        x_temp = [x_cur[j] + (h/2) * k2[j] for j in range(len_x0)]
        k3 = func(globConf, x_temp)
        
        # Для k4 используем x_cur + k3
        x_temp = [x_cur[j] + k3[j] for j in range(len_x0)]
        k4 = func(globConf, x_temp)
        
        # Новое состояние
        x_new = [x_cur[j] + (h/6) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) for j in range(len_x0)]
        final_res.append(PointInTime(x_list=x_new, time=t_cur + h))
    
    return final_res