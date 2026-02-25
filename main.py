from formPainter import FormPainter
from classes import GlobalConfig, PointInTime
from utilities import model4, runge_kut, print_dct, find_relative_error
from matplotlib.widgets import Button, TextBox
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
import time

def punkt1(globalConfig: GlobalConfig):
    fp = FormPainter(globalConfig) 

def punkt2(globalConfig: GlobalConfig, h_values: list[float]):
    t0, T = globalConfig.T_interval

    print("\nПУНКТ 2: Анализ зависимости точности и трудоемкости")
    delta_values = []
    time_values = []
    steps_values = []

    print("Вычисление эталонного решения:")
    start_et = time.time()
    results_et = runge_kut(globalConfig, model4, 0.001)
    time_et = time.time() - start_et
    x1_et = results_et[-1].x_list[0]
    print(f"Эталонное x1({T}) = {x1_et:.6f} (время: {time_et:.4f} с)")

    print("\nАнализ для разных шагов:")
    print(f"{'h':<10} {'x1(T)':<15} {'Погрешность %':<15} {'Время, с':<12} {'Шагов':<10}")

    for h in h_values:
        start_time = time.time()
        results = runge_kut(globalConfig, model4, h)
        elapsed_time = time.time() - start_time + 0.0001
        
        # Значение x1 в конце
        x1_h = results[-1].x_list[0]
        
        # Количество шагов
        num_steps = len(results) - 1
        
        # Погрешность относительно эталона
        delta = abs((x1_et - x1_h) / x1_et) * 100
        
        delta_values.append(delta)
        time_values.append(elapsed_time)
        steps_values.append(num_steps)
        
        print(f"{h:<10.4f} {x1_h:<15.6f} {delta:<15.6f} {elapsed_time:<12.4f} {num_steps:<10}")

    # Построение графиков для пункта 2
    plt.figure(2, figsize=(14, 10))

    # График зависимости погрешности от шага
    plt.subplot(2, 2, 1)
    plt.loglog(h_values, delta_values, 'bo-', linewidth=2, markersize=8)
    for i, (h, delta) in enumerate(zip(h_values, delta_values), 1):
        plt.annotate(str(i), (h, delta), 
        textcoords="offset points", 
        xytext=(5, 5), 
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='blue', alpha=0.9))
    plt.xlabel('Шаг h')
    plt.ylabel('Относительная погрешность SIGMA, %')
    plt.title('Зависимость погрешности от шага')
    plt.grid(True, which='both', alpha=0.3)

    # График зависимости времени от шага
    plt.subplot(2, 2, 2)
    plt.loglog(h_values, time_values, 'ro-', linewidth=2, markersize=8)
    for i, (h, t) in enumerate(zip(h_values, time_values), 1):
        plt.annotate(str(i), (h, t), 
                    textcoords="offset points", 
                    xytext=(5, 5), 
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red', alpha=0.9))
    plt.xlabel('Шаг h')
    plt.ylabel('Время вычислений, с')
    plt.title('Зависимость трудоемкости от шага')
    plt.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.show()

def punkt3(globalConfig: GlobalConfig, h_start: float):    
    print("\nПУНКТ 3: Автоматический выбор шага интегрирования (SIGMA <= 1 %)")
    h_current = h_start
    max_iter = 20
    iter_count = 0
    delta_current = 100.0
    
    iter_history = []
    
    print(f"Начальный шаг: h = {h_current}")
    print(f"\n{'Итерация':<10} {'h':<15} {'x1(T)':<15} {'Погрешность %':<15} {'Решение с h/2':<15}")
    
    while delta_current > 1.0 and iter_count < max_iter:
        iter_count += 1
        results_cur = runge_kut(globalConfig, model4, h_current)
        x1_cur = results_cur[-1].x_list[0]
        
        # Шагом h/2
        h_half = h_current / 2
        results_half = runge_kut(globalConfig, model4, h_half)
        x1_half = results_half[-1].x_list[0]
        
        delta_current = abs((x1_half - x1_cur) / x1_half) * 100
        iter_history.append({
            'iter': iter_count,
            'h': h_current,
            'x1': x1_cur,
            'delta': delta_current,
            'x1_half': x1_half
        })
        
        print(f"{iter_count:<10} {h_current:<15.6f} {x1_cur:<15.6f} {delta_current:<15.6f} {x1_half:<15.6f}")
        
        if delta_current > 1.0:
            h_current = h_half
    
    # Итоговый
    h_final = h_current
    
    print("\n\nРЕЗУЛЬТАТ ПОДБОРА:")
    print(f"Подобранный шаг: h = {h_final:.8f}")
    print(f"Количество итераций: {iter_count}")
    print(f"Достигнутая относительная погрешность: SIGMA = {delta_current:.6f}% (<= 1%)")
    
    # Финальное решение с найденным шагом
    print("\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    results_final = runge_kut(globalConfig, model4, h_final)
    
    # Извлекаем данные для графиков
    times_final = [p.time for p in results_final]
    x1_final_vals = [p.x_list[0] for p in results_final]
    x2_final_vals = [p.x_list[1] for p in results_final]
    x3_final_vals = [p.x_list[2] for p in results_final]

    # График ПУНКТ 3
    plt.figure(3, figsize=(12, 8))
    plt.plot(times_final, x1_final_vals, 'b-', linewidth=1.5, label='X1')
    plt.plot(times_final, x2_final_vals, 'r-', linewidth=1.5, label='X2')
    plt.plot(times_final, x3_final_vals, 'g-', linewidth=1.5, label='X3')
    plt.xlabel('Время t, с')
    plt.ylabel('Значения переменных')
    plt.title(f'Финальное решение с шагом h = {h_final:.6f} (SIGMA = {delta_current:.4f}%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    x0 = [0.0, 0.0, 1100.0]
    T_interval = (0.0, 11.0)
    c = 8000
    u = 20
    h_tv = 9900
    g = 9.81

    globConf = GlobalConfig(x0=x0, T_interval=T_interval, c=c, u=u, h_tv=h_tv, g=g) 

    punkt1(globalConfig=globConf)
    #punkt2(globalConfig=globConf, h_values=[2, 1.5, 1, 0.5, 0.1, 0.05, 0.02, 0.005, 0.002])
    #punkt3(globalConfig=globConf, h_start=3)

main()