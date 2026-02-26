from .globalConfig import GlobalConfig
from .pointInTime import PointInTime
from utilities import model4, runge_kut
from matplotlib.widgets import Button, TextBox, RadioButtons
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import time

class FormPainter():
    def __init__(self, globalConfig: GlobalConfig):
        self.globalConfig = globalConfig
        self.current_mode = "auto"
        self.example_h_vals = [2, 1.5, 1, 0.5, 0.1, 0.05, 0.02, 0.005, 0.002]

        self.fig = plt.figure(figsize=(16, 10), num="МС ЛБ2 - Каневский Глеб 23ВП2")
        self.gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 2.5, 1.3], wspace=0.3, left=0.05, right=0.95)
        self.ax_input = self.fig.add_subplot(self.gs[0])
        self.ax_input.axis('off')
        
        self.ax_plots = self.fig.add_subplot(self.gs[1])
        self.ax_plots.axis('off')
        
        self.ax_results = self.fig.add_subplot(self.gs[2])
        self.ax_results.axis('off')

        self.radio_switch = None
        self.h_input = None
        self.fin_values_text = None
        self.fin_values_text_lable = None
        self.step_value = None
        self.step_value_label = None
        self.delta_value = None
        self.delta_value_label = None
        self.etalon_h_box_axes = None
        self.etalon_h_input = None
        self.etalon_label = None
        self.buttonStart = None
        self.buttonEnd = None
        self.history_label = None
        self.iter_table_axes = None
        self.analysis_table_axes = None
        self.plot_axes = []

        # Формирование элементов
        t0, T = self.globalConfig.T_interval
        # КОЛОНКА 1 - ИНПУТ
        # LABEL - Режим
        self.ax_input.text(0.01, 1.1, 'Выберите режим:', 
            ha='left', fontsize=14, fontweight='bold', transform=self.ax_input.transAxes)
        
        # RADIO BUTTONS
        rax = plt.axes([0.06, 0.85, 0.15, 0.08])
        self.radio_switch = RadioButtons(rax, ('Авто SIGMA <= 1%', 'Фикс. h = 0.001'), active=0)

        self.etalon_label = self.ax_input.text(0.01, 0.9, 'Эталонный h:', 
            ha='left', fontsize=12, fontweight='bold', transform=self.ax_input.transAxes)
        self.etalon_label.set_visible(False)

        self.etalon_h_box_axes = plt.axes([0.16, 0.79, 0.05, 0.05])
        self.etalon_h_box_axes.set_visible(False)
        self.etalon_h_input = TextBox(self.etalon_h_box_axes, '', initial='0.01')
        
        # ПАРАМЕТРЫ МОДЕЛИ
        self.ax_input.text(0.01, 0.83, 'ПАРАМЕТРЫ МОДЕЛИ', 
            ha='left', fontsize=12, fontweight='bold', transform=self.ax_input.transAxes)
        self.ax_input.text(0.01, 0.60, self.globalConfig.return_params_str(), 
            ha='left', fontsize=11, transform=self.ax_input.transAxes)
        
        # Начальные условия
        self.ax_input.text(0.01, 0.5, 'НАЧАЛЬНЫЕ УСЛОВИЯ', 
            ha='left', fontsize=12, fontweight='bold', transform=self.ax_input.transAxes)
        
        init_text = ""
        for i in range(len(self.globalConfig.x0)):
            init_text += f"X{i+1}(0) = {self.globalConfig.x0[i]}\n"    
        self.ax_input.text(0.01, 0.35, init_text, ha='left', fontsize=11,
            transform=self.ax_input.transAxes)
        
        # Создаем область для текстового ввода (в координатах фигуры)
        axbox = plt.axes([0.1, 0.3, 0.1, 0.05])
        self.h_input = TextBox(axbox, '', initial='0.01')
        self.h_input.label.set_visible(False)
        
        # Статус
        status_text = self.ax_input.text(0.1, 0.0, '', ha='center', fontsize=12, 
            color='blue', transform=self.ax_input.transAxes)
        
        axbutton_start = plt.axes([0.05, 0.24, 0.07, 0.05])
        self.buttonStart = Button(axbutton_start, 'СТАРТ', color='white', hovercolor='green')
        
        axbutton_end = plt.axes([0.13, 0.24, 0.07, 0.05])
        self.buttonEnd = Button(axbutton_end, 'ЗАКРЫТЬ', color='white', hovercolor='red')
        
        # Поле ввода шага
        self.ax_input.text(0.01, 0.27, 'Шаг h:', 
                    ha='left', fontsize=11, fontweight='bold', transform=self.ax_input.transAxes)

        # КОЛОНКА 3 - РЕЗУЛЬТАТЫ
        self.ax_results.text(0.5, 1.1, 'РЕЗУЛЬТАТЫ РАСЧЕТА:', 
            ha='center', fontsize=14, fontweight='bold', transform=self.ax_results.transAxes)
        
        # Блок с информацией о шаге
        self.step_value_label = self.ax_results.text(0.01, 1.04, 'ШАГ ИНТЕГРИРОВАНИЯ:', 
            ha='left', fontsize=11, fontweight='bold', transform=self.ax_results.transAxes)
        self.step_value = self.ax_results.text(0.01, 1.0, '', ha='left', fontsize=12, 
            transform=self.ax_results.transAxes)
        
        # Блок с погрешностью
        self.delta_value_label = self.ax_results.text(0.01, 0.94, 'ОТНОСИТЕЛЬНАЯ ПОГРЕШНОСТЬ:', 
            ha='left', fontsize=11, fontweight='bold', transform=self.ax_results.transAxes)
        self.delta_value = self.ax_results.text(0.01, 0.9, '', ha='left', fontsize=12,
            transform=self.ax_results.transAxes, fontweight='bold', color='red')
        
        # Блок со значениями переменных
        self.fin_values_text_lable = self.ax_results.text(0.01, 0.84, f"ЗНАЧЕНИЯ В КОНЕЧНОЙ ТОЧКЕ:", 
            ha='left', fontsize=11, fontweight='bold', transform=self.ax_results.transAxes)
        self.fin_values_text = self.ax_results.text(0.01, 0.68, '', ha='left', fontsize=12,
            transform=self.ax_results.transAxes)
        
        # Заголовок для таблицы
        self.history_label = self.ax_results.text(0.5, 0.9, 'ИСТОРИЯ ПОДБОРА ШАГА:', 
            ha='center', fontsize=10, fontweight='bold', 
            transform=self.ax_results.transAxes)
        self.history_label.set_visible(False)

        # Кнопка для вывода Анализа
        axbutton_anal = plt.axes([0.05, 0.18, 0.15, 0.05])
        self.buttonAnal = Button(axbutton_anal, 'Хочу посмотреть анализ!', color='white', hovercolor='yellow')

        # Выбор режима
        def mode_changed(event):
            if self.current_mode == "fixed":
                self.etalon_h_box_axes.set_visible(False)
                self.etalon_label.set_visible(False)
                self.current_mode = "auto"
                status_text.set_text("Режим: автоматический подбор шага")
            else:
                self.etalon_h_box_axes.set_visible(True)
                self.etalon_label.set_visible(True)
                self.current_mode = "fixed"
                status_text.set_text("От-но эталонного h")
            self.fig.canvas.draw()
        
        # КНОПКА СТАРТ
        def on_start(event):
            self.set_elements(vis_flag=True)

            try:
                h_user = float(self.h_input.text)
                if h_user <= 0:
                    status_text.set_text("Ошибка: шаг должен быть положительным!")
                    self.fig.canvas.draw()
                    return
                elif h_user >= T:
                    status_text.set_text(f"Ошибка: шаг должен быть меньше {T}!")
                    self.fig.canvas.draw()
                    return

                if self.current_mode == "fixed":
                    h_etalon = float(self.etalon_h_input.text)
                    if h_etalon <= 0:
                        status_text.set_text("Ошибка: эталонный шаг должен быть положительным!")
                        self.fig.canvas.draw()
                        return
                    elif h_etalon >= T:
                        status_text.set_text(f"Ошибка: эталонный шаг должен быть меньше {T}!")
                        self.fig.canvas.draw()
                        return
                
                status_text.set_text(f"Выполняется расчет...")
                self.fig.canvas.draw()
                
                if self.current_mode == "fixed":
                    results = runge_kut(self.globalConfig, model4, h_user)
                else:
                    h_user, results = self.auto_solution(h_user=h_user)
                    h_etalon = None
                
                # Обновлить графики
                self.update_plots(results, h_user)
                self.update_results(results, h_user, h_etalon)
                status_text.set_text(f"Расчет завершен!")
                self.fig.canvas.draw()
                
            except ValueError:
                status_text.set_text("Ошибка: введите число! Проверьте ввод!")
                self.fig.canvas.draw()
        
        # КНОПКА ЗАКРЫТЬ
        def on_end(event):
            plt.close('all')

        # КНОПКА АНАЛИЗ
        def on_analyze(event):
            self.set_elements(False)
            self.analyze_dependence(self.example_h_vals)

        self.radio_switch.on_clicked(mode_changed)
        self.buttonStart.on_clicked(on_start)
        self.buttonEnd.on_clicked(on_end)
        self.buttonAnal.on_clicked(on_analyze)

        plt.tight_layout()
        plt.show()

    def update_plots(self, results: list[PointInTime], h_user: float):
        self.ax_plots.clear()
        if hasattr(self, 'plot_axes'):
            for ax in self.plot_axes:
                ax.remove()
        self.plot_axes = []
        self.ax_plots.axis('off')

        count_plots = len(self.globalConfig.x0)
        gs_plots = gridspec.GridSpecFromSubplotSpec(count_plots, 1, subplot_spec=self.gs[1], hspace=0.3)
        colors_lst = ['b-', 'r-', 'g-']

        times = [p.time for p in results]

        for i in range(count_plots):
            new_plot = self.fig.add_subplot(gs_plots[i])
            self.plot_axes.append(new_plot)
            color = colors_lst[i % len(colors_lst)]
            new_plot.plot(times, [p.x_list[i] for p in results], color, linewidth=1.5)
            new_plot.set_ylabel(f"X{i + 1}", fontsize=10, fontweight='bold')
            new_plot.grid(True, alpha=0.3, linestyle='--')
            new_plot.ticklabel_format(style='sci', scilimits=(-3,3), axis='y', useOffset=False)
            new_plot.tick_params(axis='x', labelbottom=False)

        self.plot_axes[0].set_title(f'Решение с шагом h = {h_user}', fontsize=11, fontweight='bold')
        self.plot_axes[count_plots - 1].set_xlabel('Время t, с', fontsize=10, fontweight='bold')
        
        self.fig.canvas.draw()

    def update_results(self, results: list[PointInTime], h_user: float, h_etalon: float = None):
        t0, T = self.globalConfig.T_interval
        x1_vals = [p.x_list[0] for p in results]
        
        # Обновление значений
        self.step_value.set_text(f'h = {h_user:.6f}')

        init_text = ""
        for i in range(len(self.globalConfig.x0)):
            init_text += f"X{i+1}({T}) = {results[-1].x_list[i]:.6f}\n"
        self.fin_values_text.set_text(init_text)
        
        # Обновление прогрешности
        try:
            if self.current_mode == "fixed":
                results_et = runge_kut(self.globalConfig, model4, h_etalon)
                x1_et = results_et[-1].x_list[0]
                delta = abs((x1_et - x1_vals[-1]) / x1_et) * 100
                self.delta_value.set_text(f'{delta:.6f}%')
            else:
                results_2h = runge_kut(self.globalConfig, model4, h_user/2)
                x1_et = results_2h[-1].x_list[0]
                delta = abs((x1_et - x1_vals[-1]) / x1_et) * 100
                self.delta_value.set_text(f'{delta:.6f}%')
            
            if delta <= 1.0:
                self.delta_value.set_color('green')
            else:
                self.delta_value.set_color('red')
        except:
            self.delta_value.set_text('Ошибка вычисления')
        
        self.fig.canvas.draw()

    def auto_solution(self, h_user: float) -> tuple[float, list[PointInTime]]:
        h_current = h_user
        max_iter = 20
        iter_count = 0
        delta_current = 100.0
        
        iter_history_dct = {
            'iter': [],
            'h': [],
            'X1': [],
            'delta': [],
            'X1_half': []
        }
        while delta_current > 1.0 and iter_count < max_iter:
            iter_count += 1
            
            results_cur = runge_kut(self.globalConfig, model4, h_current)
            x1_cur = results_cur[-1].x_list[0]
            
            # Шаг h/2
            h_half = h_current / 2
            results_half = runge_kut(self.globalConfig, model4, h_half)
            x1_half = results_half[-1].x_list[0]
            
            delta_current = abs((x1_half - x1_cur) / x1_half) * 100

            iter_history_dct['iter'].append(iter_count)
            iter_history_dct['h'].append(h_current)
            iter_history_dct['X1'].append(x1_cur)
            iter_history_dct['delta'].append(delta_current)
            iter_history_dct['X1_half'].append(x1_half)
            
            if delta_current > 1.0:
                h_current = h_half
        
        h_final = h_current
        results_final = runge_kut(self.globalConfig, model4, h_final)
        self.create_iteration_table(iter_history_dct)

        return (h_final, results_final)
    
    def create_iteration_table(self, iter_history_dct):
        try:
            if hasattr(self, 'iter_table_axes'):
                self.iter_table_axes.remove()
        except:
            pass

        # Оси для таблицы
        self.iter_table_axes = self.fig.add_axes([0.76, 0.15, 0.2, 0.25])
        self.iter_table_axes.axis('off')
        
        table_data = []
        lst_iter = iter_history_dct['iter']
        lst_h = iter_history_dct['h']
        lst_x1 = iter_history_dct['X1']
        lst_delta = iter_history_dct['delta']

        for i in range(len(lst_iter)):
            table_data.append([
                f"{lst_iter[i]}",
                f"{lst_h[i]:.6f}",
                f"{lst_x1[i]:.6f}",
                f"{lst_delta[i]:.4f}%"
            ])
        
        table = self.iter_table_axes.table(
            cellText=table_data,
            colLabels=['Итер', 'h', 'X1', 'ОТ.П%'],
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.3, 0.3, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # ЗАКРАШИВАНИЕ
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('grey')
                cell.set_text_props(color='white', weight='bold')
            elif i == len(lst_iter):
                cell.set_facecolor('green')
        
        self.fig.canvas.draw()

    def analyze_dependence(self, h_values: list[float]):        
        iter_history_dct = {
            'iter': [],
            'h': [],
            'delta': [],
            'time': [],
            'num_steps': [],
            'X1': []
        }
        
        results_et = runge_kut(self.globalConfig, model4, 0.001)
        x1_et = results_et[-1].x_list[0]
        
        for i in range(len(h_values)):
            start_time = time.time()
            results = runge_kut(self.globalConfig, model4, h_values[i])
            elapsed_time = time.time() - start_time + 0.0001
            
            x1_h = results[-1].x_list[0]
            num_steps = len(results) - 1
            delta = abs((x1_et - x1_h) / x1_et) * 100
            
            iter_history_dct['iter'].append(i+1)
            iter_history_dct['h'].append(h_values[i])
            iter_history_dct['delta'].append(delta)
            iter_history_dct['time'].append(elapsed_time)
            iter_history_dct['num_steps'].append(num_steps)
            iter_history_dct['X1'].append(x1_h)
        
        if hasattr(self, 'plot_axes'):
            for ax in self.plot_axes:
                ax.remove()
        self.plot_axes = []
        self.ax_plots.clear()
        self.ax_plots.axis('on')
        
        gs_plots = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.gs[1], hspace=0.4)
        
        # График погрешности
        ax1 = self.fig.add_subplot(gs_plots[0])
        self.plot_axes.append(ax1)
        ax1.loglog(h_values, iter_history_dct['delta'], 'bo-', linewidth=2, markersize=8)
        
        # Номера точек
        for i, (h, delta) in enumerate(zip(h_values, iter_history_dct['delta']), 1):
            ax1.annotate(str(i), (h, delta), 
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='blue', alpha=0.9))
        
        ax1.set_xlabel('Шаг h')
        ax1.set_ylabel('Отн погрешность, %')
        ax1.set_title('Зависимость погрешности от шага')
        ax1.grid(True, which='both', alpha=0.3)
        
        # График времени
        ax2 = self.fig.add_subplot(gs_plots[1])
        self.plot_axes.append(ax2)
        ax2.loglog(h_values, iter_history_dct['time'], 'ro-', linewidth=2, markersize=8) 
        # Номера точек
        for i, (h, t) in enumerate(zip(h_values, iter_history_dct['time']), 1):
            ax2.annotate(str(i), (h, t), 
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red', alpha=0.9))
        
        ax2.set_xlabel('Шаг h')
        ax2.set_ylabel('Время, с')
        ax2.set_title('Зависимость времени вычислений от шага')
        ax2.grid(True, which='both', alpha=0.3)
        
        # Таблица в правой колонке
        self.create_analysis_table(iter_history_dct)
        
        self.fig.canvas.draw()

    def create_analysis_table(self, iter_history_dct: dict):
        try:
            if hasattr(self, 'analysis_table_axes'):
                self.analysis_table_axes.remove()
        except:
            pass
        
        self.analysis_table_axes = self.fig.add_axes([0.76, 0.3, 0.2, 0.35])
        self.analysis_table_axes.axis('off')
        
        # ТАБЛИЦА ШАГОВ        
        table_data = []
        lst_iter = iter_history_dct['iter']
        lst_h = iter_history_dct['h']
        lst_delta = iter_history_dct['delta']
        lst_time = iter_history_dct['time']
        lst_num_steps = iter_history_dct['num_steps']

        lst_x1 = iter_history_dct['X1']
        for i in range(len(lst_iter)):
            table_data.append([
                f"{lst_iter[i]}",
                f"{lst_h[i]:.4f}",
                f"{lst_delta[i]:.4f}%",
                f"{lst_time[i]:.4f}c",
                f"{lst_num_steps[i]:}",
                f"{lst_x1[i]:.4f}"
            ])

        table = self.analysis_table_axes.table(
            cellText=table_data,
            colLabels=['№', 'h', 'ОП%', 'Время', 'Шаги', 'X1(T)'],
            cellLoc='center',
            loc='center',
            colWidths=[0.08, 0.18, 0.16, 0.16, 0.14, 0.18]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.2, 1.5)
        
        # Закраш заголовки
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('grey')
                cell.set_text_props(color='white', weight='bold')
            elif j == 2:  # Колонка с погрешностью
                value = float(cell.get_text().get_text().replace('%', ''))
                if value <= 1.0:
                    cell.set_facecolor('green')
                elif value <= 5.0:
                    cell.set_facecolor('yellow')
                else:
                    cell.set_facecolor('orange') 
        
        # Инфо эталон
        t0, T = self.globalConfig.T_interval
        self.analysis_table_axes.text(0.5, -0.1, f'Эталон: x1({T}) с h=0.001', 
            ha='center', fontsize=7, transform=self.analysis_table_axes.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        self.fig.canvas.draw()

    def set_elements(self, vis_flag: bool):
        self.step_value.set_visible(vis_flag)
        self.step_value_label.set_visible(vis_flag)
        self.delta_value.set_visible(vis_flag)
        self.delta_value_label.set_visible(vis_flag)
        self.fin_values_text.set_visible(vis_flag)
        self.fin_values_text_lable.set_visible(vis_flag)
        self.history_label.set_visible(not vis_flag)

        try:
            if hasattr(self, 'iter_table_axes'):
                self.iter_table_axes.remove()
        except:
            pass

        try:
            if hasattr(self, 'analysis_table_axes'):
                self.analysis_table_axes.remove()
        except:
            pass