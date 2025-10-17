#!/usr/bin/env python3
"""
Monte Carlo bakery newsvendor simulation with a Tkinter UI, results table and plots.
Save as: monte_carlo_bakery.py
Run: python3 monte_carlo_bakery.py

Requirements:
- Python 3
- matplotlib (install: pip install matplotlib)

Features:
- Input mean demand, std dev, price p, cost c, salvage v, days (trials)
- Computes newsvendor critical fractile alpha and Q* = mean + z(alpha)*sigma (uses Acklam inverse CDF)
- Runs Monte Carlo simulation of daily demand (normal distribution rounded to integers)
- Shows average daily profit, average excess loss and average shortage loss
- Shows a detailed table (one row per simulated day) with export to CSV
- Shows interactive plots (demand over days with Q line, and profit histogram)

No external libraries required except matplotlib for plotting (optional but recommended).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import math
import csv

# Try to import matplotlib; handle if missing
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def inverse_normal_cdf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
        (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0))


def critical_fractile(p: float, c: float, v: float) -> float:
    if p == v:
        return 1.0
    return (p - c) / (p - v)


def simulate_with_details(mean, std, p, c, v, Q, days, seed=None):
    rnd = random.Random(seed)
    total_profit = 0.0
    total_excess_loss = 0.0
    total_shortage_loss = 0.0
    rows = []

    for day in range(1, days + 1):
        d_double = mean + rnd.gauss(0, 1) * std
        d = max(0, int(round(d_double)))
        sold = min(d, Q)
        leftover = max(0, Q - d)
        unmet = max(0, d - Q)

        revenue = sold * p + leftover * v
        cost = Q * c
        profit = revenue - cost

        excess_loss = leftover * (c - v)
        shortage_loss = unmet * (p - c)

        total_profit += profit
        total_excess_loss += excess_loss
        total_shortage_loss += shortage_loss

        rows.append({
            'day': day,
            'demand': d,
            'sold': sold,
            'leftover': leftover,
            'unmet': unmet,
            'revenue': revenue,
            'cost': cost,
            'profit': profit,
            'excess_loss': excess_loss,
            'shortage_loss': shortage_loss
        })

    avg_profit = total_profit / days
    avg_excess_loss = total_excess_loss / days
    avg_shortage_loss = total_shortage_loss / days

    return avg_profit, avg_excess_loss, avg_shortage_loss, rows


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Monte Carlo — Bakery Newsvendor (с графиком)')
        self.resizable(False, False)

        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0)

        # Inputs
        self.mean_var = tk.StringVar(value='2500')
        self.std_var = tk.StringVar(value='200')
        self.p_var = tk.StringVar(value='30')
        self.c_var = tk.StringVar(value='20')
        self.v_var = tk.StringVar(value='13')
        self.days_var = tk.StringVar(value='365')
        self.q_var = tk.StringVar(value='')

        entries = [
            ('Средний спрос (mean):', self.mean_var),
            ('Стандартное отклонение (std):', self.std_var),
            ('Цена продажи (p):', self.p_var),
            ('Себестоимость (c):', self.c_var),
            ('Цена уценки (v):', self.v_var),
            ('Число дней (trials):', self.days_var),
        ]

        for i, (label, var) in enumerate(entries):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky='w', pady=3)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=i, column=1, pady=3)

        ttk.Label(frm, text='Q (опционально)').grid(row=len(entries), column=0, sticky='w', pady=6)
        ttk.Entry(frm, textvariable=self.q_var, width=12).grid(row=len(entries), column=1, pady=6)

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=len(entries)+1, column=0, columnspan=2, pady=(6, 0))

        ttk.Button(btn_frame, text='Вычислить Q* и смоделировать', command=self.compute_and_simulate).grid(row=0, column=0, padx=6)
        ttk.Button(btn_frame, text='Симуляция с Q', command=self.simulate_with_q).grid(row=0, column=1, padx=6)
        ttk.Button(btn_frame, text='Показать таблицу', command=self.show_table).grid(row=0, column=2, padx=6)
        ttk.Button(btn_frame, text='Экспорт в CSV', command=self.export_csv).grid(row=0, column=3, padx=6)
        ttk.Button(btn_frame, text='Показать графики', command=self.show_plots).grid(row=0, column=4, padx=6)

        # Output
        self.out = tk.Text(self, width=88, height=10, state='disabled', wrap='word')
        self.out.grid(row=1, column=0, padx=12, pady=(6,12))

        # storage for last simulation
        self.last_rows = []
        self.last_summary = None

    def _write_out(self, text):
        self.out['state'] = 'normal'
        self.out.delete('1.0', 'end')
        self.out.insert('1.0', text)
        self.out['state'] = 'disabled'

    def compute_and_simulate(self):
        try:
            mean = float(self.mean_var.get())
            std = float(self.std_var.get())
            p = float(self.p_var.get())
            c = float(self.c_var.get())
            v = float(self.v_var.get())
            days = int(self.days_var.get())

            alpha = critical_fractile(p, c, v)
            z = inverse_normal_cdf(alpha)
            q_opt = int(round(mean + z * std))

            q_to_use = q_opt if self.q_var.get().strip() == '' else int(self.q_var.get().strip())

            avg_profit, avg_excess_loss, avg_shortage_loss, rows = simulate_with_details(mean, std, p, c, v, q_to_use, days)

            self.last_rows = rows
            self.last_summary = (alpha, z, q_opt, q_to_use, days)

            out = []
            out.append(f'Alpha (критический коэффициент) = {alpha:.6f}')
            out.append(f'z(alpha) = {z:.6f}')
            out.append(f'Оптимальный запас Q* = {q_opt} упаковок')
            out.append('')
            out.append(f'Используемый запас Q = {q_to_use}')
            out.append(f'Средняя дневная прибыль = {avg_profit:,.2f} руб.')
            out.append(f'Средний убыток на избытке (в день) = {avg_excess_loss:,.2f} руб.')
            out.append(f'Средний убыток на недостатке (в день) = {avg_shortage_loss:,.2f} руб.')
            out.append('\n(Примечание: прибыль учитывает себестоимость и доход от уценки.)')

            self._write_out('\n'.join(out))

        except Exception as e:
            messagebox.showerror('Ошибка', str(e))

    def simulate_with_q(self):
        try:
            mean = float(self.mean_var.get())
            std = float(self.std_var.get())
            p = float(self.p_var.get())
            c = float(self.c_var.get())
            v = float(self.v_var.get())
            days = int(self.days_var.get())
            if self.q_var.get().strip() == '':
                messagebox.showinfo('Инфо', 'Введите Q в поле "Q (опционально)" или нажмите "Вычислить Q* и смоделировать"')
                return
            q = int(self.q_var.get().strip())

            avg_profit, avg_excess_loss, avg_shortage_loss, rows = simulate_with_details(mean, std, p, c, v, q, days)

            self.last_rows = rows
            self.last_summary = (None, None, None, q, days)

            out = []
            out.append(f'Используемый запас Q = {q}')
            out.append(f'Средняя дневная прибыль = {avg_profit:,.2f} руб.')
            out.append(f'Средний убыток на избытке (в день) = {avg_excess_loss:,.2f} руб.')
            out.append(f'Средний убыток на недостатке (в день) = {avg_shortage_loss:,.2f} руб.')
            out.append('\n(Примечание: прибыль учитывает себестоимость и доход от уценки.)')

            self._write_out('\n'.join(out))

        except Exception as e:
            messagebox.showerror('Ошибка', str(e))

    def show_table(self):
        if not self.last_rows:
            messagebox.showinfo('Инфо', 'Сначала выполните симуляцию (кнопкой "Вычислить Q* и смоделировать" или "Симуляция с Q").')
            return

        win = tk.Toplevel(self)
        win.title('Таблица результатов — детальные дни')

        cols = ('day', 'demand', 'sold', 'leftover', 'unmet', 'revenue', 'cost', 'profit', 'excess_loss', 'shortage_loss')
        tree = ttk.Treeview(win, columns=cols, show='headings', height=20)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor='center', width=90)

        vsb = ttk.Scrollbar(win, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        tree.pack(side='left', fill='both', expand=True)

        for r in self.last_rows:
            tree.insert('', 'end', values=(
                r['day'], r['demand'], r['sold'], r['leftover'], r['unmet'],
                f"{r['revenue']:.2f}", f"{r['cost']:.2f}", f"{r['profit']:.2f}",
                f"{r['excess_loss']:.2f}", f"{r['shortage_loss']:.2f}"
            ))

    def export_csv(self):
        if not self.last_rows:
            messagebox.showinfo('Инфо', 'Сначала выполните симуляцию для получения данных.');
            return

        filename = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv')])
        if not filename:
            return

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['day','demand','sold','leftover','unmet','revenue','cost','profit','excess_loss','shortage_loss']
                writer.writerow(header)
                for r in self.last_rows:
                    writer.writerow([
                        r['day'], r['demand'], r['sold'], r['leftover'], r['unmet'],
                        f"{r['revenue']:.2f}", f"{r['cost']:.2f}", f"{r['profit']:.2f}",
                        f"{r['excess_loss']:.2f}", f"{r['shortage_loss']:.2f}"
                    ])
            messagebox.showinfo('Готово', f'Данные экспортированы в {filename}')
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))

    def show_plots(self):
        if not self.last_rows:
            messagebox.showinfo('Инфо', 'Сначала выполните симуляцию (кнопкой "Вычислить Q* и смоделировать" или "Симуляция с Q").')
            return
        if not HAS_MPL:
            messagebox.showerror('Ошибка', 'matplotlib не найден. Установите его: pip install matplotlib')
            return

        # data
        days = [r['day'] for r in self.last_rows]
        demand = [r['demand'] for r in self.last_rows]
        profit = [r['profit'] for r in self.last_rows]
        leftover = [r['leftover'] for r in self.last_rows]
        unmet = [r['unmet'] for r in self.last_rows]
        q_used = None
        if self.last_summary:
            q_used = self.last_summary[3]

        # Plot 1: demand over days with Q line
        win1 = tk.Toplevel(self)
        win1.title('График: спрос по дням (и линия Q)')
        fig1 = plt.Figure(figsize=(8, 3.5))
        ax1 = fig1.add_subplot(111)
        ax1.plot(days, demand)
        if q_used is not None:
            ax1.axhline(y=q_used, linestyle='--')
            ax1.text(0.98, 0.95, f'Q = {q_used}', transform=ax1.transAxes, ha='right', va='top')
        ax1.set_xlabel('День')
        ax1.set_ylabel('Спрос')
        ax1.set_title('Спрос по дням')

        canvas1 = FigureCanvasTkAgg(fig1, master=win1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)

        # Plot 2: histogram of daily profit
        win2 = tk.Toplevel(self)
        win2.title('Гистограмма: дневная прибыль')
        fig2 = plt.Figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        ax2.hist(profit, bins=30)
        ax2.set_xlabel('Прибыль (руб.)')
        ax2.set_ylabel('Частота')
        ax2.set_title('Распределение дневной прибыли')

        canvas2 = FigureCanvasTkAgg(fig2, master=win2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)


if __name__ == '__main__':
    app = App()
    app.mainloop()
