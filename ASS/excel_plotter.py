# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 18:25:02 2025

@author: marti
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ExcelPlotWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Excel Plotter")
        self.window.geometry("1000x600")

        self.df = None
        self._create_widgets()

    # def _create_widgets(self):
    #     # Left panel
    #     self.left_panel = tk.Frame(self.window, width=300, padx=10, pady=10)
    #     self.left_panel.pack(side=tk.LEFT, fill=tk.Y)

    #     # Load button
    #     self.load_button = tk.Button(self.left_panel, text="Load Excel File", command=self._load_excel)
    #     self.load_button.pack(pady=(0, 10))

    #     # X selection
    #     self.x_label = tk.Label(self.left_panel, text="X Column:")
    #     self.x_label.pack()
    #     self.x_combo = ttk.Combobox(self.left_panel, state='disabled')
    #     self.x_combo.pack(fill=tk.X, pady=(0, 10))

    #     self.x_axis_label = tk.Entry(self.left_panel)
    #     self.x_axis_label.insert(0, "")  # empty by default
    #     self.x_axis_label.pack(fill=tk.X, pady=(0, 10))
    #     self.x_axis_label.config(state='disabled')

    #     # Y selection
    #     self.y_label = tk.Label(self.left_panel, text="Y Column:")
    #     self.y_label.pack()
    #     self.y_combo = ttk.Combobox(self.left_panel, state='disabled')
    #     self.y_combo.pack(fill=tk.X, pady=(0, 10))

    #     self.y_axis_label = tk.Entry(self.left_panel)
    #     self.y_axis_label.insert(0, "")
    #     self.y_axis_label.pack(fill=tk.X, pady=(0, 10))
    #     self.y_axis_label.config(state='disabled')

    #     # Plot button
    #     self.plot_button = tk.Button(self.left_panel, text="Plot", state='disabled', command=self._plot_data)
    #     self.plot_button.pack(pady=(10, 10))

    #     # Save button
    #     self.save_button = tk.Button(self.left_panel, text="Save as PNG", state='disabled', command=self._save_plot)
    #     self.save_button.pack(pady=(0, 10))

    #     # Right panel for canvas
    #     self.right_panel = tk.Frame(self.window)
    #     self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    #     self.fig, self.ax = plt.subplots(figsize=(5, 4))
    #     self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
    #     self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_widgets(self):
        # Left panel (ttk.Frame)
        self.left_panel = ttk.Frame(self.window, padding=10)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
    
        # Load button
        self.load_button = ttk.Button(self.left_panel, text="Load Excel File", command=self._load_excel)
        self.load_button.pack(fill=tk.X, pady=(0, 10))
    
        # X column
        ttk.Label(self.left_panel, text="X Column:").pack(anchor='w')
        self.x_combo = ttk.Combobox(self.left_panel, state='disabled')
        self.x_combo.pack(fill=tk.X, pady=(0, 5))
    
        self.x_axis_label = ttk.Entry(self.left_panel, state='disabled')
        self.x_axis_label.pack(fill=tk.X, pady=(0, 10))
    
        # Y column
        ttk.Label(self.left_panel, text="Y Column:").pack(anchor='w')
        self.y_combo = ttk.Combobox(self.left_panel, state='disabled')
        self.y_combo.pack(fill=tk.X, pady=(0, 5))
    
        self.y_axis_label = ttk.Entry(self.left_panel, state='disabled')
        self.y_axis_label.pack(fill=tk.X, pady=(0, 10))
    
        # Plot button
        self.plot_button = ttk.Button(self.left_panel, text="Plot", state='disabled', command=self._plot_data)
        self.plot_button.pack(fill=tk.X, pady=(10, 10))
    
        # Save button
        self.save_button = ttk.Button(self.left_panel, text="Save as PNG", state='disabled', command=self._save_plot)
        self.save_button.pack(fill=tk.X)
    
        # Right panel for canvas
        self.right_panel = ttk.Frame(self.window)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if not path:
            return

        try:
            self.df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load Excel file:\n{e}")
            return

        # Populate dropdowns
        columns = list(self.df.columns)
        self.x_combo.config(values=columns, state='readonly')
        self.y_combo.config(values=columns, state='readonly')
        self.x_combo.set('')
        self.y_combo.set('')
        self.x_axis_label.config(state='normal')
        self.y_axis_label.config(state='normal')
        self.plot_button.config(state='normal')
        self.save_button.config(state='normal')

    def _plot_data(self):
        x_col = self.x_combo.get()
        y_col = self.y_combo.get()
        if not x_col or not y_col:
            messagebox.showwarning("Missing Selection", "Please select both X and Y columns.")
            return

        try:
            x = self.df[x_col]
            y = self.df[y_col]
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not extract data:\n{e}")
            return

        self.ax.clear()
        self.ax.scatter(x, y)

        # Axis labels
        x_label = self.x_axis_label.get().strip()
        y_label = self.y_axis_label.get().strip()
        self.ax.set_xlabel(x_label if x_label else x_col)
        self.ax.set_ylabel(y_label if y_label else y_col)

        # self.ax.set_title("Scatter Plot")
        self.canvas.draw()

    def _save_plot(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if path:
            try:
                self.fig.savefig(path, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image:\n{e}")
                
    def test():
        pass
    
