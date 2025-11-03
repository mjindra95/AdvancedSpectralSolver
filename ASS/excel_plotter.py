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
        
        # --- Add after Y controls ---
        ttk.Label(self.left_panel, text="Color Variable (optional):").pack(anchor='w')
        self.c_combo = ttk.Combobox(self.left_panel, state='disabled')
        self.c_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Color axis label (optional)
        self.c_axis_label = ttk.Entry(self.left_panel, state='disabled')
        self.c_axis_label.pack(fill=tk.X, pady=(0, 10))

        
        ttk.Label(self.left_panel, text="Colormap:").pack(anchor='w')
        self.cmap_combo = ttk.Combobox(
            self.left_panel,
            values=sorted(plt.colormaps()),
            state='readonly'
        )
        self.cmap_combo.set("viridis")
        self.cmap_combo.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.left_panel, text="Plot title:").pack(anchor='w')
        self.title = ttk.Entry(self.left_panel, state='disabled')
        self.title.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.left_panel, text="Plot style:").pack(anchor='w')
        self.plot_style_combo = ttk.Combobox(
            self.left_panel,
            values=["Scatter", "Line", "Line + Markers"],
            state='disabled'
        )
        # default
        self.plot_style_combo.set("Scatter")
        self.plot_style_combo.pack(fill=tk.X, pady=(0, 10))
    
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
        for combo in [self.x_combo, self.y_combo, self.c_combo]:
            combo.config(values=columns, state='readonly')
            combo.set('')
        # self.x_combo.set('')
        # self.y_combo.set('')
        self.x_axis_label.config(state='normal')
        self.y_axis_label.config(state='normal')
        self.c_axis_label.config(state='normal')
        self.title.config(state='normal')
        self.plot_style_combo.config(state='readonly')
        self.plot_button.config(state='normal')
        self.save_button.config(state='normal')
    
    def _plot_data(self):
        x_col = self.x_combo.get()
        y_col = self.y_combo.get()
        c_col = self.c_combo.get()
        cmap = self.cmap_combo.get()
        style = self.plot_style_combo.get()
    
        if not x_col or not y_col:
            messagebox.showwarning("Missing Selection", "Please select both X and Y columns.")
            return
    
        try:
            x = self.df[x_col]
            y = self.df[y_col]
            c = self.df[c_col] if c_col else None
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not extract data:\n{e}")
            return
    
        # --- Destroy and recreate the figure + canvas ---
        for widget in self.right_panel.winfo_children():
            widget.destroy()
            
        # Close the previous figure if it exists
        try:
            plt.close(self.fig)
        except AttributeError:
            pass
    
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # --- Plotting logic ---
        # if c_col and c is not None:
        #     sc = self.ax.scatter(x, y, c=c, cmap=cmap)
        #     cbar = self.fig.colorbar(sc, ax=self.ax)
        #     c_label = self.c_axis_label.get().strip()
        #     cbar.set_label(c_label if c_label else c_col)
        # else:
        #     self.ax.scatter(x, y, color='C0')
        
        if c_col and c is not None:
            # We will always use scatter + colorbar here
            if style == "Scatter":
                sc = self.ax.scatter(x, y, c=c, cmap=cmap)
                cbar = self.fig.colorbar(sc, ax=self.ax)
                c_label = self.c_axis_label.get().strip()
                cbar.set_label(c_label if c_label else c_col)
            elif style == "Line":
                self.ax.plot(x, y, linestyle='-', marker=None, color='C0')
            elif style == "Line + Markers":
                self.ax.plot(x, y, linestyle='-', marker=None, color='grey')
                sc = self.ax.scatter(x, y, c=c, cmap=cmap)
                cbar = self.fig.colorbar(sc, ax=self.ax)
                c_label = self.c_axis_label.get().strip()
                cbar.set_label(c_label if c_label else c_col)
        
            # Gentle heads-up if user asked for line-based style
            # if style in ["Line", "Line + Markers"]:
            #     messagebox.showinfo(
            #         "Plot Style Notice",
            #         "Color-mapped data is shown as a scatter plot. "
            #         "Line styles with per-point colormap are not supported."
            #     )
        
        else:
            # No color variable -> honor requested style
            if style == "Scatter":
                self.ax.scatter(x, y, color='C0')
        
            elif style == "Line":
                # line only
                self.ax.plot(x, y, linestyle='-', marker=None, color='C0')
        
            elif style == "Line + Markers":
                # line + markers
                self.ax.plot(x, y, linestyle='-', marker='o', color='C0')
        
            else:
                # Fallback safety (shouldn't happen because combo is restricted)
                self.ax.scatter(x, y, color='C0')
            
        # Axis labels
        x_label = self.x_axis_label.get().strip()
        y_label = self.y_axis_label.get().strip()
        title = self.title.get().strip()
        self.ax.set_xlabel(x_label if x_label else x_col)
        self.ax.set_ylabel(y_label if y_label else y_col)
        self.ax.set_title(title if title else None)
    
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