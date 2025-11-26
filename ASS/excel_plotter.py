# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 18:25:02 2025

@author: marti
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import io
import win32clipboard

from ASS.logic import Plotting

class ExcelPlotWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Excel Plotter")
        self.window.geometry("1000x600")

        self.df = None
        self._create_widgets()
        self._bind_shortcuts()  
    
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
        
        ttk.Label(self.left_panel, text="Fit function:").pack(anchor='w')
        self.fit_type_combo = ttk.Combobox(
            self.left_panel,
            values=["None", "Linear", "Exponential", "Quadratic", "log", "log10", "Power"],
            state='disabled'
        )
        # default
        self.fit_type_combo.set("None")
        self.fit_type_combo.pack(fill=tk.X, pady=(0, 10))
    
        # Plot button
        self.fit_button = ttk.Button(self.left_panel, text="Fit", state='disabled', command=self._fit_data)
        self.fit_button.pack(fill=tk.X, pady=(10, 10))
    
        # Save button
        self.save_button = ttk.Button(self.left_panel, text="Save as PNG", state='disabled', command=self._save_plot)
        self.save_button.pack(fill=tk.X)
    
        # Right panel for canvas
        self.right_panel = ttk.Frame(self.window)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        # self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # add right-click popup
        widget.bind("<Button-3>", self._on_canvas_right_click)
        
        self.canvas.draw()
        
    def _bind_shortcuts(self):
        self.window.bind("<Control-c>", lambda event: Plotting.copy_figure_to_clipboard(self.fig))
    
    def _on_canvas_right_click(self, event):
        menu = tk.Menu(self.right_panel, tearoff=0)
        menu.add_command(label="Update plot", command=self._plot_data)
        menu.add_command(label="Save plot", command=self._save_plot)
        menu.add_command(label="Copy to Clipboard", command=lambda: Plotting.copy_figure_to_clipboard(self.fig))
        menu.post(event.x_root, event.y_root)

    def _load_excel(self):
        path = filedialog.askopenfilename(parent = self.window, filetypes=[("Excel files", "*.xlsx *.xls")])
        if not path:
            return
        try:
            self.df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load Excel file:\n{e}")
            return

        # Populate dropdowns
        columns = ["None"] + list(self.df.columns)
        for combo in [self.x_combo, self.y_combo, self.c_combo]:
            combo.config(values=columns, state='readonly')
            combo.set("None")
        # self.x_combo.set('')
        # self.y_combo.set('')
        self.x_axis_label.config(state='normal')
        self.y_axis_label.config(state='normal')
        self.c_axis_label.config(state='normal')
        self.title.config(state='normal')
        self.plot_style_combo.config(state='readonly')
        self.plot_button.config(state='normal')
        self.fit_type_combo.config(state='readonly')
        self.fit_button.config(state='normal')
        self.save_button.config(state='normal')
    
    def _plot_data(self):
        x_col = self.x_combo.get()
        y_col = self.y_combo.get()
        c_col = self.c_combo.get()
        cmap = self.cmap_combo.get()
        style = self.plot_style_combo.get()
    
        if not x_col or not y_col:
            messagebox.showwarning("Missing Selection", "Please select both X and Y columns.", parent = self.window)
            return
    
        try:
            x = self.df[x_col]
            y = self.df[y_col]
            c = None if c_col == "None" else self.df[c_col]
            # c = self.df[c_col] if c_col else None
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not extract data:\n{e}", parent = self.window)
            return
    
        # --- Destroy and recreate the figure + canvas ---
        for widget in self.right_panel.winfo_children():
            widget.destroy()
            
        # Close the previous figure if it exists
        try:
            plt.close(self.fig)
        except AttributeError:
            pass
    
        # self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        # self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        widget.bind("<Button-3>", self._on_canvas_right_click) 
        
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
                
    # def copy_figure_to_clipboard(self):
    #     """
    #     Copy a Matplotlib figure to the Windows clipboard as an image (CF_DIB format).
    
    #     Parameters
    #     ----------
    #     fig : matplotlib.figure.Figure
    #         The Matplotlib figure to copy.
    #     """
    #     if self.fig is None:
    #         print("⚠️ No figure available to copy.")
    #         messagebox.showinfo("Clipboard", "⚠️ No figure available to copy.")
    #         return
        
    #     try:
    #         # --- Save the figure to an in-memory PNG buffer ---
    #         buf = io.BytesIO()
    #         self.fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    #         buf.seek(0)
    
    #         # --- Load with Pillow ---
    #         img = Image.open(buf)
    
    #         # --- Convert to DIB (Device Independent Bitmap) format ---
    #         output = io.BytesIO()
    #         img.convert("RGB").save(output, "BMP")
    #         data = output.getvalue()[14:]  # remove BMP header
    #         output.close()
    
    #         # --- Send to Windows clipboard ---
    #         win32clipboard.OpenClipboard()
    #         win32clipboard.EmptyClipboard()
    #         win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    #         win32clipboard.CloseClipboard()
    
    #         # print("✅ Figure copied to clipboard as image.")
    #         # messagebox.showinfo("Clipboard", "✅ Figure copied to clipboard as image.")
    #     except Exception as e:
    #         print(f"⚠️ Failed to copy figure to clipboard: {e}")
    #         messagebox.showinfo("Clipboard", f"⚠️ Failed to copy figure to clipboard: {e}")
            
        
    def _fit_data(self):
        x_col = self.x_combo.get()
        y_col = self.y_combo.get()
        c_col = self.c_combo.get()
        cmap = self.cmap_combo.get()
        fit_type = self.fit_type_combo.get()
    
        if not x_col or not y_col:
            messagebox.showwarning("Missing Selection", "Please select both X and Y columns.", parent=self.window)
            return
        if fit_type == "None":
            messagebox.showinfo("Fit Info", "Please select a fitting function.", parent=self.window)
            return
    
        # --- Extract data ---
        try:
            x = np.array(self.df[x_col], dtype=float)
            y = np.array(self.df[y_col], dtype=float)
            c = None if c_col == "None" else self.df[c_col]
        except Exception as e:
            messagebox.showerror("Fit Error", f"Could not extract data:\n{e}", parent=self.window)
            return
    
        # --- Define models ---
        # def linear(x, a, b): return a * x + b
        # def exponential(x, a, b): return a * np.exp(b * x)
        # def quadratic(x, a, b, c): return a * x**2 + b * x + c
        # def log(x, a, b): return a * np.log(x) + b
        # def log10(x, a, b): return a * np.log10(x) + b
        # def power(x, a, b): return a * np.power(x, b)
        
        # models = {
        #     "Linear": (linear, ["a", "b"]),
        #     "Exponential": (exponential, ["a", "b"]),
        #     "Quadratic": (quadratic, ["a", "b", "c"]),
        #     "log": (log, ["a", "b"]),
        #     "log10": (log10, ["a", "b"]),
        #     "Power": (power, ["a", "b"]),
        # }
        
        def linear(x, a, b): return a * x + b
        def exponential(x, a, b, c): return a * np.exp(b * x) + c
        def quadratic(x, a, b, c): return a * x**2 + b * x + c
        def log(x, a, b, c): return a * np.log(b*x) + c
        def log10(x, a, b, c): return a * np.log10(b*x) + c
        def power(x, a, b, c): return a * np.power(x, b) + c
    
    
        models = {
            "Linear": (linear, ["a", "b"]),
            "Exponential": (exponential, ["a", "b", "c"]),
            "Quadratic": (quadratic, ["a", "b", "c"]),
            "log": (log, ["a", "b", "c"]),
            "log10": (log10, ["a", "b", "c"]),
            "Power": (power, ["a", "b", "c"]),
        }
    
        func, param_names = models.get(fit_type, (None, []))
        if func is None:
            messagebox.showerror("Fit Error", f"Unsupported fit type: {fit_type}", parent=self.window)
            return
    
        # --- Clean invalid points ---
        mask = np.isfinite(x) & np.isfinite(y)
        if fit_type in ["log", "log10", "Power"]:
            mask &= (x > 0)
        x_fit = x[mask]
        y_fit = y[mask]
    
        if len(x_fit) < len(param_names) + 1:
            messagebox.showerror("Fit Error", "Not enough valid data points.", parent=self.window)
            return
    
        # --- Perform the fit ---
        try:
            popt, _ = curve_fit(func, x_fit, y_fit, maxfev=5000)
            y_pred = func(x_fit, *popt)
    
            # R²
            ss_res = np.sum((y_fit - y_pred)**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
            # Dense x for smooth fit
            x_dense = np.linspace(x_fit.min(), x_fit.max(), 300)
            y_dense = func(x_dense, *popt)
    
            # Equation string
            # coeffs = [f"{n}={v:.3g}" for n, v in zip(param_names, popt)]
            # eq_str = f"{fit_type}: " + ", ".join(coeffs) + f"  (R²={r2:.3f})"
            
            # # Format coefficients
            # a, *rest = popt
            # if fit_type == "Linear":
            #     eq_str = f"{a:.3g}·x + {rest[0]:.3g}  (R²={r2:.3f})"
            # elif fit_type == "Exponential":
            #     eq_str = f"{a:.3g}·exp({rest[0]:.3g}·x)  (R²={r2:.3f})"
            # elif fit_type == "Quadratic":
            #     eq_str = f"{a:.3g}·x² + {rest[0]:.3g}·x + {rest[1]:.3g}  (R²={r2:.3f})"
            # elif fit_type == "log":
            #     eq_str = f"{a:.3g}·ln(x) + {rest[0]:.3g}  (R²={r2:.3f})"
            # elif fit_type == "log10":
            #     eq_str = f"{a:.3g}·log₁₀(x) + {rest[0]:.3g}  (R²={r2:.3f})"
            # elif fit_type == "Power":
            #     eq_str = f"{a:.3g}·x^{rest[0]:.3g}  (R²={r2:.3f})"
            # else:
            #     eq_str = f"{fit_type}: " + ", ".join(f"{v:.3g}" for v in popt) + f"  (R²={r2:.3f})"
            
            # Format coefficients
            a, *rest = popt
            if fit_type == "Linear":
                eq_str = f"{a:.3g}·x + {rest[0]:.3g}  (R²={r2:.3f})"
            elif fit_type == "Exponential":
                eq_str = f"{a:.3g}·exp({rest[0]:.3g}·x)+{rest[1]:.3g}  (R²={r2:.3f})"
            elif fit_type == "Quadratic":
                eq_str = f"{a:.3g}·x² + {rest[0]:.3g}·x + {rest[1]:.3g}  (R²={r2:.3f})"
            elif fit_type == "log":
                eq_str = f"{a:.3g}·log({rest[0]:.3g}·x) + {rest[1]:.3g}  (R²={r2:.3f})"
            elif fit_type == "log10":
                eq_str = f"{a:.3g}·log₁₀({rest[0]:.3g}·x) + {rest[1]:.3g}  (R²={r2:.3f})"
            elif fit_type == "Power":
                eq_str = f"{a:.3g}·x^{rest[0]:.3g} + {rest[1]:.3g}  (R²={r2:.3f})"
            else:
                eq_str = f"{fit_type}: " + ", ".join(f"{v:.3g}" for v in popt) + f"  (R²={r2:.3f})"
    
        except Exception as e:
            messagebox.showerror("Fit Error", f"Curve fitting failed:\n{e}", parent=self.window)
            return
    
        # --- Recreate figure ---
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        try:
            plt.close(self.fig)
        except AttributeError:
            pass
    
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        widget.bind("<Button-3>", self._on_canvas_right_click)
    
        # --- Plot data and fit ---
        # self.ax.scatter(x_fit, y_fit, color="C0", label="Data")
        if c_col and c is not None:
            sc = self.ax.scatter(x, y, c=c, cmap=cmap)
            cbar = self.fig.colorbar(sc, ax=self.ax)
            c_label = self.c_axis_label.get().strip()
            cbar.set_label(c_label if c_label else c_col)
        else:
            self.ax.scatter(x_fit, y_fit, color="C0")
        self.ax.plot(x_dense, y_dense, "r--", label=eq_str)
    
        # Axis labels and title
        x_label = self.x_axis_label.get().strip()
        y_label = self.y_axis_label.get().strip()
        title = self.title.get().strip()
        self.ax.set_xlabel(x_label if x_label else x_col)
        self.ax.set_ylabel(y_label if y_label else y_col)
        self.ax.set_title(title if title else f"{fit_type} Fit")
    
        self.ax.legend(fontsize=8)
        self.canvas.draw()