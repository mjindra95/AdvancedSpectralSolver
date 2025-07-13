# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 23:28:24 2025

@author: marti
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector, SpanSelector
import numpy as np
import pandas as pd
import time

from ASS.file_utils import File_utils
from ASS.logic import Loading, Plotting, Processing

class Map_1D(tk.Toplevel):
    """
    A top‐level window for loading data and generating a heatmap or plot.
    Layout:
      - Left column: various input controls (dropdowns, entries, buttons, slider)
      - Right column: a Canvas (or Matplotlib figure) for displaying results
    """
    # def __init__(self, master):
    def __init__(self, main_win, plot_callback):
        #super().__init__(master)
        super().__init__(main_win.root)
        self.main = main_win
        self.title("1D Map Analysis")
        self.geometry("1000x800")  # adjust as needed
        self.resizable(True, True)
        self.plot_callback = plot_callback
        
        # Make column 1 (the right panel) expandable
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Variables for all controls
        self.source_var   = tk.StringVar(value="Horiba")
        self.mode_var     = tk.StringVar(value="SEC")
        self.lower_shift_var    = tk.StringVar()
        self.upper_shift_var    = tk.StringVar()
        self.lower_idx_var    = tk.StringVar()
        self.upper_idx_var    = tk.StringVar()
        self.cmap_var     = tk.StringVar(value="viridis")
        self.corr_var     = tk.StringVar(value="None")
        self.orient_var   = tk.StringVar(value="ascending")
        self.interpolation_var   = tk.StringVar(value="none")
        self.slider_var   = tk.IntVar(value=20)
        self.plot_type = None
        self.overlay_data = None
        self.heatmap_data = None
        self.directory_path = None

        # Build the UI
        self._create_widgets()

    def _create_widgets(self):
        # --- LAYOUT CONFIGURATION ---
        self.columnconfigure(0, minsize=250)  # left panel fixed width
        self.columnconfigure(1, weight=1)     # right panel grows
        self.rowconfigure(0, weight=1)
        
        # --- LEFT PANEL: controls ---
        left_frame = ttk.Frame(self)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_frame.grid_propagate(False)
        left_frame.configure(width=250)
        left_frame.columnconfigure(0, weight=1)

        # 1) Source dropdown
        ttk.Label(left_frame, text="Source:").grid(row=0, column=0, sticky="w", pady=(0,2))
        source_menu = ttk.Combobox(
            left_frame,
            textvariable=self.source_var,
            values=["Witec", "Horiba"],
            state="readonly"
        )
        source_menu.grid(row=1, column=0, sticky="ew", pady=(0,10))

        # 2) Mode dropdown
        ttk.Label(left_frame, text="Mode:").grid(row=2, column=0, sticky="w", pady=(0,2))
        mode_menu = ttk.Combobox(
            left_frame,
            textvariable=self.mode_var,
            values=["Linescan", "Timescan", "SEC"],
            state="readonly"
        )
        mode_menu.grid(row=3, column=0, sticky="ew", pady=(0,10))

        # 3) Load button
        load_btn = ttk.Button(left_frame, text="Load", command=self._on_load)
        load_btn.grid(row=4, column=0, sticky="ew", pady=(0,15))
        
        # 1) make a little container for both fields
        limit_frame = ttk.Frame(left_frame)
        limit_frame.grid(row=5, column=0, sticky="ew", pady=(0,15))
        limit_frame.columnconfigure(0, weight=1)
        limit_frame.columnconfigure(1, weight=1)
        
        # 2) pack Lower Limit in column 0
        ttk.Label(limit_frame, text="Left cutoff:").grid(row=0, column=0, sticky="w")
        lower_shift = ttk.Entry(limit_frame, textvariable=self.lower_shift_var, width=10)
        lower_shift.grid(row=1, column=0, sticky="ew", padx=(0,5))
        
        # 3) pack Upper Limit in column 1
        ttk.Label(limit_frame, text="Right cutoff:").grid(row=0, column=1, sticky="w")
        upper_shift = ttk.Entry(limit_frame, textvariable=self.upper_shift_var, width=10)
        upper_shift.grid(row=1, column=1, sticky="ew", padx=(5,0))
        
        # 2) pack Lower Limit in column 0
        ttk.Label(limit_frame, text="Lower index:").grid(row=2, column=0, sticky="w")
        lower_index = ttk.Entry(limit_frame, textvariable=self.lower_idx_var, width=10)
        lower_index.grid(row=3, column=0, sticky="ew", padx=(0,5))
        
        # 3) pack Upper Limit in column 1
        ttk.Label(limit_frame, text="Upper index:").grid(row=2, column=1, sticky="w")
        upper_index = ttk.Entry(limit_frame, textvariable=self.upper_idx_var, width=10)
        upper_index.grid(row=3, column=1, sticky="ew", padx=(5,0))

        # 5) Colormap dropdown
        ttk.Label(left_frame, text="Colorscale:").grid(row=9, column=0, sticky="w", pady=(0,2))
        cmap_menu = ttk.Combobox(
            left_frame,
            textvariable=self.cmap_var,
            values=["viridis", "cividis", "plasma", "brg", "nipy_spectral", "inferno", "YlOrBr", "YlOrBr_r"],
            state="readonly"
        )
        cmap_menu.grid(row=10, column=0, sticky="ew", pady=(0,15))

        # 6) Correction type dropdown
        ttk.Label(left_frame, text="Correction:").grid(row=11, column=0, sticky="w", pady=(0,2))
        corr_menu = ttk.Combobox(
            left_frame,
            textvariable=self.corr_var,
            values=["Linear", "Zero", "None"],
            state="readonly"
        )
        corr_menu.grid(row=12, column=0, sticky="ew", pady=(0,15))

        # 7) Orientation dropdown
        ttk.Label(left_frame, text="Orientation:").grid(row=13, column=0, sticky="w", pady=(0,2))
        orient_menu = ttk.Combobox(
            left_frame,
            textvariable=self.orient_var,
            values=["descending", "ascending"],
            state="readonly"
        )
        orient_menu.grid(row=14, column=0, sticky="ew", pady=(0,15))
        
        ttk.Label(left_frame, text="Interpolation:").grid(row=15, column=0, sticky="w", pady=(0,2))
        interpolation_menu = ttk.Combobox(
            left_frame,
            textvariable=self.interpolation_var,
            values=['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                    'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                    'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'],
            state="readonly"
        )
        interpolation_menu.grid(row=16, column=0, sticky="ew", pady=(0,15))

        # 8) Heatmap button
        heatmap_btn = ttk.Button(left_frame, text="Heatmap", command=self._on_heatmap)
        heatmap_btn.grid(row=17, column=0, sticky="ew", pady=(0,15))

        # 9) Slider (0–100, default 20)
        ttk.Label(left_frame, text="Threshold:").grid(row=18, column=0, sticky="w", pady=(0,2))
        slider = ttk.Scale(
            left_frame,
            from_=0, to=100,
            variable=self.slider_var,
            orient=tk.HORIZONTAL
        )
        slider.grid(row=19, column=0, sticky="ew", pady=(0,5))
        # Show current slider value
        self._slider_label = ttk.Label(left_frame, text="20")
        self._slider_label.grid(row=20, column=0, sticky="e", pady=(0,15))
        self.slider_var.trace_add("write", self._on_slider_change)

        # 10) Plot button
        plot_btn = ttk.Button(left_frame, text="Plot", command=self._on_plot)
        plot_btn.grid(row=21, column=0, sticky="ew", pady=(0,10))
        
        # 11) Plot button
        fit_btn = ttk.Button(left_frame, text="Fit", command=self._on_fit)
        fit_btn.grid(row=22, column=0, sticky="ew", pady=(0,10))
    
        
        save_btn = ttk.Button(left_frame, text="Save", command=self._on_save)
        save_btn.grid(row=23, column=0, sticky="ew", pady=(0,10))
        
        # Fill extra vertical space
        left_frame.rowconfigure(18, weight=1)
        
        # --- RIGHT PANEL: Matplotlib figure ---
        right_frame = ttk.Frame(self, relief=tk.SUNKEN)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        self.fig = Figure(figsize=(6,4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        # self.canvas.mpl_connect("button_press_event", self._on_map_click)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_map_click)

    # --- Callback stubs below; user fills in actual logic ---

    def _on_load(self):
        """
        Called when “Load” button is pressed.
        Choose loader based on self.source_var.get().
        """
        src = self.source_var.get()
        #mod = self.mode_var.get()
        if src == "Horiba":
            self.horiba_load()
        elif src == "Witec":
            self.witec_load()
        else:
            messagebox.showerror("Load Error", f"Unknown source: {src}")
            
    def horiba_load(self):
        mod = self.mode_var.get()
        # directory_path = File_utils.ask_directory()
        self.directory_path = filedialog.askdirectory(parent = self , title = "Choose directory...")
        if not self.directory_path:
            return
        self.df = Loading.load_horiba_1D(self.directory_path, mod)
        self.lower_shift_var.set(self.df.columns.min())
        self.upper_shift_var.set(self.df.columns.max())
        self.lower_idx_var.set(self.df.index.min())
        self.upper_idx_var.set(self.df.index.max())
        messagebox.showinfo("Load", "Data were loaded succesfully", parent = self)
            
    def witec_load(self):
        mod = self.mode_var.get()
        if mod == 'SEC':
            messagebox.showinfo("Load", "Would load Witec SEC data here.")
        elif mod == 'Linescan':
            messagebox.showinfo("Load", "Would load Witec Linescan data here.")
        elif mod == 'Timescan':
            messagebox.showinfo("Load", "Would load Witec Timescan data here.")
        else:
            messagebox.showinfo("Load Error", f"Unknown mode: {mod}")
    
    def _on_heatmap(self):
        df = self.df  # your loaded DataFrame
    
        cmap_name   = self.cmap_var.get()
        correction  = self.corr_var.get()
        orientation = self.orient_var.get()
        interpolation = self.interpolation_var.get()
        
        mode        = self.mode_var.get()
    
        new_ax, Data_df = Plotting.plot_1D_map(
            ax=            self.ax,
            df=            df,
            lower_shift=   float(self.lower_shift_var.get()),
            upper_shift=   float(self.upper_shift_var.get()),
            lower_index=   float(self.lower_idx_var.get()),
            upper_index=   float(self.upper_idx_var.get()),
            cmap_name=     cmap_name,
            correction=    correction,
            orientation=   orientation,
            mode=          mode,
            interpolation= interpolation)
    
        # 5) Redraw the embedded figure
        self.ax = new_ax
        self.heatmap_data = Data_df
        self.canvas.draw()
        
        self.plot_type = "Heatmap"

    def _on_plot(self):
        # 1) read bounds & options
        ls = self.lower_shift_var.get().strip()
        us = self.upper_shift_var.get().strip()
        li = self.lower_idx_var.get().strip()
        ui = self.upper_idx_var.get().strip()
    
        new_ax, Data_df = Plotting.plot_1d_overlay(
            ax=             self.ax,
            df=             self.df,
            lower_shift=    float(ls) if ls else None,
            upper_shift=    float(us) if us else None,
            lower_index=    float(li) if li else None,
            upper_index=    float(ui) if ui else None,
            offset_percent= self.slider_var.get(),
            cmap_name=      self.cmap_var.get(),
            correction=     self.corr_var.get(),
            mode=           self.mode_var.get()
        )
    
        # 2) redraw
        self.ax = new_ax
        self.overlay_data = Data_df
        self.canvas.draw()
        
        self.plot_type = "Lineplot"
   
    def _on_save(self):
        # Ask the user where to save the current figure
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save figure as…",
            defaultextension=".png",
            filetypes=[
                ("PNG image","*.png"),
                ("PDF file","*.pdf"),
                ("SVG vector","*.svg"),
                ("All files","*.*")
            ]
        )
        if not path:
            return  # user cancelled
    
        try:
            # Save the Matplotlib Figure to the chosen file
            # you can adjust dpi or bbox_inches as needed
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Figure saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save figure:\n{e}")

    def _on_slider_change(self, *_):
        # Update the label showing current slider value
        val = self.slider_var.get()
        self._slider_label.config(text=str(val))
        
    def _on_map_click(self, event):
        # print("Click recieved")
        # print(f"Clicked: button={event.button}, x={event.xdata}, y={event.ydata}, ax={event.inaxes}")
        # only care about right clicks *inside* your heatmap axes
        # if event.inaxes is not self.ax:
        #     return
        
        if event.button != 3:
            return
    
        # record which pixel
        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))
        self._last_pixel = (xpix, ypix)
        
        print(self._last_pixel)
    
        # build the menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Plot in Main", command=self.plot_spectrum_to_main)
        menu.add_command(label="Select region", command=self.start_region_selection)
        menu.add_command(label="Reset region", command=self.reset_region_selection)
        menu.add_command(label="Save map", command=self._on_save)
    
        # use the underlying Tk event's root coords
        try:
            xr = event.guiEvent.x_root
            yr = event.guiEvent.y_root
        except Exception:
            # fallback: center of canvas
            w = self.canvas.get_tk_widget()
            xr = w.winfo_rootx() + w.winfo_width() // 2
            yr = w.winfo_rooty() + w.winfo_height() // 2

        menu.tk_popup(xr, yr)
        menu.grab_release()
        
    # def start_region_selection(self):
        
    #     # def safe_onselect(eclick, erelease):
    #     #     # Only allow selection if both click and release were in the main Axes
    #     #     if eclick.inaxes != self.ax or erelease.inaxes != self.ax:
    #     #         print("Selection started or ended outside main axes — ignoring.")
    #     #         return
    #     #     self._on_region_selected(eclick, erelease)
        
    #     # If already exist, just re–activate it
    #     if hasattr(self, '_region_selector') and self._region_selector:
    #         self._region_selector.set_active(True)
    #         return

    #     self._region_selector = RectangleSelector(
    #         self.ax,
    #         onselect=self._on_region_selected,
    #         #drawtype='box',
    #         useblit=True,
    #         button=[1],            # left click only
    #         spancoords='data',     # interpret coords in data space
    #         # rectprops=dict(edgecolor='red', fill=False, linewidth=1)
    #         handle_props=dict(edgecolor='red', fill=False, linewidth=1)
    #     )
        
    #     # self._region_selector = RectangleSelector(
    #     #     self.ax,
    #     #     onselect=safe_onselect,  # <- use wrapped handler
    #     #     useblit=True,
    #     #     button=[1],  # left-click only
    #     #     spancoords='data',
    #     #     props=dict(edgecolor='red', facecolor='none', linewidth=1)
    #     # )
        
    #     # self.canvas.draw()
    #     # self.canvas.get_tk_widget().focus_set()
        
    # def start_region_selection(self):
    #     """Start RectangleSelector restricted to self.ax only."""
    
    #     def safe_onselect(eclick, erelease):
    #         # Only accept selection if entirely inside self.ax
    #         if eclick.inaxes != self.ax or erelease.inaxes != self.ax:
    #             print("Selection ignored (outside main plot area).")
    #             return
    #         self._on_region_selected(eclick, erelease)
    
    #     # If already exists, re-activate it
    #     if hasattr(self, '_region_selector') and self._region_selector:
    #         self._region_selector.set_active(True)
    #         return
    
    #     self._region_selector = RectangleSelector(
    #         self.ax,
    #         onselect=safe_onselect,
    #         useblit=True,
    #         button=[1],  # Left click only
    #         spancoords='data',
    #         props=dict(edgecolor='red', facecolor='none', linewidth=1)
    #     )
    
    #     self.canvas.draw()
    #     self.canvas.get_tk_widget().focus_set()
    
    # def _on_region_selected(self, eclick, erelease):
        
    #     if not hasattr(self, 'df') or self.df is None:
    #         messagebox.showerror("Selection Error", "No data loaded.")
    #         return
    
    #     # Extract and sort the selected corners
    #     x0, x1 = sorted([eclick.xdata, erelease.xdata])
    #     y0, y1 = sorted([eclick.ydata, erelease.ydata])
        
    #     print(x0, x1, y0, y1)
        
    #     low_x = float(self.lower_shift_var.get())
    #     upp_x = float(self.upper_shift_var.get())
        
    #     low_y = float(self.lower_idx_var.get())
    #     upp_y = float(self.upper_idx_var.get())
        
    #     dx = upp_x - low_x
    #     dy = upp_y - low_y
        
    #     x0_corr = low_x + x0*dx
    #     x1_corr = low_x + x1*dx
        
    #     y0_corr = low_y + y0*dy
    #     y1_corr = low_y + y1*dy
        
    #     print(x0_corr, x1_corr, y0_corr, y1_corr)
    
    #     # Extract real data axis values
    #     shift_values = np.array(self.df.columns, dtype=float)   # x-axis: Raman shifts
    #     index_values = np.array(self.df.index, dtype=float)     # y-axis: scan axis
    
    #     # Find nearest values
    #     low_x_value = shift_values[np.abs(shift_values - x0_corr).argmin()]
    #     upper_x_value = shift_values[np.abs(shift_values - x1_corr).argmin()]
    #     low_y_value = index_values[np.abs(index_values - y0_corr).argmin()]
    #     upper_y_value = index_values[np.abs(index_values - y1_corr).argmin()]
    
    #     # Update the StringVars
    #     self.lower_shift_var.set(f"{low_x_value:.2f}")
    #     self.upper_shift_var.set(f"{upper_x_value:.2f}")
    #     self.lower_idx_var.set(f"{low_y_value:.2f}")
    #     self.upper_idx_var.set(f"{upper_y_value:.2f}")
    
    #     # print(f"Selected shift range: {low_x:.2f} → {high_x:.2f}")
    #     # print(f"Selected index range: {low_y:.2f} → {high_y:.2f}")

    #     # clean up: turn the selector off
    #     self._region_selector.set_active(False)
    #     self._region_selector.disconnect_events()
    #     self._region_selector = None
    
    # def _on_region_selected(self, eclick, erelease):
    #     """
    #     Handle region selection; get real data coordinates and update fields.
    #     """
    
    #     x0, x1 = sorted([eclick.xdata, erelease.xdata])
    #     y0, y1 = sorted([eclick.ydata, erelease.ydata])
    
    #     print(f"Selected region in data coords: x = ({x0:.2f}, {x1:.2f}), y = ({y0:.2f}, {y1:.2f})")
    
    #     # Update your boundary StringVars
    #     self.lower_shift_var.set(f"{x0:.2f}")
    #     self.upper_shift_var.set(f"{x1:.2f}")
    #     self.lower_idx_var.set(f"{y0:.2f}")
    #     self.upper_idx_var.set(f"{y1:.2f}")
    
    #     # Disable the selector
    #     if hasattr(self, '_region_selector') and self._region_selector:
    #         self._region_selector.set_active(False)
    #         self._region_selector.disconnect_events()
    #         self._region_selector = None
    
    # def start_region_selection(self):
    #     """Turn on a RectangleSelector on the map axes."""
    #     # If already exist, just re–activate it
    #     if hasattr(self, '_region_selector') and self._region_selector:
    #         self._region_selector.set_active(True)
    #         return
        
    #     if self.plot_type == "Heatmap":
    #         self._region_selector = RectangleSelector(
    #             self.ax,
    #             onselect=self._on_region_selected,
    #             #drawtype='box',
    #             useblit=True,
    #             button=[1],            # left click only
    #             spancoords='data',     # interpret coords in data space
    #             # rectprops=dict(edgecolor='red', fill=False, linewidth=1)
    #             handle_props=dict(edgecolor='red', fill=False, linewidth=1)
    #         )
    #     elif self.plot_type == "Lineplot"
    #         # Activate spanselector instead - select only the range on the x axis using _on_range_selected
    
    def start_region_selection(self):
        """Turn on appropriate selector based on current plot type."""
    
        # Disable any existing selector
        if hasattr(self, '_region_selector') and self._region_selector:
            self._region_selector.set_active(False)
            self._region_selector.disconnect_events()
            self._region_selector = None
    
        if hasattr(self, '_span_selector') and self._span_selector:
            self._span_selector.set_active(False)
            self._span_selector.disconnect_events()
            self._span_selector = None
    
        if self.plot_type == "Heatmap":
            self._region_selector = RectangleSelector(
                self.ax,
                onselect=self._on_region_selected,
                useblit=True,
                button=[1],  # left click only
                spancoords='data',
                props=dict(edgecolor='red', facecolor='none', linewidth=1)
            )
    
        elif self.plot_type == "Lineplot":
            self._span_selector = SpanSelector(
                self.ax,
                onselect=self._on_range_selected,
                direction='horizontal',
                useblit=True,
                #span_stays=True,
                button=1,  # left click only
                props=dict(alpha=0.5, facecolor='red')
            )

    def _on_region_selected(self, eclick, erelease):
        """
        RectangleSelector callback: eclick and erelease are the mouse‐down
        and mouse‐up events.  We take their data coords, sort them,
        and shove them into the four StringVars.
        """
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        
        # Extract real data axis values
        shift_values = np.array(self.df.columns, dtype=float)   # x-axis: Raman shifts
        index_values = np.array(self.df.index, dtype=float)     # y-axis: scan axis
    
        # Find nearest values
        low_x_value = shift_values[np.abs(shift_values - x0).argmin()]
        upper_x_value = shift_values[np.abs(shift_values - x1).argmin()]
        low_y_value = index_values[np.abs(index_values - y0).argmin()]
        upper_y_value = index_values[np.abs(index_values - y1).argmin()]
    
        # Update the StringVars
        self.lower_shift_var.set(f"{low_x_value:.2f}")
        self.upper_shift_var.set(f"{upper_x_value:.2f}")
        self.lower_idx_var.set(f"{low_y_value:.2f}")
        self.upper_idx_var.set(f"{upper_y_value:.2f}")

        # clean up: turn the selector off
        self._region_selector.set_active(False)
        self._region_selector.disconnect_events()
        self._region_selector = None
        
        # # Replot the Heatmap - call the _on_heatmap funciton
        # self._on_heatmap()
    def _on_range_selected (self, x0, x1):
        """
        Select the range on the x axis and populate the self.lower_shift_var and self.upper_shift_var
        """
        low, high = sorted([x0, x1])
        shift_values = np.array(self.df.columns, dtype=float)
        # Find nearest values
        low_x_value = shift_values[np.abs(shift_values - low).argmin()]
        upper_x_value = shift_values[np.abs(shift_values - high).argmin()]
        self.lower_shift_var.set(f"{low_x_value:.2f}")
        self.upper_shift_var.set(f"{upper_x_value:.2f}")
        # clean up: turn the selector off
        self._span_selector.set_active(False)
        self._span_selector.disconnect_events()
        self._span_selector = None

        # # Replot the overlay plot - call the _on_plot funciton
        # self._on_plot()
        
    def reset_region_selection(self):
        # Make StringVars empty
        self.lower_shift_var.set(self.df.columns.min())
        self.upper_shift_var.set(self.df.columns.max())
        self.lower_idx_var.set(self.df.index.min())
        self.upper_idx_var.set(self.df.index.max())
        
    def plot_spectrum_to_main(self):
        if self.plot_type == "Lineplot":
            messagebox.showinfo("Not yet", "Use the plotting from the Heatmap")
            return
            #data = self._on_overlay_click()
        elif self.plot_type == "Heatmap":
            data = self._on_heatmap_click()
            
        self.plot_callback(data)
        
    def _on_overlay_click(self):
        # Get the y-coordinate of the click
        y_clicked = self._last_pixel[1]
        
        # 6) compute offset step
        max_val     = self.overlay_data.values.max()
        min_val     = self.overlay_data.values.min()
    
        offset_percent = self.slider_var.get()  # slider from 0 to 100
        offset = offset_percent / 100 * max_val
        
        guess_idx = (y_clicked - min_val)/offset
        
        indexes = self.overlay_data.index.to_numpy().astype(float)
        
        idx_estimated = indexes[np.abs(indexes - guess_idx).argmin()]
        
        print(f'estimated index:{idx_estimated}')
        
        shifts = self.overlay_data.columns.to_numpy().astype(float)
        intensity = self.overlay_data.loc[idx_estimated].to_numpy()

    
        data = pd.DataFrame({"X": shifts, "Y": intensity})        
        return data
    
    def _on_heatmap_click(self):
        click_y = self._last_pixel[1]
        
        index_values = np.array(self.df.index, dtype=float)
        index_guess = index_values[np.abs(index_values - click_y).argmin()]
        
        print(f"Selected row (Heatmap): {index_guess}")
        
        shifts = self.heatmap_data.columns.to_numpy().astype(float)
        intensity = self.heatmap_data.loc[index_guess].to_numpy()
        
        data = pd.DataFrame({"X": shifts, "Y": intensity})
        return data

    def _on_fit(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("No data", "No 1D map data is loaded.", parent=self)
            return
    
        if not getattr(self.main, "components", None):
            messagebox.showwarning("No model", "Define a model in the main window.", parent=self)
            return
    
        # lower = self.lower_shift_var.get()
        # upper = self.upper_shift_var.get()
        
        lower = float(self.lower_shift_var.get() or self.df.columns.min())
        upper = float(self.upper_shift_var.get() or self.df.columns.min())
    
        df_fit = Processing.compute_1d_fit(
            self.df,
            self.main.components,
            lower_shift=lower,
            upper_shift=upper,
        )
    
        if df_fit.empty:
            messagebox.showinfo("No results", "Fitting failed for all spectra.")
            return
    
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Save fitting results as…",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if path:
            df_fit.to_excel(path)
            messagebox.showinfo("Saved", f"Fitting results saved to:\n{path}")
