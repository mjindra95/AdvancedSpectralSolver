# -*- coding: utf-8 -*-
"""
Created on Sun May 11 15:19:48 2025

@author: Martin Jindra
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Tk, Label
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import sys, json, csv, os, webbrowser
import numpy as np
import pandas as pd
from collections import OrderedDict

from ASS.logic import Loading, Plotting, Processing, Filtering
from ASS.file_utils import File_utils
from ASS.model_builder_5 import ModelBuilderWindow
from ASS.functions import model_dict
from ASS.filtering import FilterWindow
from ASS.map_1D import Map_1D
from ASS.map_2D import Map_2D
from ASS.excel_plotter import ExcelPlotWindow

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Spectral Solver")
        # # Get user screen resolution
        # screen_width = self.root.winfo_screenwidth()
        # screen_height = self.root.winfo_screenheight()

        # # Set window size (for example, 90% of screen dimensions)
        # window_width = int(screen_width * 0.8)
        # window_height = int(screen_height * 0.8)

        # # Center the window
        # x_pos = (screen_width - window_width) // 2
        # y_pos = (screen_height - window_height) // 2

        # self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        self.root.state("zoomed")
        self.root.resizable(True, True)

        # self.root.geometry("1600x900")
        # self.root.resizable(True, True)
        self._create_menu()
        self._create_layout()
        self._create_plot()
        self._bind_shortcuts()
        self.zoom_enabled = False
        self.batch_xlim = None
        self.rectangle_selector = None
        self.filtered_data = None
        self.active_filter = None
        self.compare_data = None
        self.compare_trigger = False
        
        self.filename = None
        self.compare_filename = None
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Bind resize event for the plot panel
        self.plot_panel.bind("<Configure>", self._resize_plot)
        
        self.components = [] 
        
        # Force an initial resize once layout is ready
        self.root.after(100, self._force_initial_resize)

    def _force_initial_resize(self):
        """Trigger an initial resize after the window is fully drawn."""
        w = self.plot_panel.winfo_width()
        h = self.plot_panel.winfo_height()
        if w > 10 and h > 10:
            dpi = self.fig.get_dpi()
            self.fig.set_size_inches(w / dpi, h / dpi)
            self.canvas.draw()
        
    # def _resize_plot(self, event):
    #     """Resize the matplotlib figure when the plot panel changes size."""
    #     if event.width > 50 and event.height > 50:  # avoid tiny startup events
    #         self.fig.set_size_inches(event.width / 300, event.height / 300)
    #         self.canvas.draw()

    def on_close(self):
        self.root.quit()
        self.root.destroy()
        sys.exit()

    def _create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Horiba spectrum", command=self.load_horiba_data, accelerator="Ctrl+h")
        file_menu.add_command(label="Load Witec spectrum", command=self.load_witec_data, accelerator="Ctrl+w")
        file_menu.add_command(label="Load default spectrum", command=self.load_default_data, accelerator="Ctrl+d")
        # file_menu.add_separator()
        # file_menu.add_command(label="Save Plot",    command=self.save_plot,    accelerator="Ctrl+p")
        # file_menu.add_command(label="Save Report",  command=self.save_report,  accelerator="Ctrl+r")
        # file_menu.add_command(label="Save Both",    command=self.save_both,    accelerator="Ctrl+e")
        # file_menu.add_command(label="Save data", command = self.save_data)
        # file_menu.add_command(label="Save spectrum", command = self.save_spectrum)
        # file_menu.add_separator()
        # file_menu.add_command(label="Compare plot", command=self.show_compare)
        menubar.add_cascade(label="File", menu=file_menu)
        
        spectrum_menu = tk.Menu(menubar, tearoff=0)
        spectrum_menu.add_command(label="Compare spectrum", command=self.compare_spectrum)
        spectrum_menu.add_command(label="Disable compare", command=self.disable_compare)
        spectrum_menu.add_separator()
        spectrum_menu.add_command(label="Spectrum operation", command=self.spectrum_operation)
        spectrum_menu.add_command(label="Save spectrum (data)", command = self.save_spectrum)
        spectrum_menu.add_command(label="Save spectrum (plot)", command = self.save_plot)
        menubar.add_cascade(label="Spectrum", menu=spectrum_menu)

        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Build/Edit Model", command=self.open_model_builder, accelerator = "Ctrl+b")
        model_menu.add_separator()
        model_menu.add_command(label="Load Model", command=self.load_model, accelerator="Ctrl+l")
        model_menu.add_command(label="Save Model", command=self.save_model, accelerator="Ctrl+m")
        model_menu.add_command(label="Clear Model", command=self.clear_model)
        model_menu.add_separator()
        model_menu.add_command(label="Optimize Model", command=self.optimize_model, accelerator="Ctrl+f")
        model_menu.add_separator()
        model_menu.add_command(label="Save Plot",    command=self.save_plot,    accelerator="Ctrl+p")
        model_menu.add_command(label="Save Report",  command=self.save_report,  accelerator="Ctrl+r")
        model_menu.add_command(label="Save Both",    command=self.save_both,    accelerator="Ctrl+e")
        model_menu.add_command(label="Save data", command = self.save_data)
        menubar.add_cascade(label="Model", menu=model_menu)

        advanced_menu = tk.Menu(menubar, tearoff=0)
        # advanced_menu.add_command(label="Batch Fit", command=self.batch_fit)
        # advanced_menu.add_separator()
        advanced_menu.add_command(label="Filtering", command=self.open_filter_window)
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Batch analysis", command=self.batch_fit)
        advanced_menu.add_command(label="1D Map analysis", command=self.map_1D)
        advanced_menu.add_command(label="2D Map analysis", command=self.map_2D)
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Excel plot", command = self.excel_plot)
        menubar.add_cascade(label="Advanced", menu=advanced_menu)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About", command=self.show_about)
        about_menu.add_command(label="Help", command=self.show_help)
        menubar.add_cascade(label="About", menu=about_menu)

        self.root.config(menu=menubar)

    def _create_layout(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_frame, width=250)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.plot_panel = tk.Frame(self.main_frame)
        self.plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_path = resource_path("ASS/just_logo_nobcg.png")
        img = Image.open(image_path)
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        self.logo_img = ImageTk.PhotoImage(img)  # store as instance attribute!
        self.logo_label = tk.Label(self.left_panel, image=self.logo_img)
        self.logo_label.pack(pady=(0, 5))
        
        # ASS Workflow
        self.info_label = ttk.Label(self.left_panel, text="Analysis Workflow", anchor = "center")
        self.info_label.pack(fill=tk.X, pady=(10, 2))
        self.sep = ttk.Separator(self.left_panel, orient="horizontal")
        self.sep.pack(fill=tk.X, pady=(5, 5))
        # self.info_label = ttk.Label(self.left_panel, text="---Load data---", anchor = "center")
        # self.info_label.pack(fill=tk.X, pady=(10, 2))
        self.load_button = ttk.Button(self.left_panel, text="Load", command=self.open_load_window)
        self.load_button.pack(fill=tk.X, pady=5)
        self.sep = ttk.Separator(self.left_panel, orient="horizontal")
        self.sep.pack(fill=tk.X, pady=(5, 5))
        # self.info_label = ttk.Label(self.left_panel, text="---Zoom in---", anchor = "center")
        # self.info_label.pack(fill=tk.X, pady=(10, 2))
        self.zoom_button = ttk.Button(self.left_panel, text="Zoom", command=self.toggle_zoom)
        self.zoom_button.pack(fill=tk.X, pady=5)
        ttk.Button(self.left_panel, text="Reset Zoom", command=self.reset_zoom).pack(fill=tk.X, pady=5)
        self.sep = ttk.Separator(self.left_panel, orient="horizontal")
        self.sep.pack(fill=tk.X, pady=(5, 5))
        # self.info_label = ttk.Label(self.left_panel, text="---Model---", anchor = "center")
        # self.info_label.pack(fill=tk.X, pady=(10, 2))
        ttk.Button(self.left_panel, text="Built/Edit Model", command=self.open_model_builder).pack(fill=tk.X, pady=5)
        ttk.Button(self.left_panel, text="Optimize Model", command=self.optimize_model).pack(fill=tk.X, pady=5)

    def _create_plot(self):
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Spectrum")
        self.ax.set_xlabel("Raman Shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        # self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # add right-click popup
        widget.bind("<Button-3>", self._on_canvas_right_click)

        self.canvas.draw()
        
        # Bind the resize of plot_panel to rescale the figure
        self.plot_panel.bind("<Configure>", self._resize_plot)
        
    def _resize_plot(self, event):
        """Rescale the Matplotlib canvas when the plot panel is resized."""
        if event.width > 10 and event.height > 10:
            dpi = self.fig.get_dpi()
            # convert panel size in pixels to figure size in inches
            self.fig.set_size_inches(event.width / dpi, event.height / dpi)
            self.canvas.draw()
        
    def _on_canvas_right_click(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Save Plot",   command=self.save_plot)
        menu.add_command(label="Save Report", command=self.save_report)
        menu.add_command(label="Save Both",   command=self.save_both)
        menu.add_command(label="Save Data", command=self.save_data)
        menu.add_command(label="Save Spectrum", command=self.save_spectrum)
        menu.tk_popup(event.x_root, event.y_root)

    def _bind_shortcuts(self):
        self.root.bind("<Control-h>", lambda event: self.load_horiba_data())
        self.root.bind("<Control-w>", lambda event: self.load_witec_data())
        self.root.bind("<Control-d>", lambda event: self.load_default_data())
        self.root.bind("<Control-m>", lambda event: self.save_model())
        self.root.bind("<Control-l>", lambda event: self.load_model())
        self.root.bind("<Control-f>", lambda event: self.optimize_model())
        self.root.bind("<Control-p>", lambda event: self.save_plot())
        self.root.bind("<Control-r>", lambda event: self.save_report())
        self.root.bind("<Control-e>", lambda event: self.save_both())
        self.root.bind("<Control-b>", lambda event: self.open_model_builder())

    def load_horiba_data(self):
        print("Load Horiba data triggered")
        path = File_utils.ask_spectrum_file("Load Horiba spectrum")
        if not path:
            return
        try:
            if self.compare_trigger == True:
                self.compare_data_x, self.compare_data_y, self.compare_filename = Loading.load_horiba(path)
                self.compare_data_raw = pd.DataFrame({'X': self.compare_data_x, 'Y': self.compare_data_y})
                self.compare_data = self.compare_data_raw.copy()
                if self.batch_xlim is not None:
                    xmin, xmax = self.batch_xlim
                    xmin, xmax = float(xmin), float(xmax)
                    mask = (self.compare_data['X'] >= xmin) & (self.compare_data['X'] <= xmax)
                    self.compare_data = self.compare_data.loc[mask].reset_index(drop=True)
                self.compare_trigger = False
            else:
                self.x_data, self.y_data, self.filename = Loading.load_horiba(path)
                self.raw_data = pd.DataFrame({'X': self.x_data, 'Y': self.y_data})
                self.display_data = self.raw_data.copy()
                self.filtered_data = None
                if getattr(self, "active_filter", None) is not None:
                    func_name, fparams = self.active_filter
                    x_slice = self.display_data["X"].values
                    y_slice = self.display_data["Y"].values
                    try:
                        y_filt = getattr(Filtering, func_name)(x_slice, y_slice, **fparams)
                        self.filtered_data = pd.DataFrame({"X": x_slice, "Y": y_filt})
                    except Exception as fe:
                        messagebox.showwarning(
                            "Filter Warning",
                            f"Could not reapply filter to loaded Horiba data:\n{fe}\n"
                            "Showing unfiltered data instead."
                        )
                        self.filtered_data = None
            self.update_composite_plot()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Horiba data: {e}")
            
    def load_witec_data(self):
        print("Load Witec data triggered")
        path = File_utils.ask_spectrum_file("Load Witec spectrum")
        if not path:
            return
        try:
            if self.compare_trigger == True:
                self.compare_data_x, self.compare_data_y, self.compare_filename = Loading.load_witec(path)
                self.compare_data_raw = pd.DataFrame({'X': self.compare_data_x, 'Y': self.compare_data_y})
                self.compare_data = self.compare_data_raw.copy()
                if self.batch_xlim is not None:
                    xmin, xmax = self.batch_xlim
                    xmin, xmax = float(xmin), float(xmax)
                    mask = (self.compare_data['X'] >= xmin) & (self.compare_data['X'] <= xmax)
                    self.compare_data = self.compare_data.loc[mask].reset_index(drop=True)
                self.compare_trigger = False
            else:
                self.x_data, self.y_data, self.filename = Loading.load_witec(path)
                self.raw_data = pd.DataFrame({'X': self.x_data, 'Y': self.y_data})
                self.display_data = self.raw_data.copy()
                self.filtered_data = None
                if getattr(self, "active_filter", None) is not None:
                    func_name, fparams = self.active_filter
                    x_slice = self.display_data["X"].values
                    y_slice = self.display_data["Y"].values
                    try:
                        y_filt = getattr(Filtering, func_name)(x_slice, y_slice, **fparams)
                        self.filtered_data = pd.DataFrame({"X": x_slice, "Y": y_filt})
                    except Exception as fe:
                        messagebox.showwarning(
                            "Filter Warning",
                            f"Could not reapply filter to loaded data:\n{fe}\n"
                            "Showing unfiltered data instead."
                        )
                        self.filtered_data = None
            self.update_composite_plot()
            # Plotting.plot_data(self.ax, self.display_data, label=filename)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Witec data: {e}")
            
    def load_default_data(self):
        print("Load default data triggered")
        path = File_utils.ask_spectrum_file("Load default datafile")
        if not path:
            return
        try:
            if self.compare_trigger == True:
                self.compare_data_x, self.compare_data_y, self.compare_filename = Loading.load_default(path)
                self.compare_data_raw = pd.DataFrame({'X': self.compare_data_x, 'Y': self.compare_data_y})
                self.compare_data = self.compare_data_raw.copy()
                if self.batch_xlim is not None:
                    xmin, xmax = self.batch_xlim
                    xmin, xmax = float(xmin), float(xmax)
                    mask = (self.compare_data['X'] >= xmin) & (self.compare_data['X'] <= xmax)
                    self.compare_data = self.compare_data.loc[mask].reset_index(drop=True)
                self.compare_trigger = False
            else:
                self.x_data, self.y_data, self.filename = Loading.load_default(path)
                self.raw_data = pd.DataFrame({'X': self.x_data, 'Y': self.y_data})
                self.display_data = self.raw_data.copy()
                self.filtered_data = None
                if getattr(self, "active_filter", None) is not None:
                    func_name, fparams = self.active_filter
                    x_slice = self.display_data["X"].values
                    y_slice = self.display_data["Y"].values
                    try:
                        y_filt = getattr(Filtering, func_name)(x_slice, y_slice, **fparams)
                        self.filtered_data = pd.DataFrame({"X": x_slice, "Y": y_filt})
                    except Exception as fe:
                        messagebox.showwarning(
                            "Filter Warning",
                            f"Could not reapply filter to loaded Horiba data:\n{fe}\n"
                            "Showing unfiltered data instead."
                        )
                        self.filtered_data = None
            self.update_composite_plot()
            #Plotting.plot_data(self.ax, self.display_data, label=filename)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default data: {e}")

    def open_load_window(self):
        """Popup a small window with Load Horiba / Load Witec buttons."""
        win = tk.Toplevel(self.root)
        win.title("Load Data")
        win.geometry("240x120")
        win.transient(self.root)
        win.grab_set()   # make it modal

        ttk.Button(
            win,
            text="Load Horiba",
            command=lambda: (win.destroy(), self.load_horiba_data())
        ).pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            win,
            text="Load Witec",
            command=lambda: (win.destroy(), self.load_witec_data())
        ).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            win,
            text="Load default",
            command=lambda: (win.destroy(), self.load_default_data())
        ).pack(fill=tk.X, padx=10, pady=5)

                
    def toggle_zoom(self):
        if self.rectangle_selector and self.rectangle_selector._selection_artist:
            self.rectangle_selector._selection_artist.remove()
    
        if not self.zoom_enabled:
            self.zoom_enabled = True
            self.span_selector = SpanSelector(
                self.ax,
                onselect=self._apply_zoom_and_disable,
                direction='horizontal',
                useblit=True,
                button=1
            )
            self.zoom_button.config(text="Zoom (ON)")
        else:
            self._disable_zoom()
    
    def _disable_zoom(self):
        self.zoom_enabled = False
        if self.span_selector:
            self.span_selector.disconnect_events()
            self.span_selector = None
        self.zoom_button.config(text="Zoom")
    
    def _apply_zoom_and_disable(self, xmin, xmax):
        # 1) Zoom the displayed raw data
        mask = (self.display_data['X'] >= xmin) & (self.display_data['X'] <= xmax)
        self.display_data = self.display_data.loc[mask].reset_index(drop=True)
        if self.compare_data is not None:
            compare_mask = (self.compare_data['X'] >= xmin) & (self.compare_data['X'] <= xmax)
            self.compare_data = self.compare_data.loc[compare_mask].reset_index(drop=True)
    
        # 2) If a filter is active, re‐apply it to the *new* display_data
        if getattr(self, "active_filter", None) is not None:
            func_name, fparams = self.active_filter
            xz = self.display_data["X"].values
            yz = self.display_data["Y"].values
            try:
                y_filt = getattr(Filtering, func_name)(xz, yz, **fparams)
                self.filtered_data = pd.DataFrame({"X": xz, "Y": y_filt})
            except Exception as fe:
                messagebox.showwarning(
                    "Filter Warning",
                    f"Could not reapply filter '{func_name}' on zoomed region:\n{fe}\n"
                    "Showing unfiltered data instead."
                )
                self.filtered_data = None
    
        # 3) Store zoom limits for batch fit
        self.batch_xlim = (xmin, xmax)
    
        # 4) Redraw everything via the unified plot routine
        self.update_composite_plot()
        self.canvas.draw()
    
        # 5) Turn zoom off
        self._disable_zoom()

    def reset_zoom(self):
        self.display_data = self.raw_data.copy()
        self.compare_data = self.compare_data_raw.copy()
        if getattr(self, "active_filter", None) is not None:
            func_name, fparams = self.active_filter
            xz = self.display_data["X"].values
            yz = self.display_data["Y"].values
            try:
                y_filt = getattr(Filtering, func_name)(xz, yz, **fparams)
                self.filtered_data = pd.DataFrame({"X": xz, "Y": y_filt})
            except Exception as fe:
                messagebox.showwarning(
                    "Filter Warning",
                    f"Could not reapply filter '{func_name}' on zoomed region:\n{fe}\n"
                    "Showing unfiltered data instead."
                )
                self.filtered_data = None
        self.update_composite_plot()
    
    def open_model_builder(self):
        # 1) Build `existing` from your in-memory components
        existing = []
        for i, comp in enumerate(self.components, start=1):
            existing.append({
                'index': i,
                'model_name': comp['model_name'],
                'label':      comp.get('label', comp['model_name']),
                'params': comp['params'],
                'bounds': comp['bounds'],
                'locks':      comp.get('locks', {})
            })
    
        self.builder_window = ModelBuilderWindow(
            master=self.root,
            span_request_callback=self.start_span_selector_for_guess,
            save_callback=self.on_component_saved,
            clear_callback = self.clear_model,
            existing=existing
        )
        
        # if not hasattr(self, "model_builder_window") or not self.builder_window.winfo_exists():
        #     self.builder_window = ModelBuilderWindow(
        #         master=self.root,
        #         span_request_callback=self.start_span_selector_for_guess,
        #         save_callback=self.on_component_saved,
        #         clear_callback=self.clear_model,
        #         existing=self.components
        #     )
        # else:
        #     self.builder_window.lift()  # Bring it to front if already open

            
    def request_span_selection(self, callback):
        if self.span_request_delegate:
            self.grab_release()  # Let the main window get mouse input
            self.span_request_delegate(callback)
    
    def start_span_selector_for_guess(self, block, index, model_name):
        """
        block: the FunctionBlock instance that called Guess
        index/model_name: passed for clarity, but block already knows them
        """
        # 1) Hide the builder
        self.builder_window.withdraw()
    
        # 2) Activate SpanSelector
        def _onselect(xmin, xmax):
            # Re-show the builder
            self.builder_window.deiconify()
            
            mask   = (self.display_data['X'] >= xmin) & (self.display_data['X'] <= xmax)
            x_sel  = self.display_data['X'][mask].values
            y_sel  = self.display_data['Y'][mask].values
            
            # sel = self.display_data[(self.display_data["X"] >= xmin)
            #            & (self.display_data["X"] <= xmax)]
            # x_sel = sel["X"].to_numpy()
            # y_sel = sel["Y"].to_numpy()

            # --- new: compute the composite model over x_sel ---
            comp = np.zeros_like(x_sel)
            for comp_def in self.components:
                fn   = model_dict[comp_def['model_name']]['func']
                pns  = model_dict[comp_def['model_name']]['params']
                pvs  = [comp_def['params'][pn] for pn in pns]
                comp += fn(x_sel, *pvs)
            residual = y_sel - comp
            
            # Let the block fill its fields
            block.receive_span_selection(xmin, xmax, x_sel, y_sel, residual)
            # Clean up
            self.span_selector.set_active(False)
            self.span_selector.disconnect_events()
            self.span_selector = None
            self.canvas.draw()
    
        self.span_selector = SpanSelector(
            self.ax, _onselect,
            direction='horizontal',
            useblit=True,
            button=1)
   
    def _finalize_guess_with_selection(self, xmin, xmax):
        data_slice = self.raw_data[
            (self.raw_data["X"] >= xmin) & (self.raw_data["X"] <= xmax)
        ]
        x = data_slice["X"].values
        y = data_slice["Y"].values
        
        function_param_dict = {name: data["params"] for name, data in model_dict.items()}

        self.builder_window = ModelBuilderWindow(
            self.root,
            function_param_dict,
            span_request_delegate=self.start_span_selector_for_guess,
            prefill_data={"index": self.pending_guess["index"],
                          "model_name": self.pending_guess["model_name"],
                          "x": x,
                          "y": y,
                          "xmin": xmin,
                          "xmax": xmax})
        
    def open_model_builder_with_prefill(self, prefill, existing):
        # Destroy any old builder
        if hasattr(self, 'builder_window') and self.builder_window.winfo_exists():
            self.builder_window.destroy()
        # Then open fresh
        self.builder_window = ModelBuilderWindow(
            master=self.root,
            span_delegate=self.start_span_selector_for_guess,
            existing=existing,
            prefill=prefill
        )
    
    def on_component_saved(self, idx, comp):
        if comp is None:
            # delete
            if 1 <= idx <= len(self.components):
                self.components.pop(idx-1)
        else:
            # upsert
            if idx <= len(self.components):
                self.components[idx-1] = comp
            else:
                self.components.append(comp)
        # redraw
        self.update_composite_plot()
        
    def update_composite_plot(self):
        Plotting.update_plot(self.fig, self.display_data, self.filename, self.components, self.filtered_data, self.compare_data, self.compare_filename)
        # rebind the left-axis in case update_plot re-created it
        self.ax = self.fig.axes[0]
        self.canvas.draw()
        
    def clear_model(self):
        # 1) wipe out the component list
        self.components.clear()

        # 2) redraw the (now empty) composite
        self.update_composite_plot()

    def load_model(self):
        print("Load model triggered")
        path = filedialog.askopenfilename(
            title="Load composite model…",
            filetypes=[("JSON files","*.json"),("All files","*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                comps = json.load(f)
            # Optional validation could go here…
            self.components = comps
            self.update_composite_plot()
            messagebox.showinfo("Loaded", f"Model loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load model:\n{e}")

    def save_model(self):
        print("Save model triggered")
        path = filedialog.asksaveasfilename(
            title="Save composite model…",
            defaultextension=".json",
            filetypes=[("JSON files","*.json"),("All files","*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.components, f, indent=2)
            messagebox.showinfo("Saved", f"Model saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save model:\n{e}")

    def optimize_model(self):
        if not hasattr(self, 'display_data') or self.display_data.empty:
            messagebox.showwarning("Optimize", "No data to fit—please load and zoom or reset first.")
            return
        if not self.components:
            messagebox.showwarning("Optimize", "No model components defined.")
            return
        
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Fitting…")
        progress_win.geometry("300x50")
        progress_win.transient(self.root)
        progress_win.grab_set()
        
        pb = ttk.Progressbar(progress_win, mode="indeterminate")
        pb.pack(fill="x", padx=10, pady=10)
        pb.start(10)                     # move every 10 ms
        self.root.update_idletasks()     # ensure it displays
    
        # 1) Build p0 and bounds arrays
        p0_list = []
        lb_list = []
        ub_list = []
        for comp in self.components:
            name   = comp['model_name']
            pnames = model_dict[name]['params']
            for pn in pnames:
                p0_list.append(comp['params'][pn])
                lb, ub = comp['bounds'][pn]
                lb_list.append(-np.inf if lb is None else lb)
                ub_list.append( np.inf if ub is None else ub)
        
        p0     = np.array(p0_list, dtype=float)
        bounds = (np.array(lb_list, dtype=float), np.array(ub_list, dtype=float))
    
        # 2) Run the fit
        if self.filtered_data is not None:
            proc = Processing(self.filtered_data)
        else:
            proc = Processing(self.display_data)
        try:
            popt, pcov = proc.fit(self.components, p0, bounds)
            perr = np.sqrt(np.diag(pcov))
            self.param_uncertainties = perr
        except Exception as e:
            messagebox.showerror("Fit Error", f"Model optimization failed:\n{e}")
            return
        finally:
            pb.stop()
            progress_win.destroy()
    
        # 3) write back EVERY parameter (locked ones will stay the same)
        idx = 0
        for comp in self.components:
            pnames = model_dict[comp["model_name"]]["params"]
            for pn in pnames:
                comp["params"][pn] = popt[idx]
                idx += 1
        
        if self.filtered_data is not None:
            n = len(self.filtered_data["X"])
        else:
            n = len(self.display_data["X"])
        p = len(popt)
        
        # 4) Redraw the composite + data
        self.update_composite_plot()
        
        # 5) (Optional) Show fit statistics
        residuals = proc.y - proc.composite_func(proc.x, *popt, components=self.components)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((proc.y - np.mean(proc.y))**2)
        r2 = 1 - ss_res/ss_tot
        r2_adj  = 1 - (1 - r2)*(n - 1)/(n - p)
        rmse    = np.sqrt(np.mean(residuals**2))
        # AIC  (Akaike Information Criterion)
        aic = n*np.log(ss_res/n) + 2*p
        # BIC  (Bayesian Information Criterion)
        bic = n*np.log(ss_res/n) + p*np.log(n)
        
        # Correlation matrix from covariance
        #    corr[i,j] = cov[i,j] / sqrt(cov[i,i]*cov[j,j])
        stddevs = np.sqrt(np.diag(pcov))
        corr    = pcov / np.outer(stddevs, stddevs)
        # guard against division by zero if any stddev is zero:
        corr[np.isnan(corr)] = 0
        
        messagebox.showinfo("Fit Complete", f"R² = {r2:.4f}")
        
        # 5) Store the fit statistics so save_report can see them
        self.fit_statistic = {
            'R²':           r2,
            'Adjusted R²':  r2_adj,
            'RMSE':         rmse,
            'AIC':          aic,
            'BIC':          bic,
            'covariance':    pcov.tolist(),
            'correlation':   corr.tolist()
        }
        
        # print("Updating model builder window 2")
        
        # # 6) Notify model builder if it's open
        # if hasattr(self, "model_builder_window") and self.builder_window.winfo_exists():
        #     self.builder_window.update_parameters_from_main(self.components)
            
        # print("Updating model builder window 3")
        self.builder_window.destroy()
        
        existing = []
        for i, comp in enumerate(self.components, start=1):
            existing.append({
                'index': i,
                'model_name': comp['model_name'],
                'label':      comp.get('label', comp['model_name']),
                'params': comp['params'],
                'bounds': comp['bounds'],
                'locks':      comp.get('locks', {})
            })
            
        self.builder_window = ModelBuilderWindow(
            master=self.root,
            span_request_callback=self.start_span_selector_for_guess,
            save_callback=self.on_component_saved,
            clear_callback = self.clear_model,
            existing=existing
        )

        
    def save_plot(self):
        """Export the current canvas as an image."""
        path = filedialog.asksaveasfilename(
            title="Save plot as…",
            defaultextension=".png",
            filetypes=[("PNG image","*.png"),("PDF","*.pdf"),("All files","*.*")]
        )
        if not path: return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Plot Error", str(e))
    
    def save_report(self):
        if not hasattr(self, 'fit_statistic'):
            messagebox.showwarning("No Fit", "You must optimize the model first.")
            return
    
        path = filedialog.asksaveasfilename(
            title="Save report as…",
            defaultextension=".txt",                  # default to .txt
            filetypes=[("Text file","*.txt"),
                       ("JSON file","*.json"),
                       ("CSV file","*.csv"),])
        if not path:
            return
    
        try:
            # build a flat report object
            report = {
                "components": [
                    {"label":       comp.get("label", comp["model_name"]),
                     "model_name":  comp["model_name"],
                     **comp["params"]}
                    for comp in self.components],
                "statistics": self.fit_statistic}
    
            if path.lower().endswith(".txt"):
                # plain‐text report
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n=== Component Parameters ===\n")
                    idx = 0
                    for comp in report["components"]:
                        lbl = comp.pop("label")
                        mn  = comp.pop("model_name")
                        f.write(f"\n[{lbl} ({mn})]\n")
                        for pname, pval in comp.items():
                            err = self.param_uncertainties[idx]
                            f.write(f"  {pname}: {pval:.4g} ± {err:.4g}\n")
                            idx += 1
                    f.write("=== Fit Statistics ===\n")
                    for k, v in self.fit_statistic.items():
                        if k in ('covariance','correlation'):
                            f.write(f"\n{k} matrix:\n")
                            mat = np.array(v)
                            for row in mat:
                                f.write("  " + "  ".join(f"{x:.4g}" for x in row) + "\n")
                        else:
                            f.write(f"{k}: {v:.6g}\n")
            
            elif path.lower().endswith(".json"):
                # build enriched report with uncertainties
                json_report = {
                    "components": [],
                    "statistics": self.fit_statistic
                }
                idx = 0
                for comp_def in self.components:
                    model = comp_def["model_name"]
                    lbl   = comp_def.get("label", model)
                    pnames = model_dict[model]["params"]
            
                    comp_entry = {
                        "label":        lbl,
                        "model_name":   model,
                        "params":       {},
                        "uncertainties": {}
                    }
                    for pn in pnames:
                        comp_entry["params"][pn]       = comp_def["params"][pn]
                        comp_entry["uncertainties"][pn] = self.param_uncertainties[idx]
                        idx += 1
            
                    json_report["components"].append(comp_entry)
            
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(json_report, f, indent=2)

            else:  # CSV fallback
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
            
                    # Build a header that alternates param and param_err
                    header = ["Component", "Model"]
                    for comp_def in self.components:
                        lbl = comp_def.get("label", comp_def["model_name"])
                        for pn in model_dict[comp_def["model_name"]]["params"]:
                            header.append(f"{lbl}_{pn}")
                            header.append(f"{lbl}_{pn}_err")
                    writer.writerow(header)
            
                    # Now write the single row of values
                    row = []
                    row.append(self.components[0].get("label"))   # assuming single‐spectrum save
                    row.append(self.components[0]["model_name"])  # or better: loop each component
                    idx = 0
                    for comp_def in self.components:
                        for pn in model_dict[comp_def["model_name"]]["params"]:
                            row.append(comp_def["params"][pn])
                            row.append(self.param_uncertainties[idx])
                            idx += 1
                    writer.writerow(row)
            
                    # blank + stats
                    writer.writerow([])
                    for k, v in self.fit_statistic.items():
                        writer.writerow([k, v])

            messagebox.showinfo("Saved", f"Report saved to:\n{path}")
    
        except Exception as e:
            messagebox.showerror("Save Report Error", str(e))

    def save_both(self):
        """Save plot and report in one go."""
        self.save_plot()
        self.save_report()
        
    def save_spectrum(self):
        # 1) Check we have something to save
        if not hasattr(self, "display_data") or self.display_data is None:
            messagebox.showwarning("Save Data", "No spectrum to save. Load/zoom some data first.")
            return
    
        # 2) Ask for a filename
        path = filedialog.asksaveasfilename(
            title="Save spectrum as…",
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"),
                ("CSV file", "*.csv"),
                ("Excel file", "*.xlsx"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
    
        ext = os.path.splitext(path)[1].lower()
    
        try:
            if ext == ".txt":
                # plain two-column whitespace-separated
                with open(path, "w", encoding="utf-8") as f:
                    for x, y in zip(self.display_data["X"], self.display_data["Y"]):
                        f.write(f"{x}\t{y}\n")
    
            elif ext == ".csv":
                # comma-separated with header
                self.display_data.to_csv(path, index=False)
    
            elif ext in (".xls", ".xlsx"):
                # one-sheet Excel
                self.display_data.to_excel(path, index=False)
    
            else:
                # fallback: CSV
                self.display_data.to_csv(path, index=False)
    
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save spectrum:\n{e}")
            return
    
        messagebox.showinfo("Saved", f"Spectrum saved to:\n{path}")
        
    def save_data(self):
        #save the data (raw, composite and each peak separately) in excel, csv or txt file
        # messagebox.showinfo("Save plot data", "I see, that you want to save the plot data, well you are lucky, because this feature will be added soon!")
        # 1) Ensure there is data to save
        if not hasattr(self, "display_data") or self.display_data is None:
            messagebox.showwarning("Save Data", "No data to save. Load/zoom some data first.")
            return
        if not self.components:
            messagebox.showwarning("Save Data", "No composite model defined. Build a model first.")
            return
        
        # 2) Ask for a filename
        path = filedialog.asksaveasfilename(
            title="Save displayed data as…",
            defaultextension=".xlsx",
            filetypes=[("Excel file","*.xlsx"), ("All files","*.*")]
        )
        if not path:
            return
        
        try:
            # 3) Build the “Raw & Residual” sheet
            df_raw = self.display_data.copy().reset_index(drop=True)
            x_raw  = df_raw["X"].values
            y_raw  = df_raw["Y"].values
            n_raw = len(x_raw)
            
            if hasattr(self, "filtered_data") and self.filtered_data is not None:
                # Make sure filtered_data is indexed same as display_data
                df_filt = self.filtered_data
                # If user zoomed, df_filt["X"] should match x_raw exactly
                df_raw["Filtered"] = df_filt["Y"].values.astype(float)
            else:
                # no filter → fill with NaN
                df_raw["Filtered"] = np.full(n_raw, np.nan, dtype=float)
        
            # 3a) Compute model_at_x for residual:
            model_at_x = np.zeros_like(x_raw, dtype=float)
            for comp in self.components:
                name   = comp["model_name"]
                func   = model_dict[name]["func"]
                pnames = model_dict[name]["params"]
                pvals  = [comp["params"].get(pn, 0.0) for pn in pnames]
                model_at_x += func(x_raw, *pvals)
        
            df_raw["Residual"] = y_raw - model_at_x
        
            # 4) Build the “Model & Components” sheet
            # Use a dense X grid ten times finer than the raw points:
            x_min, x_max = x_raw.min(), x_raw.max()
            x_dense = np.linspace(x_min, x_max, len(x_raw) * 10)
            composite_dense = np.zeros_like(x_dense, dtype=float)
        
            # For each component, evaluate on x_dense and accumulate
            comp_dict = {}  # will hold each component’s y_dense array
            for comp in self.components:
                name   = comp["model_name"]
                label  = comp.get("label", name)
                func   = model_dict[name]["func"]
                pnames = model_dict[name]["params"]
                pvals  = [comp["params"].get(pn, 0.0) for pn in pnames]
        
                y_comp_dense = func(x_dense, *pvals)
                composite_dense += y_comp_dense
                # store it under its label (so the column is named after it)
                comp_dict[label] = y_comp_dense
        
            # Now build a DataFrame whose first two columns are X_dense & Composite:
            df_model = pd.DataFrame({
                "X_dense":  x_dense,
                "Composite": composite_dense
            })
            # Then tack on one column per component, in the same order:
            for label, y_c in comp_dict.items():
                df_model[label] = y_c
        
            # 5) Write both sheets into one Excel file
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df_raw.to_excel(writer, sheet_name="Raw data", index=False)
                df_model.to_excel(writer, sheet_name="Model & Components", index=False)
        
            messagebox.showinfo("Saved", f"Displayed data saved to:\n{path}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save:\n{e}")

    def _do_batch_fit(self, loader_fn):
        """
        Batch‐fit each .txt spectrum in a folder. Steps for each file:
          1) Load and slice (zoom limits).
          2) Re‐apply filter (if any).
          3) Plot the (possibly filtered) data on the main canvas.
          4) Perform the fit (using existing self.components as the model template).
          5) Update self.components with the optimized parameters.
          6) Re‐plot the fitted model on the main canvas.
          7) Save the canvas as an image in plots_dir.
          8) Record parameters + 1σ errors into results[].

        After all files are done, write a single Excel summary in out_base.
        """
        # 1) Ask for a folder
        folder = filedialog.askdirectory(title="Select folder of .txt spectra")
        if not folder:
            return

        # 2) Prepare output directories
        out_base = os.path.join(folder, "batch_fit_results")
        i = 1
        while os.path.exists(f"{out_base}_{i}"):
            i += 1
        out_base = f"{out_base}_{i}"
        plots_dir   = os.path.join(out_base, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 3) Gather all .txt files
        files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".txt")]
        n_files = len(files)
        if n_files == 0:
            messagebox.showinfo("Batch Fit", "No .txt files found in the selected folder.")
            return

        # 4) Show a small progress window
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Batch fitting…")
        progress_win.geometry("400x80")
        progress_win.transient(self.root)
        progress_win.grab_set()

        label = ttk.Label(progress_win, text="Starting batch fit…")
        label.pack(fill="x", padx=10)

        pb = ttk.Progressbar(progress_win, mode="determinate", maximum=n_files)
        pb.pack(fill="x", padx=10, pady=10)

        # 5) Prepare to collect results
        results = []
        x0, x1 = self.batch_xlim
        x0, x1 = float(x0), float(x1)

        # 6) Loop over each file
        for idx_file, fn in enumerate(files, start=1):
            label.config(text=f"Fitting {fn} ({idx_file}/{n_files})")
            progress_win.update_idletasks()

            path = os.path.join(folder, fn)
            try:
                x_all, y_all, basename = loader_fn(path)
                x_all = np.asarray(x_all, dtype=float)
                y_all = np.asarray(y_all, dtype=float)
            except Exception as e:
                print(f"Skipping {fn}: {e}")
                pb.step(1)
                continue

            # 6a) Slice to the zoom range
            mask = (x_all >= x0) & (x_all <= x1)
            x_slice = x_all[mask]
            y_slice = y_all[mask]
            if len(x_slice) == 0:
                print(f"Skipping {fn}: no points in zoom range [{x0}, {x1}]")
                pb.step(1)
                continue

            # Build a DataFrame for the slice (raw)
            self.display_data = pd.DataFrame({"X": x_slice, "Y": y_slice})

            # 6b) If a filter is active, reapply it to this slice
            if self.active_filter is not None:
                func_name, fparams = self.active_filter
                try:
                    y_filt = getattr(Filtering, func_name)(x_slice, y_slice, **fparams)
                    self.filtered_data = pd.DataFrame({"X": x_slice, "Y": y_filt})
                except Exception as fe:
                    messagebox.showwarning(
                        "Filter Warning",
                        f"Could not reapply filter to {fn}: {fe}\nProceeding with unfiltered slice."
                    )
                    self.filtered_data = None
            else:
                self.filtered_data = None

            # 6c) Plot the (possibly filtered) data on the main canvas
            self.update_composite_plot()  
            # At this point, update_composite_plot sees filtered_data != None and draws raw vs. filtered.
            # OR, if no filter, it draws raw + model (but model still has old params).
            self.root.update_idletasks()

            # 6d) Build initial guess p0 & bounds
            p0_list, lb_list, ub_list = [], [], []
            for comp in self.components:
                name   = comp["model_name"]
                pnames = model_dict[name]["params"]
                for pn in pnames:
                    p0_list.append(comp["params"][pn])
                    lo, hi = comp["bounds"][pn]
                    lb_list.append(-np.inf if lo is None else lo)
                    ub_list.append( np.inf if hi is None else hi)
            p0     = np.array(p0_list, dtype=float)
            bounds = (np.array(lb_list, dtype=float), np.array(ub_list, dtype=float))

            # 6e) Fit to the DataFrame (filtered or raw)
            # Use the DataFrame with column "Y" set to filtered if active, else raw
            df_fit = pd.DataFrame({"X": x_slice, "Y": 
                                   (self.filtered_data["Y"].values if self.filtered_data is not None 
                                    else y_slice)})

            proc = Processing(df_fit)
            try:
                popt, pcov = proc.fit(self.components, p0, bounds)
                perr = np.sqrt(np.diag(pcov))
            except Exception as fe:
                print(f"Fit failed for {fn}: {fe}")
                pb.step(1)
                continue

            # 6f) Update self.components with the optimized parameters
            idx_param = 0
            for comp in self.components:
                name   = comp["model_name"]
                pnames = model_dict[name]["params"]
                for pn in pnames:
                    comp["params"][pn] = popt[idx_param]
                    idx_param += 1

            # 6g) Re‐plot the fitted model on the main canvas
            # Clear filtered_data so update_composite_plot draws raw+model
            self.filtered_data = None
            self.update_composite_plot()
            self.root.update_idletasks()

            # 6h) Save the current figure as an image (png)
            img_path = os.path.join(plots_dir, f"{basename}.png")
            self.fig.savefig(img_path, dpi=150)

            # 6i) Record a single row of results (values + errors)
            row = {"file": basename, "zoom_min": x0, "zoom_max": x1}
            flat_idx = 0
            for comp in self.components:
                lbl    = comp.get("label", comp["model_name"])
                pname_list = model_dict[comp["model_name"]]["params"]
                for pname in pname_list:
                    row[f"{lbl}_{pname}"]      = popt[flat_idx]
                    row[f"{lbl}_{pname}_err"]  = perr[flat_idx]
                    flat_idx += 1
            results.append(row)

            # Advance progress bar
            pb.step(1)

        # 7) Tear down the progress window
        progress_win.destroy()

        # 8) Save one Excel summary (all rows in one sheet)
        if results:
            df_res = pd.DataFrame(results)
            excel_path = os.path.join(out_base, "batch_summary.xlsx")
            df_res.to_excel(excel_path, index=False)
            messagebox.showinfo(
                "Batch Fit Complete",
                f"Batch fitting finished!\n\n"
                f"Saved {len(results)} fit results to:\n{excel_path}\n"
                f"Saved plots under:\n{plots_dir}"
            )
        else:
            messagebox.showwarning("Batch Fit", "No successful fits to save.")
    
    def batch_fit(self):
        print("Batch fit triggered")
        # 1) Preconditions
        if not hasattr(self, 'components') or not self.components:
            messagebox.showwarning("Batch Fit", "Define and model first.")
            return
        if not hasattr(self, 'batch_xlim'):
            messagebox.showwarning("Batch Fit", "Zoom into the spectral region of interest")
            return

        # 2) Let user pick which loader via a modal Toplevel
        win = tk.Toplevel(self.root)
        win.title("Which spectrometer?")
        win.geometry("240x160")
        win.transient(self.root)
        win.grab_set()
    
        def choose(loader_fn):
            win.destroy()
            self._do_batch_fit(loader_fn)
    
        ttk.Button(win, text="Horiba spectra",
                   command=lambda: choose(Loading.load_horiba))\
           .pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(win, text="Witec spectra",
                   command=lambda: choose(Loading.load_witec))\
           .pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(win, text="Default spectra",
                   command=lambda: choose(Loading.load_default))\
           .pack(fill=tk.X, padx=10, pady=5)

    def show_about(self):
        messagebox.showinfo("About ASS", "This is the Advanced Spectral Solver, app which made complex spectral analysis easy! Effort = ZE RO!")
    
    def show_help(self):
        help_pdf = os.path.join(os.path.dirname(__file__), "help.pdf")
        if not os.path.exists(help_pdf):
            messagebox.showerror("Help file not found", help_pdf)
            return
        # On Windows, you could also do: os.startfile(help_pdf)
        webbrowser.open_new(help_pdf)
        
    def show_compare(self):
        messagebox.showinfo("Compare data", "Compare of the data will be added soon!")

    def open_filter_window(self):
        """
        Launch the FilterWindow.  If a filter is already active,
        pass its name+params so we can pre-select and prefill.
        """
        FilterWindow(
            master=self.root,
            apply_callback=self.apply_filter,
            disable_callback=self.disable_filter,
            initial=self.active_filter
        )
    
    def apply_filter(self, func_name, params):
        try:
            # 1) If the user has zoomed, use only the displayed subset; otherwise use full raw
            if hasattr(self, "display_data") and self.display_data is not None:
                df_source = self.display_data
            else:
                # fallback to full raw data if no zoom has been applied
                df_source = pd.DataFrame({"X": self.x_data, "Y": self.y_data})
    
            x_src = df_source["X"].values
            y_src = df_source["Y"].values
    
            # 2) apply the filter to just that subset
            y_filt = getattr(Filtering, func_name)(x_src, y_src, **params)
    
            # 3) store filtered only on the zoomed subset
            self.filtered_data = pd.DataFrame({"X": x_src, "Y": y_filt})
            
            self.active_filter = (func_name, params)
    
            # 4) re-plot (update_plot will now show raw vs filtered on the zoomed range)
            self.update_composite_plot()
    
        except Exception as e:
            messagebox.showerror("Filter Error", f"Could not apply filter:\n{e}")


    def disable_filter(self):
        """
        Called by FilterWindow “Disable.” Remove any filter and redraw.
        """
        self.filtered_data = None
        self.active_filter = None
        self.update_composite_plot()

    def map_2D(self):
        Map_2D(self, plot_callback=self.plot_pixel_spectrum)
        # map_2D_window = Map_2D(self.root, plot_callback=self.plot_pixel_spectrum)
        # map_2D_window.transient(self.root)
        # map_2D_window.grab_set()
    
    def plot_pixel_spectrum(self, x, y, label=None):
        # 1) pack the x/y into a DataFrame exactly the way load_data does
        df = pd.DataFrame({"X": x, "Y": y})
        
        self.raw_data     = df.copy()
        self.display_data = df.copy()
        
        self.x_data = x
        self.y_data = y
        
        if getattr(self, "active_filter", None) is not None:
            func_name, fparams = self.active_filter
            x_slice = self.display_data["X"].values
            y_slice = self.display_data["Y"].values
            try:
                y_filt = getattr(Filtering, func_name)(x_slice, y_slice, **fparams)
                self.filtered_data = pd.DataFrame({"X": x_slice, "Y": y_filt})
            except Exception as fe:
                messagebox.showwarning(
                    "Filter Warning",
                    f"Could not reapply filter to loaded Horiba data:\n{fe}\n"
                    "Showing unfiltered data instead."
                )
                self.filtered_data = None
        
        self.update_composite_plot()
    
        # 5) finally, push it onto the canvas
        self.canvas.draw()
        
    # def map_1D(self):
    #     # Open the Map_1D window on top of the main window
    #     Map_1D(self.root)
    
    def map_1D(self):
        Map_1D(self, plot_callback=self.plot_index_spectrum)
        
    def plot_index_spectrum(self, data):
        #this will plot spectrum to Main Window
        #print("Test of plot from 1D map window")
        self.raw_data     = data.copy()
        self.display_data = data.copy()
        
        self.x_data = data["X"]
        self.y_data = data["Y"]
        
        if getattr(self, "active_filter", None) is not None:
            func_name, fparams = self.active_filter
            x_slice = self.display_data["X"].values
            y_slice = self.display_data["Y"].values
            try:
                y_filt = getattr(Filtering, func_name)(x_slice, y_slice, **fparams)
                self.filtered_data = pd.DataFrame({"X": x_slice, "Y": y_filt})
            except Exception as fe:
                messagebox.showwarning(
                    "Filter Warning",
                    f"Could not reapply filter to loaded Horiba data:\n{fe}\n"
                    "Showing unfiltered data instead."
                )
                self.filtered_data = None
        
        self.update_composite_plot()
    
        self.canvas.draw()
        
    def excel_plot(self):
        ExcelPlotWindow(self.root)
        
    def compare_spectrum(self):
        #messagebox.showinfo("Compare spectrum", "Compare spectrum will be added")
        self.compare_trigger = True
        print("Trigger set to True")
        """Popup a small window with Load Horiba / Load Witec buttons."""
        win = tk.Toplevel(self.root)
        win.title("Load Data")
        win.geometry("240x120")
        win.transient(self.root)
        win.grab_set()   # make it modal

        ttk.Button(
            win,
            text="Load Horiba",
            command=lambda: (win.destroy(), self.load_horiba_data())
        ).pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            win,
            text="Load Witec",
            command=lambda: (win.destroy(), self.load_witec_data())
        ).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            win,
            text="Load default",
            command=lambda: (win.destroy(), self.load_default_data())
        ).pack(fill=tk.X, padx=10, pady=5)
        
    def disable_compare(self):
        # self.compare_data_raw = None
        self.compare_data = None
        self.compare_filename = None
        
        self.update_composite_plot()
        self.canvas.draw()
        
    def spectrum_operation(self):
        messagebox.showinfo("Spectrum calculation", "Spectrum calculation will be added")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MainWindow()
    app.run()