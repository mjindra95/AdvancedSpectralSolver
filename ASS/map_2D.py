# -*- coding: utf-8 -*-
"""
ASS Interafce for analysis of 2D maps
Author: Martin Jindra
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

from ASS.logic import Loading, Processing, Plotting

class Map_2D(tk.Toplevel):
    def __init__(self, main_win, plot_callback):
        # super().__init__(master)
        super().__init__(main_win.root)
        self.main = main_win
        self.title("2D Map Analysis")
        self.geometry("1000x800")
        self.resizable(True, True)
        self.plot_callback = plot_callback
        # self.components = components

        # # keep above the main window
        # self.transient(master)

        # Grid: col0=canvas, col1=controls
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, minsize=300)
        self.rowconfigure(0, weight=1)

        # --- VARIABLES ---
        self.source_var     = tk.StringVar(value="Witec file")
        self.lower_shift    = tk.StringVar()
        self.upper_shift    = tk.StringVar()
        self.lower_x        = tk.StringVar()
        self.upper_x        = tk.StringVar()
        self.lower_y        = tk.StringVar()
        self.upper_y        = tk.StringVar()
        self.plot_var       = tk.StringVar(value="area")
        self.scale_min      = tk.StringVar()
        self.scale_max      = tk.StringVar()
        self.cmap_var       = tk.StringVar(value="viridis")
        self.interp_var     = tk.StringVar(value="none")
        self.fit_component  = tk.StringVar(value="none")

        # this will hold your 3D cube or dict of spectra:
        self.df2d = None
        self.check_4_metric = (None, None)
        self.fit_df = None

        # build UI
        self._create_widgets()

    def _create_widgets(self):
        # ── LEFT: Matplotlib canvas ───────────────────────────────
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6,6), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        # self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        widget = self.canvas.get_tk_widget()
        # widget.pack(fill=tk.BOTH, expand=True)
        widget.grid(row=0, column=0, sticky="nsew")
        # widget.bind("<Button-3>", self._on_map_click)
        # self.canvas.draw()
        
        self.canvas.draw()
        
        self.canvas.mpl_connect("button_press_event", self._on_map_click)
        # ── RIGHT: Controls ────────────────────────────────────────
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.columnconfigure(0, weight=1)

        row = 0
        # Source + Load
        ttk.Label(right, text="Source:").grid(row=row, column=0, sticky="w")
        ttk.Combobox(right, textvariable=self.source_var,
                     values=["Witec file","Horiba split", "Horiba file", "Default split"], state="readonly")\
           .grid(row=row+1, column=0, sticky="ew", pady=(0,10))
        row = row+1
        ttk.Button(right, text="Load", command=self._on_load)\
           .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
          
        #ttk.Separator(right, orient="horizontal")

        # Shift limits
        sf = ttk.Frame(right)
        sf.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        sf.columnconfigure(0, weight=1)
        sf.columnconfigure(1, weight=1)
        ttk.Label(sf, text="Shift min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(sf, textvariable=self.lower_shift).grid(row=1, column=0, sticky="ew", padx=(0,5))
        ttk.Label(sf, text="Shift max:").grid(row=0, column=1, sticky="w")
        ttk.Entry(sf, textvariable=self.upper_shift).grid(row=1, column=1, sticky="ew", padx=(5,0))

        # X limits
        xf = ttk.Frame(right)
        xf.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        xf.columnconfigure(0, weight=1)
        xf.columnconfigure(1, weight=1)
        ttk.Label(xf, text="X min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(xf, textvariable=self.lower_x).grid(row=1, column=0, sticky="ew", padx=(0,5))
        ttk.Label(xf, text="X max:").grid(row=0, column=1, sticky="w")
        ttk.Entry(xf, textvariable=self.upper_x).grid(row=1, column=1, sticky="ew", padx=(5,0))

        # Y limits
        yf = ttk.Frame(right)
        yf.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        yf.columnconfigure(0, weight=1)
        yf.columnconfigure(1, weight=1)
        ttk.Label(yf, text="Y min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(yf, textvariable=self.lower_y).grid(row=1, column=0, sticky="ew", padx=(0,5))
        ttk.Label(yf, text="Y max:").grid(row=0, column=1, sticky="w")
        ttk.Entry(yf, textvariable=self.upper_y).grid(row=1, column=1, sticky="ew", padx=(5,0))

        # Plot variable
        ttk.Label(right, text="Metric variable:").grid(row=row+1, column=0, sticky="w")
        row = row+1
        ttk.Combobox(right, textvariable=self.plot_var,
                     values=["area","maximum","max_position"], state="readonly")\
           .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1

        # Color‐scale bounds
        cf = ttk.Frame(right)
        cf.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        cf.columnconfigure(0, weight=1)
        cf.columnconfigure(1, weight=1)
        ttk.Label(cf, text="Scale min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.scale_min).grid(row=1, column=0, sticky="ew", padx=(0,5))
        ttk.Label(cf, text="Scale max:").grid(row=0, column=1, sticky="w")
        ttk.Entry(cf, textvariable=self.scale_max).grid(row=1, column=1, sticky="ew", padx=(5,0))

        # Colormap
        ttk.Label(right, text="Colormap:").grid(row=row+1, column=0, sticky="w")
        row = row+1
        ttk.Combobox(right, textvariable=self.cmap_var,
                     values=["viridis", "cividis", "plasma", "brg", "nipy_spectral", "inferno", "YlOrBr", "YlOrBr_r"],
                     state="readonly")\
           .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1

        # Interpolation
        ttk.Label(right, text="Interpolation:").grid(row=row+1, column=0, sticky="w")
        row = row+1
        ttk.Combobox(right, textvariable=self.interp_var,
                     values=['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                             'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                             'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'],
                     state="readonly")\
           .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1

        # Plot button
        ttk.Button(right, text="Plot metric", command=self._on_plot)\
           .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        
        # ttk.Button(right, text="Plot average to Main", command=self._on_plot_average)\
        #    .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        # row = row+1
        
        ttk.Button(right, text="Fit", command=self._on_fit) \
          .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
            
        self.param_var = tk.StringVar()
        
        ttk.Label(right, text="Fit variable:").grid(row=row+1, column=0, sticky="w")
        row = row+1
        self.param_menu = ttk.Combobox(right, textvariable=self.param_var,
                              values=[], state="disabled")
        self.param_menu.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        
        self.plot_fit_btn = ttk.Button(right, text="Plot fit",
                                       command=self._on_plot_fit,
                                       state="disabled")
        self.plot_fit_btn.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        
        self.save_fit_btn = ttk.Button(right, text="Save fit",
                                       command=self._on_save_fit,
                                       state="disabled")
        self.save_fit_btn.grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        row = row+1
        
        # # Save button
        # ttk.Button(right, text="Save picture", command=self.save_plot)\
        #    .grid(row=row+1, column=0, sticky="ew", pady=(0,15))
        # row = row+1
        
        # fit_grid = ttk.Frame(right)
        # fit_grid.grid(row=row+16, column=0, sticky="ew", pady=(0,15))
          
        # ttk.Label(fit_grid, text='Fit variable:').grid(row=row+15, column=0, sticky='w')
        # ttk.Combobox(right, textvariable=self.param_var,
        #              values=params,
        #              state="readonly")\
        #     .grid(row= 0, column=0, sticky='ew', pady = (0,15))
            
        # # Plot button
        # ttk.Button(right, text="Plot fit", command=self._on_plot_fit)\
        #    .grid(row=1, column=0, sticky="ew", pady=(0,15))

    # ─── stubs to implement ─────────────────────────────────────
    def _on_load(self):
        source = self.source_var.get()
        
        if source == "Witec file":
            path = filedialog.askopenfilename(
                parent=self,
                title="Load WITec 2D map",
                filetypes=[("TXT files","*.txt"),("All files","*.*")]
            )
            if not path:
                return
        
            try:
                df_raw, meta = Loading.load_witec_map(path)
            except Exception as e:
                messagebox.showerror("Load Error", str(e), parent=self)
                return
        
            # store for plotting & fitting
            self.df2d     = df_raw
            self.map_meta   = meta
            
            shift_low = self.df2d['X-Axis'].iloc[0]
            shift_high = self.df2d['X-Axis'].iloc[-1]
            
            self.lower_shift.set(shift_low)
            self.upper_shift.set(shift_high)
        
            messagebox.showinfo("Loaded",
                f"Map loaded ({df_raw.shape[0]-1} wavenumbers × {len(df_raw.columns)} spectra)\n"
                f"SizeX={meta.get('SizeX')} SizeY={meta.get('SizeY')}\n"
                f"ScanWidth={meta.get('ScanWidth')} ScanHeight={meta.get('ScanHeight')}",
                parent=self
            )
            
        elif source == "Horiba split" or "Default split":
            folder = filedialog.askdirectory(parent = self, title="Select folder with .txt split spectra")
            if not folder:
                return
            try:
                df_raw = Loading.load_horiba_split_2D(folder)
                # messagebox.showinfo("Load info", "loading of horiba split map", parent = self)
            except Exception as e:
                messagebox.showerror("Load error", str(e), parent = self)
                
            self.df2d = df_raw
            self.map_meta   = None
            
            shift_low = self.df2d['X-Axis'].min()
            shift_high = self.df2d['X-Axis'].max()
            
            self.lower_shift.set(shift_low)
            self.upper_shift.set(shift_high)
            
            messagebox.showinfo("Loaded",
                f"Map loaded ({df_raw.shape[0]-1} wavenumbers × {len(df_raw.columns)} spectra)\n", parent=self)

    def _on_plot(self):
        if self.df2d is None:
            messagebox.showwarning("No data", "Load a map first", parent=self)
            return

        # 1) gather bounds & options
        ls = self.lower_shift.get().strip()
        lower_shift = float(ls) if ls else None
        us = self.upper_shift.get().strip()
        upper_shift = float(us) if us else None
        lx = self.lower_x.get().strip()
        lower_x = float(lx) if lx else None
        ux = self.upper_x.get().strip()
        upper_x = float(ux) if ux else None
        ly = self.lower_y.get().strip()
        lower_y = float(ly) if ly else None
        uy = self.upper_y.get().strip()
        upper_y = float(uy) if uy else None
        
        metric = self.plot_var.get()        # "Area", "Maximum", "Position"
        cmap   = self.cmap_var.get()
        interp = self.interp_var.get()
        
        # print("variables gathered")
        
        if self.check_4_metric[0] != lower_shift or self.check_4_metric[1] != upper_shift:
            self.check_4_metric = (lower_shift, upper_shift)
            run_analysis = True
        else:
            run_analysis = False
            #print("Skipping the analysis because the bounds does not changed")
            
        #print(f"Decided if run analysis is needed, run_analysis = {run_analysis}")
            
        if run_analysis == True:
            try:
                self.metric_df = Processing.compute_2d_metric(self.df2d, lower_shift, upper_shift)
                # print("Metric calculated")
                # print(self.metric_df)
            except ValueError as e:
                messagebox.showwarning("Processing error", str(e), parent=self)
                return

        # 3) plot
        vmin = float(self.scale_min.get()) if self.scale_min.get().strip() else None
        vmax = float(self.scale_max.get()) if self.scale_max.get().strip() else None

        new_ax = Plotting.plot_2d(
            ax=         self.ax,
            df=    self.metric_df,
            meta_data = self.map_meta,
            source =    metric,
            cmap_name = cmap,
            interp =    interp,
            low_x =     lower_x,
            upp_x =     upper_x,
            low_y =     lower_y,
            upp_y =     upper_y,
            range_min = vmin,
            range_max = vmax)

        # 4) redraw
        self.ax = new_ax
        self.canvas.draw()
        
    def _on_map_click(self, event):
        # print("Click recieved")
        # only care about right clicks *inside* your heatmap axes
        if event.inaxes is not self.ax:
            return
        
        if event.button != 3:
            return
    
        # record which pixel
        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))
        self._last_pixel = (xpix, ypix)
        
        # print(self._last_pixel)
    
        # build the menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Plot Spectra in Main", command=self._plot_last_pixel)
        menu.add_command(label="Plot Average in Main", command=self._on_plot_average)
        menu.add_command(label="Select region", command=self.start_region_selection)
        menu.add_command(label="Reset region", command=self.reset_region_selection)
        menu.add_command(label="Save map", command=self.save_plot)
    
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
        
    def _plot_last_pixel(self):
        xpix, ypix = self._last_pixel
        
        ls = self.lower_shift.get().strip()
        lower_shift = float(ls) if ls else None
        us = self.upper_shift.get().strip()
        upper_shift = float(us) if us else None

        # find the column header with "(xpix/ypix)"
        pattern = f"({xpix}/{ypix})"
        col = next((c for c in self.df2d.columns if pattern in str(c)), None)
        if col is None:
            messagebox.showerror("Not found",
                                 f"No spectrum at ({xpix},{ypix})",
                                 parent=self)
            return

        # extract shift & intensity, then clip by current zoom
        shift = self.df2d.iloc[:,0].astype(float)
        inten = self.df2d[col].astype(float)
        # lo, hi = getattr(self, "batch_xlim", (shift.min(), shift.max()))
        # mask = (shift >= lo) & (shift <= hi)
        mask = (shift >= lower_shift) & (shift <= upper_shift)
        xs = shift[mask].values
        ys = inten[mask].values

        # hand off to main window
        self.plot_callback(xs, ys, label=f"Pixel {xpix},{ypix}")
        
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

    def start_region_selection(self):
        """Turn on a RectangleSelector on the map axes."""
        # If already exist, just re–activate it
        if hasattr(self, '_region_selector') and self._region_selector:
            self._region_selector.set_active(True)
            return

        self._region_selector = RectangleSelector(
            self.ax,
            onselect=self._on_region_selected,
            #drawtype='box',
            useblit=True,
            button=[1],            # left click only
            spancoords='data',     # interpret coords in data space
            # rectprops=dict(edgecolor='red', fill=False, linewidth=1)
            handle_props=dict(edgecolor='red', fill=False, linewidth=1)
        )

    def _on_region_selected(self, eclick, erelease):
        """
        RectangleSelector callback: eclick and erelease are the mouse‐down
        and mouse‐up events.  We take their data coords, sort them,
        and shove them into the four StringVars.
        """
        
        
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        
        x_low = int(x0)
        x_high = int(x1)
        
        y_low = int(y0)
        y_high = int(y1)

        # Populate your StringVars (or whatever you called them)
        self.lower_x.set(x_low)
        self.upper_x.set(x_high)
        self.lower_y.set(y_low)
        self.upper_y.set(y_high)

        # clean up: turn the selector off
        self._region_selector.set_active(False)
        self._region_selector.disconnect_events()
        self._region_selector = None
        
    def reset_region_selection(self):
        # Make StringVars empty
        self.lower_x.set("")
        self.upper_x.set("")
        self.lower_y.set("")
        self.upper_y.set("")
    
    def _on_plot_average(self):
        # 1) make sure we have data
        if getattr(self, "df2d", None) is None:
            messagebox.showwarning("No map", "Load a map first", parent=self)
            return

        # 2) grab all 6 limits (use None if blank)
        try:
            ls = float(self.lower_shift.get()) if self.lower_shift.get().strip() else None
            us = float(self.upper_shift.get()) if self.upper_shift.get().strip() else None
            lx = float(self.lower_x.get())     if self.lower_x.get().strip()     else None
            ux = float(self.upper_x.get())     if self.upper_x.get().strip()     else None
            ly = float(self.lower_y.get())     if self.lower_y.get().strip()     else None
            uy = float(self.upper_y.get())     if self.upper_y.get().strip()     else None
        except ValueError:
            messagebox.showerror("Input error", "Bounds must be numbers", parent=self)
            return
        # 3) compute average spectrum

        try:
            x_avg, y_avg = Processing.compute_average(
                self.df2d,
                lower_shift=ls, upper_shift=us,
                lower_x=lx,     upper_x=ux,
                lower_y=ly,     upper_y=uy
            )

        except Exception as e:
            messagebox.showwarning("Average failed", str(e), parent=self)
            return

        self.plot_callback(x_avg, y_avg, label="Average spectrum")
        
        # # 5) (optional) you could also store it on self for later
        # self.x_avg, self.y_avg = x_avg, y_avg
        
    def _on_fit(self):
        # 1) make sure we have a map loaded
        if getattr(self, "df2d", None) is None:
            messagebox.showwarning("No data", "Load a map first", parent=self)
            return


        if not getattr(self.main, "components", None):
            messagebox.showwarning(
                "No model",
                "Define a composite model in the main window before fitting",
                parent=self
            )
            return


        # 3) grab the shift-range (you could also reuse self.lower_shift_var, etc.)
        #    fall back to full span if the fields are blank
        shifts = self.df2d.iloc[:,0].astype(float)
        lo = float(self.lower_shift.get() or shifts.min())
        hi = float(self.upper_shift.get() or shifts.max())
        
        # 4) run the 2D fit
        try:
            self.fit_df = Processing.compute_2d_fit(
                df_raw      = self.df2d,
                # components  = main.components,
                components  = self.main.components,
                lower_shift = lo,
                upper_shift = hi
            )
            # print("DataFrame with fitted values:")
            # print(self.fit_df)
        except Exception as e:
            messagebox.showerror("Fit Error", f"2D fitting failed:\n{e}", parent=self)
            return
        print("Fit df:")
        print(self.fit_df)

        # now enable & populate the controls:
        params = [c for c in self.fit_df.columns if c not in ("X","Y")]
        self.param_menu.config(values=params, state="readonly")
        self.param_var.set(params[0])
        self.plot_fit_btn.config(state="normal")        
        self.save_fit_btn.config(state="normal")        


        # 5) notify the user
        messagebox.showinfo("Done", "2D batch fitting finished.", parent=self)


    def _on_plot_fit(self):
        if self.fit_df is None:
            messagebox.showwarning("No data", "Fit map first", parent=self)
            return
        
        lx = self.lower_x.get().strip()
        lower_x = float(lx) if lx else None
        ux = self.upper_x.get().strip()
        upper_x = float(ux) if ux else None
        ly = self.lower_y.get().strip()
        lower_y = float(ly) if ly else None
        uy = self.upper_y.get().strip()
        upper_y = float(uy) if uy else None
        
        metric = self.param_var.get()        # "Area", "Maximum", "Position"
        cmap   = self.cmap_var.get()
        interp = self.interp_var.get()
        # 3) plot
        vmin = float(self.scale_min.get()) if self.scale_min.get().strip() else None
        vmax = float(self.scale_max.get()) if self.scale_max.get().strip() else None

        new_ax = Plotting.plot_2d(
            ax=         self.ax,
            df=    self.fit_df,
            meta_data = self.map_meta,
            source =    metric,
            cmap_name = cmap,
            interp =    interp,
            low_x =     lower_x,
            upp_x =     upper_x,
            low_y =     lower_y,
            upp_y =     upper_y,
            range_min = vmin,
            range_max = vmax)

        # 4) redraw
        self.ax = new_ax
        self.canvas.draw()
    
    def _on_save_fit(self):
        # 1) Make sure we actually have fit data
        if not hasattr(self, "fit_df") or self.fit_df is None or self.fit_df.empty:
            messagebox.showwarning("No data", "Fit map first", parent=self)
            return
    
        # 2) Ask the user for a filename (including path)
        path = filedialog.asksaveasfilename(
            title="Save fit results as…",
            defaultextension=".xlsx",
            filetypes=[("Excel file", "*.xlsx"), ("All files", "*.*")]
        )
        if not path:
            return  # user cancelled
    
        # 3) Write the DataFrame to that exact file
        try:
            self.fit_df.to_excel(path, index=False)
            messagebox.showinfo("Saved", f"Fit results saved to:\n{path}", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save fit results:\n{e}", parent=self)
