# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:29:11 2025

@author: marti

Clean Model Builder with global scrolling, stable indexing,
no renumbering, no lock feature.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.interpolate import interp1d
from ASS.functions import model_dict   # your model functions dict

# ================================================================
#  SIMPLE + ROBUST SCROLLABLE FRAME (WORKS IN EXE)
# ================================================================
class ScrollableFrame(ttk.Frame):
    """A scrollable frame that can contain many child widgets."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfig(self._window_id, width=e.width))

    def scroll(self, delta):
        self.canvas.yview_scroll(delta, "units")

# ================================================================
#  MODEL BUILDER WINDOW
# ================================================================
class ModelBuilderWindow(tk.Toplevel):
    """Toplevel window for building a composite model."""
    def __init__(self, master, span_request_callback, save_callback,
                 clear_callback, existing=None, prefill=None):
        super().__init__(master)
        self.title("Model Builder")
        self.geometry("420x600")

        self.span_request_callback = span_request_callback
        self.save_callback = save_callback
        self.clear_callback = clear_callback
        self.model_functions = model_dict
        self.function_blocks = []

        # --- left side ---
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Button(left, text="Add Function", command=self._on_add_model).pack(pady=6)
        ttk.Button(left, text="Clear Model", command=self._on_clear_model).pack(pady=6)
        ttk.Button(left, text="Clip Params", command=self._on_clip_params).pack(pady=6)

        # --- main scrollable area ---
        self.scrollframe = ScrollableFrame(self)
        self.scrollframe.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self.blocks_frame = self.scrollframe.inner

        # GLOBAL wheel forwarder
        self.bind_all("<MouseWheel>", self._on_global_mousewheel)

        # Prepopulate
        if existing:
            for comp in existing:
                self._add_function_block(prefill=comp)
        if prefill:
            self._add_function_block(prefill=prefill)

    # ================================================================
    #  SCROLL HANDLER
    # ================================================================
    def _on_global_mousewheel(self, event):
        if not self.winfo_viewable():
            return

        delta = int(-event.delta / 120) if event.delta else (1 if event.num == 5 else -1)
        self.scrollframe.scroll(delta)

    # ================================================================
    #  ADD / CLEAR / DELETE
    # ================================================================
    def _on_add_model(self):
        self._add_function_block()

    def _on_clear_model(self):
        for block in list(self.function_blocks):
            block.frame.destroy()
        self.function_blocks.clear()
        if callable(self.clear_callback):
            self.clear_callback()

    def _add_function_block(self, prefill=None):
        """Create new block with stable unique index."""
        if prefill and "index" in prefill:
            idx = prefill["index"]
        else:
            existing = [blk.index for blk in self.function_blocks]
            idx = max(existing, default=0) + 1

        block = FunctionBlock(
            master=self.blocks_frame,
            index=idx,
            model_functions=self.model_functions,
            span_request_callback=self.span_request_callback,
            save_callback=self.save_callback,
            remove_callback=self._remove_block,
        )
        block.frame.pack(fill="x", pady=5, padx=5)
        self.function_blocks.append(block)

        if prefill:
            block.func_var.set(prefill["model_name"])
            block.build_fields()
            block.prefill_from_data(prefill)

    def _remove_block(self, block):
        block.frame.destroy()
        self.function_blocks.remove(block)
        self.save_callback(block.index, None)

    # ================================================================
    #  CLIP PARAMS
    # ================================================================
    def _on_clip_params(self):
        values = []
        for block in self.function_blocks:
            for pname, w in block.entries.items():
                txt = w["val"].get().strip()
                values.append(txt if txt else "")
        out = "\t".join(values)
        self.clipboard_clear()
        self.clipboard_append(out)
        self.update()

# ================================================================
#  FUNCTION BLOCK
# ================================================================
class FunctionBlock:
    """Panel for one function."""
    def __init__(self, master, index, model_functions,
                 span_request_callback, save_callback, remove_callback):

        self.master = master
        self.index = index
        self.model_functions = model_functions
        self.span_request_callback = span_request_callback
        self.save_callback = save_callback
        self.remove_callback = remove_callback
        self.entries = {}

        self.frame = ttk.LabelFrame(master, text=f"Function {index}")
        self.frame.columnconfigure(1, weight=1)

        # Function dropdown
        ttk.Label(self.frame, text="Function type:").grid(row=0, column=0, sticky="w")
        self.func_var = tk.StringVar()
        self.func_menu = tk.OptionMenu(self.frame, self.func_var,
                                       *list(model_functions.keys()),
                                       command=lambda _: self.build_fields())
        self.func_menu.grid(row=0, column=1, sticky="ew")

        # Params container
        self.params_frame = ttk.Frame(self.frame)
        self.params_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=5)

        # Buttons
        btns = ttk.Frame(self.frame)
        btns.grid(row=2, column=0, columnspan=5, pady=5)
        self.guess_btn = ttk.Button(btns, text="Guess",
                                    command=self.on_guess, state="disabled")
        self.save_btn  = ttk.Button(btns, text="Save",
                                    command=self.on_save, state="disabled")
        self.del_btn   = ttk.Button(btns, text="Delete",
                                    command=self.on_delete, state="disabled")
        self.guess_btn.grid(row=0, column=0, padx=4)
        self.save_btn.grid (row=0, column=1, padx=4)
        self.del_btn.grid  (row=0, column=2, padx=4)

    # ================================================================
    #  BUILD FIELDS
    # ================================================================
    def build_fields(self):
        for child in self.params_frame.winfo_children():
            child.destroy()
        self.entries.clear()

        # Label field
        ttk.Label(self.params_frame, text="Label:").grid(row=0, column=0, sticky="w")
        self.label_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.label_var)\
            .grid(row=0, column=1, columnspan=4, sticky="ew")

        func = self.func_var.get()
        if func not in self.model_functions:
            return

        # Headers
        ttk.Label(self.params_frame, text="Value").grid(row=1, column=1)
        ttk.Label(self.params_frame, text="Min").grid(row=1, column=2)
        ttk.Label(self.params_frame, text="Max").grid(row=1, column=3)

        for row, pname in enumerate(self.model_functions[func]["params"], start=2):
            ttk.Label(self.params_frame, text=pname).grid(row=row, column=0, sticky="w")

            e_val = ttk.Entry(self.params_frame, width=12); e_val.grid(row=row, column=1)
            e_min = ttk.Entry(self.params_frame, width=12); e_min.grid(row=row, column=2)
            e_max = ttk.Entry(self.params_frame, width=12); e_max.grid(row=row, column=3)

            self.entries[pname] = {"val": e_val, "min": e_min, "max": e_max}

        for b in (self.guess_btn, self.save_btn, self.del_btn):
            b.config(state="normal")

    # ================================================================
    #  SPAN SELECTION (GUESS)
    # ================================================================
    def on_guess(self):
        top = self.frame.winfo_toplevel()
        top.withdraw()
        self.span_request_callback(self, self.index, self.func_var.get())

    def receive_span_selection(self, xmin, xmax, x_data, y_data, residual):
        top = self.frame.winfo_toplevel()
        top.deiconify()
        top.lift()
        top.focus_force()

        if len(x_data) == 0:
            messagebox.showwarning("Selection Error", "No data points in range.")
            return

        if not self.entries:
            self.build_fields()

        self.last_span = (xmin, xmax)
        func = self.func_var.get()
        guesses = {}

        # ------- original guessing preserved -------
        if func == "Linear":
            slope = (y_data[-1]-y_data[0])/(x_data[-1]-x_data[0]) if x_data[-1]!=x_data[0] else 0
            intercept = y_data[0] - slope*x_data[0]
            guesses = {"slope": slope, "intercept": intercept}

        elif func == "Sigmoid":
            amplitude = abs(y_data[-1]-y_data[0])
            center = x_data[-1] - (x_data[-1]-x_data[0])/2
            guesses = {"amplitude": amplitude, "center": center,
                       "steepnes": 0.5, "baseline": np.min(y_data)}

        elif func in ("Lorentzian","Gaussian"):
            intensity = abs(np.trapz(residual, x_data))
            pos = np.argmax(residual)
            center = x_data[pos]
            fwhm = abs((x_data[-1]-x_data[0])/4)
            guesses = {"intensity": intensity, "center": center, "fwhm": fwhm}

        elif func == "Voigt":
            intensity = abs(np.trapz(residual, x_data))
            pos = np.argmax(residual)
            center = x_data[pos]
            sigma = abs((x_data[-1]-x_data[0])/8)
            guesses = {"intensity": intensity, "center": center,
                       "gamma": sigma, "sigma": sigma}

        elif func == "Fano":
            center = x_data[-1]-(x_data[-1]-x_data[0])/2
            fwhm = (x_data[-1]-x_data[0])/4
            guesses = {"intensity": max(residual),
                       "center": center, "fwhm": fwhm, "1/q": 0}

        elif func == "Asym_Lorentzian":
            intensity = abs(np.trapz(residual, x_data))
            pos = np.argmax(residual)
            center = x_data[pos]
            fwhm = abs((x_data[-1]-x_data[0])/4)
            y_half = y_data[pos] - ((y_data[pos]-residual[pos])/2)

            fL = interp1d(y_data[:pos+1], x_data[:pos+1], assume_sorted=False)
            fR = interp1d(y_data[pos:],   x_data[pos:],   assume_sorted=False)

            xL = fL(y_half)
            xR = fR(y_half)
            deltaL = center - xL
            deltaR = xR - center
            alpha = 2*(deltaR-deltaL)/(fwhm**2)
            guesses = {"intensity": intensity, "center": center,
                       "fwhm": fwhm, "alpha": alpha}

        elif func == "Double Lorentz":
            m = len(x_data)//2
            a1 = np.trapz(residual[:m], x_data[:m])
            a2 = np.trapz(residual[m:], x_data[m:])
            c1 = x_data[0] + (x_data[-1]-x_data[0])/4
            c2 = x_data[-1] - (x_data[-1]-x_data[0])/4
            fwhm = (x_data[-1]-x_data[0])/4
            guesses = {"intensity #1": a1, "center #1": c1,
                       "intensity #2": a2, "center #2": c2,
                       "fwhm": fwhm}

        # --- insert guesses + reasonable bounds ---
        for pname, w in self.entries.items():
            if pname in guesses:
                w["val"].delete(0, "end")
                w["val"].insert(0, f"{guesses[pname]:.4g}")

            # Bounds logic (simple)
            if pname in ("intensity","intensity #1","intensity #2","fwhm"):
                w["min"].delete(0, "end"); w["min"].insert(0, "0")
            elif pname in ("center","center #1","center #2"):
                w["min"].delete(0, "end"); w["min"].insert(0, xmin)
                w["max"].delete(0, "end"); w["max"].insert(0, xmax)

    # ================================================================
    #  SAVE
    # ================================================================
    def on_save(self):
        func = self.func_var.get().strip() or "Function"

        label = self.label_var.get().strip()
        if not label:
            label = f"{func} #{self.index}"
            self.label_var.set(label)

        pnames = model_dict[func]["params"]
        params = {}
        bounds = {}

        for pname in pnames:
            # Values
            txt = self.entries[pname]["val"].get().strip()
            try: val = float(txt)
            except: val = 0.0
            params[pname] = val

            # Bounds
            mn = self.entries[pname]["min"].get().strip()
            mx = self.entries[pname]["max"].get().strip()

            try: lb = float(mn) if mn else (0.0 if pname in ("intensity","fwhm") else -np.inf)
            except: lb = -np.inf

            try: ub = float(mx) if mx else np.inf
            except: ub = np.inf

            bounds[pname] = (lb, ub)

        comp = {
            "model_name": func,
            "label": label,
            "params": params,
            "bounds": bounds
        }

        print("🔖 Saving component:", comp)
        self.save_callback(self.index, comp)

    # ================================================================
    #  DELETE
    # ================================================================
    def on_delete(self):
        self.remove_callback(self)
        self.save_callback(self.index, None)

    # ================================================================
    #  PREFILL
    # ================================================================
    def prefill_from_data(self, data):
        if not self.entries:
            self.build_fields()

        self.label_var.set(data.get("label",""))

        for pname, val in data["params"].items():
            if pname in self.entries:
                w = self.entries[pname]["val"]
                w.delete(0, "end")
                w.insert(0, str(val))

        for pname, (lo, hi) in data["bounds"].items():
            if pname in self.entries:
                self.entries[pname]["min"].delete(0,"end")
                self.entries[pname]["max"].delete(0,"end")

                self.entries[pname]["min"].insert(0, "" if lo == -np.inf else str(lo))
                self.entries[pname]["max"].insert(0, "" if hi == np.inf else str(hi))