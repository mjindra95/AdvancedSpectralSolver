# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:14:04 2025

@author: marti
"""

# model_builder.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.interpolate import interp1d
from ASS.functions import model_dict  # your functions.py with model_dict

class ScrollableFrame(ttk.Frame):
    """A scrollable frame that can contain many child widgets."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # 1) Create the canvas + vertical scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.vsb    = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        # 2) Create the interior frame (this is where you .pack your blocks)
        self.inner = ttk.Frame(self.canvas)
        #   Keep a handle on the window id so we can resize it later:
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # 3) Whenever the interior frame changes size, update scrollregion
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # 4) Whenever the canvas itself changes width, reset the inner frame’s width
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._window_id, width=e.width)
        )

        # 5) (Optional) Mouse‐wheel support when hovering over the inner frame
        # self.inner.bind(
        #     "<Enter>",
        #     lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # )
        # self.inner.bind(
        #     "<Leave>",
        #     lambda e: self.canvas.unbind_all("<MouseWheel>")
        # )
        
        # Bind mouse wheel directly to the inner frame
        self.inner.bind("<MouseWheel>", self._on_mousewheel)
        
        # And also bind it to the canvas (in case pointer is between widgets)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        # Windows / Mac / Linux delta normalization
        delta = int(-1*(event.delta/120)) if event.delta else event.num == 5 and 1 or -1
        self.canvas.yview_scroll(delta, "units")



class ModelBuilderWindow(tk.Toplevel):
    """
    Toplevel window for building a composite model.
    """
    def __init__(self, master, span_request_callback, save_callback, clear_callback, existing=None, prefill=None):
        """
        :param master: parent Tk window
        :param span_request_callback: function(block, index, model_name) to call when Guess
        :param existing: list of dicts {'index', 'model_name', 'params', 'bounds'} to prepopulate
        :param prefill: one dict same structure to prefill a single new block after span selection
        """
        super().__init__(master)
        self.title("Model Builder")
        self.geometry("400x600")
        self.span_request_callback = span_request_callback
        self.save_callback = save_callback
        self.clear_callback = clear_callback
        self.model_functions = model_dict
        self.function_blocks = []
        # self.model_state = True
        # self.spectrum_title = None
        
        # Left: Add Model button
        left_frame = ttk.Frame(self)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        add_btn = ttk.Button(left_frame, text="Add Function", command=self._on_add_model)
        add_btn.pack(pady=10)
        clear_btn = ttk.Button(left_frame, text="Clear Model", command=self._on_clear_model)
        clear_btn.pack(pady=10)
        clip_btn = ttk.Button(left_frame, text="Clip Params", command=self._on_clip_params)
        clip_btn.pack(pady=10)

        # Center/right: scrollable area for blocks
        container = ScrollableFrame(self)
        container.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.blocks_frame = container.inner

        # Prepopulate existing blocks
        if existing:
            for comp in existing:
                self._add_function_block(prefill=comp)
        # If there's a single prefill without existing
        if prefill:
            self._add_function_block(prefill=prefill)

    def _on_add_model(self):
        """Called when the Add Model button is clicked."""
        self._add_function_block(prefill=None)
        
    def _on_clear_model(self):
        # messagebox.showinfo("Clear model", "This will delete all of the functions defined functions")
        """Clear all blocks here *and* tell the main window to drop its components."""
        # 1) destroy builder panels:
        for block in list(self.function_blocks):
            block.frame.destroy()
        self.function_blocks.clear()

        # 2) notify the main GUI to clear its model too:
        if callable(self.clear_callback):
            self.clear_callback()

    def _add_function_block(self, prefill=None):
        """Create and pack a new FunctionBlock, optionally prefilled."""
        # use the incoming index if this is an existing/prefilled block
        if prefill and "index" in prefill:
            idx = prefill["index"]
        else:
            idx = len(self.function_blocks) + 1
    
        block = FunctionBlock(
            master=self.blocks_frame,
            index=idx,
            model_functions=self.model_functions,
            span_request_callback=self.span_request_callback,
            save_callback=self.save_callback, 
            remove_callback=self._remove_block
        )
        block.frame.pack(fill="x", pady=5, padx=5)
        self.function_blocks.append(block)
    
        # now actually fill it if we were asked to prefill
        if prefill and prefill.get("index") == idx:
            block.func_var.set(prefill["model_name"])
            block.build_fields()
            block.label_var.set(prefill.get("label", ""))
            block.prefill_from_data(prefill)    # make sure this method exists

    def _remove_block(self, block):
        """Remove a FunctionBlock and renumber remaining."""
        block.frame.destroy()
        self.function_blocks.remove(block)
        
        for i, blk in enumerate(self.function_blocks, start=1):
            # if the user never set a custom label, update it:
            if not blk.label_var.get().strip():
                default = f"{blk.func_var.get()}_{i}"
                blk.label_var.set(default)
    
    # def _remove_block(self, block):
    #     """Remove a FunctionBlock and renumber remaining ones."""
    #     removed_index = block.index
        
    #     # 1) Remove this block’s frame and reference
    #     block.frame.destroy()
    #     if block in self.function_blocks:
    #         self.function_blocks.remove(block)
            
    #     # 2) Tell the main model to delete that component
    #     if callable(self.save_callback):
    #         self.save_callback(removed_index, None)
            
    #     # 3) Only renumber blocks BELOW the deleted one
    #     for blk in self.function_blocks:
    #         if blk.index > removed_index:
    #             blk.index -= 1
    #             blk.frame.config(text=f"Function {blk.index}")
    
    #             func_name = blk.func_var.get().strip() or "Function"
    #             current_label = blk.label_var.get().strip()
    #             auto_prefix = f"{func_name} #"
    
    #             # If label was auto-generated → update it to new index
    #             if not current_label or current_label.startswith(auto_prefix):
    #                 new_label = f"{func_name} #{blk.index}"
    #                 blk.label_var.set(new_label)
    
    #             # Save updated block to model
    #             if callable(self.save_callback):
    #                 comp = {
    #                     "model_name": func_name,
    #                     "label": blk.label_var.get(),
    #                     "params": {},
    #                     "bounds": {},
    #                 }
    #                 # Re-use existing entries without rebuilding everything
    #                 for pname, wdict in blk.entries.items():
    #                     val = wdict["val"].get().strip()
    #                     mn  = wdict["min"].get().strip()
    #                     mx  = wdict["max"].get().strip()
    #                     lock = wdict["lock"].get()
    
    #                     try:
    #                         val = float(val)
    #                     except ValueError:
    #                         val = 0.0
    #                     comp["params"][pname] = val
    
    #                     # basic bounds reconstruction
    #                     try:
    #                         lb = float(mn) if mn else -np.inf
    #                     except ValueError:
    #                         lb = -np.inf
    #                     try:
    #                         ub = float(mx) if mx else np.inf
    #                     except ValueError:
    #                         ub = np.inf
    #                     comp["bounds"][pname] = (lb, ub)
    
    #                 self.save_callback(blk.index, comp)
    
        # # 2) Renumber the remaining blocks
        # for i, blk in enumerate(self.function_blocks, start=1):
        #     blk.index = i
        #     blk.frame.config(text=f"Function {i}")
    
        #     # Update automatic label only if user hasn’t set a custom one
        #     current_label = blk.label_var.get().strip()
        #     func_name = blk.func_var.get().strip() or "Function"
        #     auto_label = f"{func_name} #{i}"
    
        #     # Only overwrite if the label matches the old pattern or was empty
        #     if not current_label or current_label.startswith(func_name):
        #         blk.label_var.set(auto_label)
                
        #     blk.on_save()
    
        # # 3) Notify the main GUI that the model changed
        # if callable(self.clear_callback):
        #     self.clear_callback()

                
    def _on_state(self):
        if self.model_state == True:
            self.model_state == False
        else:
            self.model_state == True
            
    def _on_clip_params(self):
        """Copy all model parameter values (only numbers) to clipboard in a single line."""
        all_values = []
    
        # Iterate through all function blocks
        for block in self.function_blocks:
            # Get only parameter values (skip min/max/lock)
            for pname, wdict in block.entries.items():
                val_str = wdict["val"].get().strip()
                if val_str:
                    all_values.append(val_str)
                else:
                    all_values.append("")  # preserve column count if empty
    
        # Join as tab-delimited row
        clip_text = "\t".join(all_values)
    
        # Copy to clipboard
        self.clipboard_clear()
        self.clipboard_append(clip_text)
        self.update()  # keep it in clipboard after window closes
    
        # messagebox.showinfo("Copied", f"Copied {len(all_values)} parameter values to clipboard.")


class FunctionBlock:
    """
    Represents one function component: dropdown + parameter fields + action buttons.
    """
    def __init__(self, master, index, model_functions, span_request_callback, save_callback, remove_callback):
        """
        :param master: parent frame
        :param index: 1-based index of this block
        :param model_functions: dict model_name -> {'func', 'params'}
        :param span_request_callback: function(block, index, model_name)
        :param remove_callback: function(block) to call on Delete
        """
        self.master = master
        self.index = index
        self.model_functions = model_functions
        self.span_request_callback = span_request_callback
        self.save_callback = save_callback
        self.remove_callback = remove_callback

        # Outer frame
        self.frame = ttk.LabelFrame(master, text=f"Function {index}")
        self.frame.columnconfigure(1, weight=1)

        # Dropdown selection
        ttk.Label(self.frame, text="Function type:").grid(row=0, column=0, sticky="w")
        self.func_var = tk.StringVar(value="")
        self.func_menu = tk.OptionMenu(
            self.frame, self.func_var,
            *list(model_functions.keys()),
            command=lambda _: self.build_fields()
        )
        self.func_menu.grid(row=0, column=1, sticky="ew")

        # Container for parameter entries
        self.params_frame = ttk.Frame(self.frame)
        self.params_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=5)
        self.entries = {}  # pname -> {'val', 'min', 'max', 'lock'}

        # Action buttons (disabled until fields built)
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=2, column=0, columnspan=5, pady=5)
        self.guess_btn  = ttk.Button(btn_frame, text="Guess",  command=self.on_guess,  state="disabled")
        self.save_btn   = ttk.Button(btn_frame, text="Save",   command=self.on_save,   state="disabled")
        self.delete_btn = ttk.Button(btn_frame, text="Delete", command=self.on_delete, state="disabled")
        self.guess_btn.grid (row=0, column=0, padx=3)
        self.save_btn.grid  (row=0, column=1, padx=3)
        self.delete_btn.grid(row=0, column=2, padx=3)

    def build_fields(self):
        """Populate parameter rows based on selected function."""
        # Clear previous
        for child in self.params_frame.winfo_children():
            child.destroy()
        self.entries.clear()
        
        # New: label row
        ttk.Label(self.params_frame, text="Label:").grid(row=0, column=0, sticky="w", pady=2)
        self.label_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.label_var).grid(row=0, column=1, columnspan=4, sticky="ew")

        func_name = self.func_var.get()
        if func_name not in self.model_functions:
            return
        
        #ttk.Label(self.params_frame, text="Parameter").grid(row=1, column=0, sticky="w", pady=(5,2))
        ttk.Label(self.params_frame, text="Value").grid(row=1, column=1, sticky="w", pady=(5,2))
        ttk.Label(self.params_frame, text="Min").grid(row=1, column=2, sticky="w", pady=(5,2))
        ttk.Label(self.params_frame, text="Max").grid(row=1, column=3, sticky="w", pady=(5,2))
        ttk.Label(self.params_frame, text="Lock").grid(row=1, column=4, sticky="w", pady=(5,2))

        params = self.model_functions[func_name]["params"]
        for row, pname in enumerate(params, start=2):
            ttk.Label(self.params_frame, text=pname).grid(row=row+1, column=0, sticky="w", pady=2)
            val = ttk.Entry(self.params_frame, width=8); val.grid(row=row+1, column=1, padx=2)
            mn  = ttk.Entry(self.params_frame, width=6); mn .grid(row=row+1, column=2, padx=2)
            mx  = ttk.Entry(self.params_frame, width=6); mx .grid(row=row+1, column=3, padx=2)
            lock_var = tk.BooleanVar()
            #chk = ttk.Checkbutton(self.params_frame, text="Lock", variable=lock_var)
            #chk = ttk.Checkbutton(self.params_frame, text="Lock", variable=lock_var, 
            #                      command=lambda pn=pname, lv=lock_var: self._toggle_lock(pn, lv))
            chk = ttk.Checkbutton(self.params_frame, variable=lock_var, 
                                  command=lambda pn=pname, lv=lock_var: self._toggle_lock(pn, lv))
            chk.grid(row=row+1, column=4, padx=2)
            self.entries[pname] = {"val": val, "min": mn, "max": mx, "lock": lock_var}
            #self.entries[pname] = {"val": val, "min": mn, "max": mx}
            
        # Enable action buttons
        for b in (self.guess_btn, self.save_btn, self.delete_btn):
            b.config(state="normal")
    
    def _toggle_lock(self, pname, lock_var):
        st = "disabled" if lock_var.get() else "normal"
        w = self.entries[pname]
        w["val"].config(state=st)
        w["min"].config(state=st)
        w["max"].config(state=st)

    def on_guess(self):
        """Hide the ModelBuilderWindow and invoke span selector."""
        # get the toplevel window owning this frame:
        toplevel = self.frame.winfo_toplevel()
        toplevel.withdraw()
    
        # now ask the main window to start its SpanSelector,
        # passing this block and its index/name
        self.span_request_callback(self, self.index, self.func_var.get())


    def receive_span_selection(self, xmin, xmax, x_data, y_data, residual):
        """Called by MainWindow with the selected span and data slice."""
        # Restore builder
        # self.frame.master.master.deiconify()
        toplevel = self.frame.winfo_toplevel()
        toplevel.deiconify()
        toplevel.lift()
        toplevel.focus_force()
        
        # --- NEW GUARD ---
        if len(x_data) == 0:
            messagebox.showwarning(
                "Selection Error",
                "No data points in that selected range.  "
                "Please drag a region that actually overlaps the spectrum."
            )
            return

        # Ensure fields exist
        if not self.entries:
            self.build_fields()
            
        self.last_span = (xmin, xmax)

        func_name = self.func_var.get()
        guesses = {}


        if func_name == "Linear":
            slope = (y_data[-1]-y_data[0])/(x_data[-1]-x_data[0]) if x_data[-1]!=x_data[0] else 0.0
            intercept = y_data[0] - slope*x_data[0]
            guesses = {"slope": slope, 
                       "intercept": intercept}
            
        elif func_name == "Sigmoid":
            if y_data[0] > y_data[-1]:
                amplitude = y_data[0]-y_data[-1]
            else:
                amplitude = y_data[-1]-y_data[0]
            center = x_data[-1]-(x_data[-1]-x_data[0])/2
            steepnes = 0.5
            baseline = np.min(y_data)
            guesses = {"amplitude" : amplitude,
                       "center" : center,
                       "steepnes" : steepnes,
                       "baseline" : baseline}
            
        elif func_name in ("Lorentzian", "Gaussian"):
            intensity = abs(np.trapz(residual, x_data))
            position_index = np.argmax(residual)
            center = x_data[position_index]
            fwhm = abs((x_data[-1]-x_data[0])/4)
            guesses = {"intensity" : intensity,
                       "center" : center,
                       "fwhm" : fwhm}
            
        elif func_name == "Voigt":
            intensity = abs(np.trapz(residual, x_data))
            position_index = np.argmax(residual)
            center = x_data[position_index]
            sigma = abs((x_data[-1]-x_data[0])/8)
            gamma = sigma
            guesses = {"intensity" : intensity,
                       "center" : center,
                       "gamma" : gamma,
                       "sigma" : sigma}
            
        elif func_name in "Fano":
            # intensity = np.trapz(residual, x_data)
            center = x_data[-1]-(x_data[-1]-x_data[0])/2
            fwhm = (x_data[-1]-x_data[0])/4
            Q = 0
            # intensity = max(residual)/(q**2)
            intensity = max(residual)
            guesses = {"intensity" : intensity,
                       "center" : center,
                       "fwhm" : fwhm,
                       "1/q" : Q}
            
        elif func_name in "Asym_Lorentzian":
            intensity = abs(np.trapz(residual, x_data))
            position_index = np.argmax(residual)
            center = x_data[position_index]
            fwhm = abs((x_data[-1]-x_data[0])/4)
            y_half = y_data[position_index] - ((y_data[position_index] - residual[position_index]) / 2)
            fL = interp1d(y_data[:position_index+1], x_data[:position_index+1], assume_sorted=False)
            fR = interp1d(y_data[position_index:],   x_data[position_index:],   assume_sorted=False)
            xL = fL(y_half)
            xR = fR(y_half) 
            deltaL = center - xL
            deltaR = xR - center
            alpha = 2 * (deltaR - deltaL) / (fwhm ** 2)
            guesses = {"intensity" : intensity,
                       "center" : center,
                       "fwhm" : fwhm,
                       "alpha" : alpha}
            
        elif func_name in "Double Lorentz":
            selected_range = len(x_data)
            mid = selected_range // 2
            
            first_half = x_data[:mid]
            first_half_data = residual[:mid]
            
            second_half = x_data[mid:]
            second_half_data = residual[mid:]
            
            area_first = np.trapz(first_half_data, first_half)
            area_second = np.trapz(second_half_data, second_half)
            
            center_fisrt = x_data[0]+(x_data[-1]-x_data[0])/4
            center_second = x_data[-1]-(x_data[-1]-x_data[0])/4
            
            fwhm = (x_data[-1]-x_data[0])/4
            
            guesses = {"intensity #1" : area_first,
                       "center #1" : center_fisrt,
                       "intensity #2" : area_second,
                       "center #2" : center_second,
                       "fwhm" : fwhm}
        else:
            messagebox.showinfo("This is not defined function")
        
        # Now fill in both the guessed values AND the bounds
        for pname, widget_dict in self.entries.items():
            # 1) Fill the guessed value into the “val” entry
            if pname in guesses:
                val = guesses[pname]
                e_val = widget_dict["val"]
                e_val.delete(0, "end")
                e_val.insert(0, f"{val:.4g}")
        
            # 2) Determine lower/upper bounds based on pname, span, and lock
            #    If “lock” is ticked, set both bounds to the guessed value.
            is_locked = widget_dict["lock"].get()
            lower_str = ""
            upper_str = ""
            if is_locked:
                # Both bounds = guessed → force that parameter to stay fixed
                if pname in guesses:
                    # lower_str = upper_str = f"{guesses[pname]:.4g}"
                    lower_str = guesses[pname]*0.99
                    upper_str = guesses[pname]*1.01
            else:
                if pname == "intensity" or pname == "intensity #1" or pname == "intensity #2":
                    # intensity ≥ 0
                    lower_str = "0"
                    # leave upper blank (None)
                    upper_str = ""
                elif pname == "fwhm":
                    # fwhm ≥ 0
                    lower_str = "0"
                    upper_str = ""
                elif pname == "center" or pname == "center #1" or pname == "center #2":
                    # center ∈ [xmin, xmax]
                    lower_str = f"{xmin:.4g}"
                    upper_str = f"{xmax:.4g}"
                else:
                    # For sigma, gamma, alpha, q, steepness, baseline, etc.—
                    # leave both bounds blank so user can adjust manually
                    lower_str = ""
                    upper_str = ""
        
            # 3) Insert those bounds into min and max entries
            e_min = widget_dict["min"]
            e_max = widget_dict["max"]
        
            e_min.delete(0, "end")
            e_min.insert(0, lower_str)
        
            e_max.delete(0, "end")
            e_max.insert(0, upper_str)
                
    def prefill_from_data(self, data):
        """
        :param data: dict with keys "params" and "bounds"
        """
        # make sure the fields exist
        if not self.entries:
            self.build_fields()
            
        if "label" in data:
            self.label_var.set(data["label"])

        # fill in the saved values
        for pname, val in data["params"].items():
            w = self.entries.get(pname)
            if w:
                w["val"].delete(0, "end")
                w["val"].insert(0, str(val))

        for pname, (lo, hi) in data["bounds"].items():
            w = self.entries.get(pname)
            if w:
                # use your min/max keys
                w["min"].delete(0, "end")
                w["min"].insert(0, "" if lo == -np.inf else str(lo))
                w["max"].delete(0, "end")
                w["max"].insert(0, "" if hi == np.inf else str(hi))
                
        for pname, locked in data.get("locks", {}).items():
            wdict = self.entries.get(pname)
            if wdict:
                wdict["lock"].set(locked)
    
    # def on_save(self):
    #     func_name = self.func_var.get()
        
    #     label = self.label_var.get().strip() or func_name
    #     pnames    = model_dict[func_name]["params"]
    
    #     params = {}
    #     bounds = {}
    
    #     for pname in pnames:
    #         # 1) Value entry (always required)
    #         txt_val = self.entries[pname]["val"].get().strip()
    #         try:
    #             val = float(txt_val) if txt_val else 0.0
    #         except ValueError:
    #             val = 0.0
    #         params[pname] = val
    
    #         # 2) Lock checkbox
    #         is_locked = self.entries[pname]["lock"].get()
    
    #         # 3) Read whatever is in the “min” / “max” entries
    #         mn_txt = self.entries[pname]["min"].get().strip()
    #         mx_txt = self.entries[pname]["max"].get().strip()
    
    #         if is_locked:
    #             # If locked, force both bounds = the chosen value
    #             # lb = ub = val
    #             if pname == "center":
    #                 lb = val - 0.1
    #                 ub = val + 0.1
    #             elif pname == "fwhm":
    #                 lb = val - 0.01
    #                 ub = val + 0.01
    #             elif pname == "intensity":
    #                 lb = val*0.99
    #                 ub = val*1.01
    #             else:
    #                 lb = val - 0.1
    #                 ub = val + 0.1

    #         else:
    #             # Not locked → pick user‐typed bounds if present,
    #             # otherwise use our parameter‐specific defaults.
    
    #             # LOWER
    #             if mn_txt:
    #                 try:
    #                     lb = float(mn_txt)
    #                 except ValueError:
    #                     lb = -np.inf
    #             else:
    #                 # no user‐entered lower bound → default by parameter:
    #                 if pname in ("intensity", "fwhm"):
    #                     lb = 0.0
    #                 elif pname == "center":
    #                     # assume self.last_span exists (xmin, xmax from receive_span_selection)
    #                     xmin, xmax = self.last_span
    #                     lb = xmin
    #                 else:
    #                     lb = -np.inf
    
    #             # UPPER
    #             if mx_txt:
    #                 try:
    #                     ub = float(mx_txt)
    #                 except ValueError:
    #                     ub = np.inf
    #             else:
    #                 # no user‐entered upper bound → default by parameter:
    #                 if pname in ("intensity", "fwhm"):
    #                     ub = np.inf
    #                 elif pname == "center":
    #                     xmin, xmax = self.last_span
    #                     ub = xmax
    #                 else:
    #                     ub = np.inf
    
    #         bounds[pname] = (lb, ub)
    
    #     comp = {
    #       "model_name": func_name,
    #       "label":      label,
    #       "params":     params,
    #       "bounds":     bounds
    #     }
    
    #     print("🔖 Saving component:", comp)   # for debugging
    #     self.save_callback(self.index, comp)

    def on_save(self):
        func_name = self.func_var.get().strip() or "Function"
        
        # If user left the label empty → assign automatic name
        label = self.label_var.get().strip()
        if not label:
            label = f"{func_name} #{self.index}"
            self.label_var.set(label)  # reflect it in the UI too
    
        pnames = model_dict[func_name]["params"]
    
        params = {}
        bounds = {}
    
        for pname in pnames:
            # --- Value ---
            txt_val = self.entries[pname]["val"].get().strip()
            try:
                val = float(txt_val) if txt_val else 0.0
            except ValueError:
                val = 0.0
            params[pname] = val
    
            # --- Lock ---
            is_locked = self.entries[pname]["lock"].get()
    
            # --- Bounds ---
            mn_txt = self.entries[pname]["min"].get().strip()
            mx_txt = self.entries[pname]["max"].get().strip()
    
            if is_locked:
                if pname == "center":
                    lb, ub = val - 0.1, val + 0.1
                elif pname == "fwhm":
                    lb, ub = val - 0.01, val + 0.01
                elif pname == "intensity":
                    lb, ub = val * 0.99, val * 1.01
                else:
                    lb, ub = val - 0.1, val + 0.1
            else:
                # Lower bound
                if mn_txt:
                    try:
                        lb = float(mn_txt)
                    except ValueError:
                        lb = -np.inf
                else:
                    if pname in ("intensity", "fwhm"):
                        lb = 0.0
                    elif pname == "center" and hasattr(self, "last_span"):
                        xmin, xmax = self.last_span
                        lb = xmin
                    else:
                        lb = -np.inf
    
                # Upper bound
                if mx_txt:
                    try:
                        ub = float(mx_txt)
                    except ValueError:
                        ub = np.inf
                else:
                    if pname in ("intensity", "fwhm"):
                        ub = np.inf
                    elif pname == "center" and hasattr(self, "last_span"):
                        xmin, xmax = self.last_span
                        ub = xmax
                    else:
                        ub = np.inf
    
            bounds[pname] = (lb, ub)
    
        comp = {
            "model_name": func_name,
            "label": label,
            "params": params,
            "bounds": bounds
        }
    
        print("🔖 Saving component:", comp)
        self.save_callback(self.index, comp)

    def on_delete(self):
        """Remove this block from the builder."""
        # 1) remove the block’s UI
        self.remove_callback(self)
        # 2) notify main window to delete that index
        self.save_callback(self.index, None)
