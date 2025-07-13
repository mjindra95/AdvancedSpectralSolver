# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 21:41:12 2025

@author: marti
"""
import tkinter as tk
from tkinter import ttk, messagebox
import inspect

from ASS.logic import Filtering  # import the class we just wrote

class FilterWindow(tk.Toplevel):
    """
    A small modal that lets the user:
      • pick a filter from a dropdown,
      • fill in its parameter fields,
      • click “Apply” to send (func_name, params) back to main,
      • or click “Disable” to cancel filtering.
    
    The caller must provide:
      - master: the MainWindow
      - apply_callback(func_name:str, params:dict) → None
      - disable_callback() → None
    """
    
    def __init__(self, master, apply_callback, disable_callback, initial=None):
        """
        :param master: parent window
        :param apply_callback: func_name, params → None
        :param disable_callback: () → None
        :param initial: either None or a tuple (func_name, params_dict)
                        representing the currently active filter.
        """
        super().__init__(master)
        self.title("Filter Data")
        self.geometry("250x150")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
    
        self.apply_callback   = apply_callback
        self.disable_callback = disable_callback
        self.initial_filter   = initial  # store it for later use
    
        # 1) Get all filter methods from Filtering (excluding private/none)
        self.filter_funcs = {
            name: getattr(Filtering, name)
            for name, obj in inspect.getmembers(Filtering, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
    
        # Top row: dropdown for filter choice
        self.func_var = tk.StringVar(value="none")
        row1 = ttk.Frame(self)
        row1.pack(fill="x", padx=10, pady=5)
        ttk.Label(row1, text="Choose filter:").pack(side="left")
        choices = list(self.filter_funcs.keys())
        self.dropdown = ttk.Combobox(row1, textvariable=self.func_var, values=choices, state="readonly")
        self.dropdown.pack(side="left", padx=5, fill="x", expand=True)
        self.dropdown.bind("<<ComboboxSelected>>", lambda e: self.build_param_fields())
    
        # Container for parameter rows (dynamically populated)
        self.params_frame = ttk.Frame(self)
        self.params_frame.pack(fill="x", padx=10, pady=5)
    
        # Bottom row: Apply / Disable buttons
        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=10, pady=5)
        self.apply_btn   = ttk.Button(row2, text="Apply",   command=self.on_apply)
        self.disable_btn = ttk.Button(row2, text="Disable", command=self.on_disable)
        self.apply_btn.pack(side="left", expand=True, fill="x", padx=5)
        self.disable_btn.pack(side="left", expand=True, fill="x", padx=5)
    
        # 2) If initial filter exists, set dropdown and build fields
        if self.initial_filter is not None:
            func_name, params = self.initial_filter
            if func_name in self.filter_funcs:
                self.func_var.set(func_name)
        # 3) Now build the parameter entry rows (either for “none” or the initial choice)
        self.param_entries = {}
        self.build_param_fields()
    
        # 4) If we did have an initial filter, prefill those parameters
        if self.initial_filter is not None:
            func_name, params = self.initial_filter
            if func_name in self.filter_funcs:
                # Ensure the param fields exist
                for pname, ent in self.param_entries.items():
                    if pname in params:
                        ent.delete(0, "end")
                        ent.insert(0, str(params[pname]))


    def build_param_fields(self):
        """
        Look at self.func_var.get(), inspect that Filtering method’s signature,
        and build Entry widgets for each parameter (excluding x,y).
        """
        # 1) clear any old widgets
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_entries.clear()

        fname = self.func_var.get()
        if fname not in self.filter_funcs:
            return

        sig = inspect.signature(self.filter_funcs[fname])
        params = [p for p in sig.parameters.values() if p.name not in ("x","y")]

        # 2) for each parameter, build label+entry
        for i, p in enumerate(params):
            lbl = ttk.Label(self.params_frame, text=p.name + ":")
            lbl.grid(row=i, column=0, sticky="w", pady=2)
            ent = ttk.Entry(self.params_frame, width=12)
            ent.grid(row=i, column=1, sticky="w", padx=5)
            # put default if it exists (use annotation or default)
            if p.default is not inspect._empty:
                ent.insert(0, str(p.default))
            self.param_entries[p.name] = ent

    def on_apply(self):
        """
        Gather parameter values (convert to float/int as needed),
        then call apply_callback(func_name, params_dict), and close.
        """
        fname = self.func_var.get()
        if fname not in self.filter_funcs:
            messagebox.showerror("Filter Error", "Invalid filter chosen.")
            return

        # 1) build params dict
        params = {}
        for pname, ent in self.param_entries.items():
            txt = ent.get().strip()
            if not txt:
                messagebox.showerror("Filter Error", f"Please enter a value for '{pname}'.")
                return
            try:
                if "." in txt:
                    params[pname] = float(txt)
                else:
                    params[pname] = int(txt)
            except ValueError:
                messagebox.showerror("Filter Error", f"Invalid number for '{pname}': {txt}")
                return

        # 2) call the callback with name+params, then close
        self.apply_callback(fname, params)
        # self.destroy()

    def on_disable(self):
        """User clicked Disable → cancel any filtering, then close."""
        self.disable_callback()
        # self.destroy()
