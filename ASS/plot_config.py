# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:54:29 2025

@author: marti
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json

class PlotConfigWindow(tk.Toplevel):
    def __init__(self, master, config_path, update_callback):
        super().__init__(master)
        self.title("Plot Config")
        self.config_path = config_path
        self.update_callback = update_callback

        # Load current config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Normalize string "None" -> None
        for k, v in self.config.items():
            if isinstance(v, str) and v.strip().lower() == "none":
                self.config[k] = None

        # Create input fields
        self.entries = {}
        for i, (key, value) in enumerate(self.config.items()):
            ttk.Label(self, text=key).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(self)

            if value is None:
                entry.insert(0, "")
            elif isinstance(value, (list, dict)):
                entry.insert(0, json.dumps(value))
            else:
                entry.insert(0, str(value))

            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[key] = entry

        ttk.Button(self, text="Apply", command=self.save_config).grid(
            row=len(self.config), column=0, columnspan=2, pady=10
        )

    def save_config(self):
        for key, entry in self.entries.items():
            value = entry.get().strip()

            if value == "":
                self.config[key] = None
            else:
                # Try to parse JSON-like input (lists, dicts, numbers)
                try:
                    parsed = json.loads(value)
                    self.config[key] = parsed
                except Exception:
                    # Leave as plain string
                    self.config[key] = value

        # Save back to JSON (real None -> null)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
            
        self.update_callback()

        messagebox.showinfo("Success", "Configuration updated!")
        self.destroy()
