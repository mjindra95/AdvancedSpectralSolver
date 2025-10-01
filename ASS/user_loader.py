# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 13:55:45 2025

@author: marti
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json

class UserLoaderConfigWindow(tk.Toplevel):
    def __init__(self, master, config_path):
        super().__init__(master)
        self.title("User Config")
        self.config_path = config_path

        # Load current config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Create input fields
        self.entries = {}
        for i, (key, value) in enumerate(self.config.items()):
            ttk.Label(self, text=key).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(self)
            # use json.dumps for lists/dicts, str() otherwise
            if isinstance(value, (list, dict)):
                entry.insert(0, json.dumps(value))  # ensures double quotes for JSON
            else:
                entry.insert(0, str(value))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[key] = entry

        ttk.Button(self, text="Apply", command=self.save_config).grid(row=len(self.config), column=0, columnspan=2, pady=10)

    def save_config(self):
        # Update config dict with user inputs
        for key, entry in self.entries.items():
            value = entry.get()
            # Convert lists back from string manually if needed
            if key in ["usecols"]:
                try:
                    value = json.loads(value)  # allow [0,1] or ["X","Y"]
                except:
                    messagebox.showerror("Error", f"Invalid format for {key}")
                    return
            elif key == "skip_rows":
                value = int(value)
            self.config[key] = value

        # Save back to JSON
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

        messagebox.showinfo("Success", "Configuration updated!")
        self.destroy()
