# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:50:11 2025

@author: Martin Jindra
"""

from tkinter import filedialog
import os, sys

class File_utils:
    
    def __init__(self, example=None):
        self.example = example

    @staticmethod
    def ask_spectrum_file(title="Select spectrum file"):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
    
    @staticmethod
    def ask_directory(title="Select directory with txt files"):
        return filedialog.askdirectory(title=title)

    @staticmethod    
    def resource_path(relative_path):
        """Get absolute path to bundled resources (icons, etc.)."""
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    @staticmethod
    def user_config_path(filename: str) -> str:
        """Get path to user config file in AppData/ASS folder."""
        base_dir = os.path.join(os.getenv("APPDATA", os.path.expanduser("~")), "ASS")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, filename)



    
