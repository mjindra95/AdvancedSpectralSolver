# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:50:11 2025

@author: Martin Jindra
"""

from tkinter import filedialog
import os, sys

class File_utils:
    
    def __init__(self, example):
        self.example = example
    
    def ask_spectrum_file(title="Select spectrum file"):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
    
    def ask_directory(title="Select directory with txt files"):
        return filedialog.askdirectory(title = title)

    @staticmethod    
    def resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller .exe"""
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    
