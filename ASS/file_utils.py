# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:50:11 2025

@author: Martin Jindra
"""

from tkinter import filedialog
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
    
