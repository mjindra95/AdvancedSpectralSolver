# -*- coding: utf-8 -*-
"""
Created on Sun May 11 15:19:04 2025

@author: Martin Jindra

this file contains all of he logical processes which are necesseary for Advanced Specral Solver app
"""
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
# from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .functions import model_dict
from scipy.signal import savgol_filter, medfilt
from scipy.integrate import simpson
import tkinter as tk
from tkinter import filedialog
import glob
import os
import re
from joblib import Parallel, delayed

class Loading:
    def __init__(self, example):
        self.example = example
    
    @staticmethod
    def load_horiba(file_path):
        """
        Loads Horiba .txt file and skips header lines starting with '#'.

        Parameters:
        - file_path (str): Path to the file.

        Returns:
        - pd.DataFrame with columns ['X', 'Y']
        """
        filename = os.path.basename(file_path)
        encodings_to_try = ['ansi', 'utf-8', 'utf-16', 'iso-8859-1', 'windows-1250']
        
        for encoding in encodings_to_try:
            try:
                # Read the file line by line and count lines starting with '#'
                with open(file_path, 'r', encoding=encoding) as f:
                    skiprows = 0
                    for line in f:
                        if line.strip().startswith('#'):
                            skiprows += 1
                        else:
                            break  # stop at first non-comment line
                
                data = pd.read_csv(
                    file_path,
                    skiprows=skiprows,
                    usecols=[0, 1],
                    sep=r'\s+',
                    header=None,
                    names=['X', 'Y'],
                    encoding=encoding,
                    engine = 'python'
                )
                return data['X'].values, data['Y'].values, filename
            except Exception:
                continue
        raise ValueError(f'Could not read file with any known encoding: {encodings_to_try}')

    
    @staticmethod
    def load_witec(file_path):
        """
        Loads Witec text data (.txt) exported from Project5.

        Parameters:
        - file_path (str): Path to the .txt file.
        - skiprows (int): Number of header lines to skip. Default is 36.

        Returns:
        - pd.DataFrame with columns ['X', 'Y']
        """
        filename = os.path.basename(file_path)
        encodings_to_try = ['ansi', 'utf-8', 'utf-16', 'iso-8859-1', 'windows-1250']
        for encoding in encodings_to_try:
            try:
                data = pd.read_csv(
                    file_path,
                    skiprows=36,
                    usecols=[0, 1],
                    sep=r'\s+',
                    header=None,
                    names=['X', 'Y'],
                    encoding=encoding,
                    engine = 'python'
                )
                filename = os.path.basename(file_path)
                return data['X'].values, data['Y'].values, filename
            except Exception:
                continue
        raise ValueError(f'Could not read file with any known encoding: {encodings_to_try}')
        
    @staticmethod
    def load_default(file_path):
        """
        Loads text data (.txt) without any header, just two columns of data.

        Parameters:
        - file_path (str): Path to the .txt file.

        Returns:
        - pd.DataFrame with columns ['X', 'Y']
        """
        filename = os.path.basename(file_path)
        encodings_to_try = ['ansi', 'utf-8', 'utf-16', 'iso-8859-1', 'windows-1250']
        for encoding in encodings_to_try:
            try:
                data = pd.read_csv(
                    file_path,
                    usecols=[0, 1],
                    sep=r'\s+',
                    header=None,
                    names=['X', 'Y'],
                    encoding=encoding,
                    engine = 'python'
                )
                filename = os.path.basename(file_path)
                return data['X'].values, data['Y'].values, filename
            except Exception:
                continue
        raise ValueError(f'Could not read file with any known encoding: {encodings_to_try}')
        
    @staticmethod
    def load_witec_map(txt_path: str):
        # 1) Read header (# lines) to grab metadata and count skiprows
        metadata = {}
        skiprows = 0
        encodings_to_try = ['ansi', 'utf-8', 'utf-16', 'iso-8859-1', 'windows-1250']
        correct_encoding = None
        for encoding in encodings_to_try:
            try:
                #df = pd.read_csv(txt_path, delimiter='\t', skiprows=17, encoding=encoding)
                print(f'Trying encoding:{encoding}')
                with open(txt_path, 'r', encoding=encoding) as file:
                    for line in file:
                        skiprows += 1
                        line = line.strip()
                        if line.startswith("SizeX"):
                            metadata['SizeX'] = float(line.strip().split('=')[1].strip())
                        elif line.startswith("SizeY"):
                            metadata['SizeY'] = float(line.strip().split('=')[1].strip())
                        elif line.startswith("ScanWidth"):
                            metadata['ScanWidth'] = float(line.strip().split('=')[1].strip())
                        elif line.startswith("ScanHeight"):
                            metadata['ScanHeight'] = float(line.strip().split('=')[1].strip())
                        elif line == "[Data]":
                            # we've counted the "[Data]" line itself
                            break
                print(f"File successfully read with encoding: {encoding} and header was read")
                correct_encoding=encoding
                break
            except UnicodeDecodeError:
                print(f"Failed to read file with encoding: {encoding}")
        else:
            raise UnicodeDecodeError("Unable to read the file with the provided encodings.")

        df = pd.read_csv(
            txt_path,
            sep='\t',
            skiprows=skiprows,
            encoding=correct_encoding,
            engine='python',
            header=0,    # the very next line is the column names
            #dtype=str    # parse strings first
        )
        
        # print("Shape of loaded dataframe is:")
        # print("Metadata")
        # print(metadata)
        # print(df.shape)
        # print(df)

        return df, metadata
    
    @staticmethod
    def load_horiba_map(txt_path: str):
        # 1) Read header (# lines) to grab metadata and count skiprows
        metadata = {}
        skiprows = 0
        encodings_to_try = ['ansi', 'utf-8', 'utf-16', 'iso-8859-1', 'windows-1250']
        correct_encoding = None
        for encoding in encodings_to_try:
            try:
                horiba_df = pd.read_csv(txt_path,
                                 sep='\t',
                                 comment='#',
                                 encoding = encoding,
                                 engine = 'python')
                print(f"File successfully read with encoding: {encoding} and header was read")
                correct_encoding=encoding
                break
            except UnicodeDecodeError:
                print(f"Failed to read file with encoding: {encoding}")
        else:
            raise UnicodeDecodeError("Unable to read the file with the provided encodings.")

        df = pd.read_csv(
            txt_path,
            sep='\t',
            skiprows=skiprows,
            encoding=correct_encoding,
            engine='python',
            header=0,    # the very next line is the column names
            #dtype=str    # parse strings first
        )
        
        # print("Shape of loaded dataframe is:")
        # print("Metadata")
        # print(metadata)
        # print(df.shape)
        # print(df)

        return df, metadata
    
    def load_horiba_1D(directory_path: str, mode: str) -> pd.DataFrame:
        """
        Load all .txt spectra in `directory_path`, stacking them into a
        2D DataFrame whose rows are the scan‐keys (voltage/time/distance)
        and whose columns are the Raman shifts.
    
        Parameters
        ----------
        directory_path : str
            Path to folder containing “*.txt” Horiba files.
        mode : str
            One of "SEC", "TIMESCAN", or "LINESCAN" (case‐insensitive).
    
        Returns
        -------
        df : pd.DataFrame
            Index = scan keys (float), sorted ascending.
            Columns = Raman shift values (float), same for every file.
            Values = intensity (float).
        """
    
        pattern = os.path.join(directory_path, "*.txt")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise ValueError(f"No .txt files in {directory_path!r}")
    
        df = None
        shift_values = None
        encodings    = ['ansi','utf-8','utf-16','iso-8859-1','windows-1250']
    
        for fp in files:
            # 1) try to load columns 0,1 skipping 40 rows under various encodings
            data = None
            for enc in encodings:
                try:
                    data = np.loadtxt(fp, usecols=(0,1),
                                      skiprows=40, encoding=enc)
                    break
                except Exception:
                    continue
            if data is None:
                raise ValueError(f"Could not read {fp} with any of {encodings}")
    
            shift, intensity = data[:,0], data[:,1]
    
            # 2) On the first file, establish the shift‐axis and empty DataFrame
            if shift_values is None:
                shift_values = shift.copy()
                df = pd.DataFrame(
                    [], 
                    columns=shift_values.astype(float),
                    dtype=float
                )
            else:
                # ensure every file has the same shift axis
                if not np.allclose(shift, shift_values):
                    print(f'file {fp} has different x axis')
                    #raise ValueError(f"Shift axis mismatch in file {fp}")
    
            # 3) Determine the “key” for this row by mode
            base = os.path.basename(fp)
            m   = None
    
            if mode.upper() == "SEC":
                m = re.search(r"([-+]?\d+(?:\.\d+)?)mV", base)
                if m:
                    key = float(m.group(1))
                else:
                    print(f"⚠️  {base} skipped: no “mV” tag found")
                    continue
    
            elif mode.upper() == "TIMESCAN":
                key = None
                with open(fp, encoding=enc) as f:
                    for line in f:
                        if line.startswith("#Time"):
                            try:
                                key = float(line.split("=",1)[1])
                            except:
                                pass
                            break
                if key is None:
                    print(f"⚠️  {base} skipped: no #Time=… in header")
                    continue
    
            elif mode.upper() == "LINESCAN":
                key = None
                with open(fp, encoding=enc) as f:
                    for line in f:
                        if line.startswith("#Distance"):
                            try:
                                key = float(line.split("=",1)[1])
                            except:
                                pass
                            break
                if key is None:
                    print(f"⚠️  {base} skipped: no #Distance=… in header")
                    continue
    
            else:
                raise ValueError(f"Unknown mode {mode!r}")
    
            # 4) Avoid duplicate keys
            if key in df.index:
                print(f"⚠️  duplicate key {key} in {base}; skipping")
                continue
    
            # 5) Append the intensity row
            df.loc[key] = intensity
    
        # 6) Finalize
        df.index.name = mode.lower()
        # print(df)
        return df.sort_index()

class Processing:
    def __init__(self, data):
        self.x = data['X'].values
        self.y = data['Y'].values
        
    def composite_func(self, x, *all_params, components):
        """
        Evaluate the sum of all components at x using the flat all_params array.
        'components' is the *structure* (list of dicts) you pass in from the GUI.
        """
        y_tot = np.zeros_like(x, dtype=float)
        idx = 0
        for comp in components:
            name   = comp['model_name']
            pnames = model_dict[name]['params']
            n = len(pnames)
            pvals = all_params[idx:idx+n]
            idx += n
            func = model_dict[name]['func']
            y_tot += func(x, *pvals)
        return y_tot
        
    def fit(self, components, p0, bounds):
        """
        components: list of dicts with 'model_name', but ignore their .params/.bounds here
        p0:     1D array of initial guesses (concatenated)
        bounds: (lower_bounds_array, upper_bounds_array)
        """
        # wrap composite_func so curve_fit can see it without the extra kwarg:
        def f(x, *params):
            return self.composite_func(x, *params, components=components)

        popt, pcov = curve_fit(
            f,
            self.x,
            self.y,
            p0=p0,
            bounds=bounds
        )
        return popt, pcov
    
    def extract_witec_coordinates(column_name):
        match = re.search(r'\((\d+)/(\d+)\)', column_name)
        if match:
            corX = int(match.group(1))
            corY = int(match.group(2))
            return corX, corY
        else:
            return None, None

    
    @staticmethod
    def calculate_region_metrics(column_name, intensity_series):
        """
        intensity_series : pd.Series
          index = shift (float), values = intensity
        Returns [X, Y, area, maximum, max_position] or None on failure.
        """
        corX, corY = Processing.extract_witec_coordinates(column_name)
        if corX is None:
            return None
    
        # ensure numeric
        x = intensity_series.index.values.astype(float)
        y = intensity_series.values.astype(float)
    
        if len(x)==0 or np.all(np.isnan(y)):
            return None
    
        # 1) area under the curve
        area = simpson(y, x)
    
        # 2) maximum and its location
        idx_max = np.nanargmax(y)
        maximum     = float(y[idx_max])
        max_position = float(x[idx_max])
    
        return [corX, corY, area, maximum, max_position]
    
    @staticmethod
    def compute_2d_metric(df_raw, lower_shift=None, upper_shift=None):
        """
        df_raw: wide DataFrame where
          - df_raw.iloc[:,0] is the shift axis (float)
          - df_raw.columns[1:] are intensity columns named "...(i/j)"
        Returns a long DataFrame with columns
          ['X','Y','area','maximum','max_position'].
        """
        # 1) extract the shift axis
        shift_col = df_raw.columns[0]
        shifts = df_raw[shift_col].astype(float)
    
        # 2) determine clip window
        lo = shifts.min() if lower_shift is None else lower_shift
        hi = shifts.max() if upper_shift is None else upper_shift
    
        # 3) boolean mask on the label‐based index
        mask = (shifts >= lo) & (shifts <= hi)
    
        # 4) slice out only those rows *via .loc*
        clipped_shifts = shifts.loc[mask].reset_index(drop=True)
        # and only the intensity columns
        int_cols      = df_raw.columns[1:]
        clipped_df    = df_raw.loc[mask, int_cols].astype(float).reset_index(drop=True)
    
        if clipped_df.shape[1] == 0:
            # no spectra in range
            return pd.DataFrame(columns=['X','Y','area','maximum','max_position'])
    
        # 5) serially compute metrics
        records = []
        for col in clipped_df.columns:
            # build a Series with shift as index
            s = pd.Series(clipped_df[col].values, index=clipped_shifts.values)
            rec = Processing.calculate_region_metrics(col, s)
            if rec is not None:
                records.append(rec)
    
        # 6) assemble long‐form DataFrame
        return pd.DataFrame.from_records(
            records,
            columns=['X','Y','area','maximum','max_position']
        )
    
    @staticmethod
    def _make_composite(x, all_params, components):
        """ Helper to evaluate the composite model. """
        y = np.zeros_like(x)
        idx = 0
        for comp in components:
            model_name = comp["model_name"]
            pnames = model_dict[model_name]["params"]
            vals = all_params[idx: idx + len(pnames)]
            idx += len(pnames)
            y += model_dict[model_name]["func"](x, *vals)
        return y
    
    # @staticmethod
    # def compute_2d_fit(df_raw, components, lower_shift=None, upper_shift=None):
    #     """
    #     For each pixel (column) in df_raw:
    #      - clip to lower/upper shift
    #      - fit the composite model defined by `components`
    #      - return a DataFrame with columns X, Y, and each fitted param
    #     """
    #     # 1) build the shift vector + intensity block
    #     shift = df_raw.iloc[:,0].astype(float).to_numpy()
    #     data  = df_raw.iloc[:,1:].astype(float)

    #     # 2) clip by shift
    #     lo = shift.min() if lower_shift is None else lower_shift
    #     hi = shift.max() if upper_shift is None else upper_shift
    #     mask = (shift >= lo) & (shift <= hi)
    #     x    = shift[mask]

    #     # 3) prepare a global p0 & bounds from your components
    #     p0_list, lb_list, ub_list = [], [], []
    #     for comp in components:
    #         model_name = comp["model_name"]
    #         pnames = model_dict[model_name]["params"]
    #         for pn in pnames:
    #             p0_list.append(comp["params"][pn])
    #             lo_b, hi_b = comp["bounds"][pn]
    #             lb_list.append(-np.inf if lo_b is None else lo_b)
    #             ub_list.append( np.inf if hi_b is None else hi_b)
    #     p0     = np.array(p0_list, dtype=float)
    #     bounds = (np.array(lb_list, dtype=float), np.array(ub_list, dtype=float))

    #     # 4) define a helper that fits one pixel
    #     def _fit_column(colname):
    #         y = data[colname].values[mask]
    #         xpix, ypix = Processing.extract_witec_coordinates(colname)
    #         if xpix is None:
    #             return None

    #         try:
    #             popt, _ = curve_fit(
    #                 lambda xx, *pp: Processing._make_composite(xx, pp, components),
    #                 x, y, p0=p0, bounds=bounds
    #             )
    #         except:
    #             return None

    #         # assemble the record
    #         rec = {"X": xpix, "Y": ypix}
    #         idx = 0
    #         for comp in components:
    #             model_name = comp["model_name"]
    #             pnames = model_dict[model_name]["params"]
    #             label = comp.get("label", model_name)
    #             for pn in pnames:
    #                 rec[f"{label}_{pn}"] = popt[idx]
    #                 idx += 1
    #         return rec

    #     # 5) run in parallel
    #     records = Parallel(n_jobs=-1)(
    #         delayed(_fit_column)(col) for col in data.columns
    #     )
    #     records = [r for r in records if r is not None]

    #     # 6) collect into a DataFrame
    #     return pd.DataFrame.from_records(records)
    
    @staticmethod
    def compute_2d_fit(df_raw, components, lower_shift=None, upper_shift=None):
        """
        For each pixel (column) in df_raw:
         - clip to lower/upper shift
         - fit the composite model defined by `components`
         - return a DataFrame with columns X, Y, and each fitted param
        """
        # 1) Build the shift vector + intensity block
        shift = df_raw.iloc[:, 0].astype(float).to_numpy()
        data = df_raw.iloc[:, 1:].astype(float)
    
        # 2) Clip by shift
        lo = shift.min() if lower_shift is None else lower_shift
        hi = shift.max() if upper_shift is None else upper_shift
        mask = (shift >= lo) & (shift <= hi)
        x = shift[mask]
    
        # 3) Prepare initial parameters and bounds
        p0_list, lb_list, ub_list = [], [], []
        for comp in components:
            model_name = comp["model_name"]
            pnames = model_dict[model_name]["params"]
            for pn in pnames:
                p0_list.append(comp["params"][pn])
                lo_b, hi_b = comp["bounds"][pn]
                lb_list.append(-np.inf if lo_b is None else lo_b)
                ub_list.append(np.inf if hi_b is None else hi_b)
        p0 = np.array(p0_list, dtype=float)
        bounds = (np.array(lb_list, dtype=float), np.array(ub_list, dtype=float))
    
        # 4) Fit each column sequentially
        records = []
        for colname in data.columns:
            y = data[colname].values[mask]
            xpix, ypix = Processing.extract_witec_coordinates(colname)
            if xpix is None:
                continue
    
            try:
                popt, _ = curve_fit(
                    lambda xx, *pp: Processing._make_composite(xx, pp, components),
                    x, y, p0=p0, bounds=bounds
                )
            except Exception:
                continue
    
            # Assemble the result record
            rec = {"X": xpix, "Y": ypix}
            idx = 0
            for comp in components:
                model_name = comp["model_name"]
                pnames = model_dict[model_name]["params"]
                label = comp.get("label", model_name)
                for pn in pnames:
                    rec[f"{label}_{pn}"] = popt[idx]
                    idx += 1
            records.append(rec)
    
        return pd.DataFrame.from_records(records)

    
    # @staticmethod
    # def compute_average(df2d,
    #                     lower_shift=None, upper_shift=None,
    #                     lower_x=None,   upper_x=None,
    #                     lower_y=None,   upper_y=None):
    #     """
    #     Take a 2D-map DataFrame whose index is Raman shift (float)
    #     and whose columns are a MultiIndex (X, Y), and return
    #     (x, y_avg) where y_avg is the mean over all pixels
    #     inside the specified shift/X/Y box.
    #     """
    #     # 1) restrict shift‐axis
    #     idx = df2d.index.astype(float)
    #     lo = idx.min() if lower_shift is None else lower_shift
    #     hi = idx.max() if upper_shift is None else upper_shift
    #     clipped = df2d.loc[(idx >= lo) & (idx <= hi)]

    #     if clipped.shape[1] == 0:
    #         raise ValueError(f"No pixels found for shift ∈ [{lo},{hi}]")

    #     # 2) restrict X/Y coordinates
    #     cols = clipped.columns  # MultiIndex [(X,Y),…]
    #     mask = np.ones(len(cols), dtype=bool)
    #     if lower_x is not None:
    #         mask &= cols.get_level_values("X") >= lower_x
    #     if upper_x is not None:
    #         mask &= cols.get_level_values("X") <= upper_x
    #     if lower_y is not None:
    #         mask &= cols.get_level_values("Y") >= lower_y
    #     if upper_y is not None:
    #         mask &= cols.get_level_values("Y") <= upper_y

    #     sel = clipped.loc[:, mask]
    #     if sel.shape[1] == 0:
    #         raise ValueError(f"No pixels in X×Y ∈ [{lower_x},{upper_x}]×[{lower_y},{upper_y}]")

    #     # 3) average across remaining columns
    #     x_vals = sel.index.to_numpy()
    #     y_avg  = sel.mean(axis=1).to_numpy()

    #     return x_vals, y_avg
    
    @staticmethod
    def compute_average(df2d,
                        lower_shift=None, upper_shift=None,
                        lower_x=None,     upper_x=None,
                        lower_y=None,     upper_y=None):
        """
        Take a “wide” map‐DataFrame `df2d` whose
        - first column is the Raman‐shift axis,
        - remaining columns are intensities at (X,Y) pixels
          (either a MultiIndex columns or names like "(x/y)").
        
        Steps:
          1) pull off that first column into an array `shifts`
          2) build an `intens` DataFrame of everything else
          3) trim rows by shift‐limits
          4) trim columns by X/Y‐limits
          5) average across the remaining columns → one spectrum
        Returns
        -------
        x_avg : np.ndarray  # the selected-shift axis
        y_avg : np.ndarray  # the mean intensity at each shift
        """
        # 1) Separate shift‐axis vs. intensities
        shifts = df2d.iloc[:, 0].astype(float).to_numpy()
        intens = df2d.iloc[:, 1:].astype(float)

        # 2) Clip rows by shift
        lo = shifts.min() if lower_shift is None else lower_shift
        hi = shifts.max() if upper_shift is None else upper_shift
        row_mask = (shifts >= lo) & (shifts <= hi)
        if not row_mask.any():
            raise ValueError(f"No data in shift‐range [{lo},{hi}]")
        shifts_sel = shifts[row_mask]
        intens_sel = intens.loc[row_mask, :]

        # 3) Figure out X,Y for each column
        cols = intens_sel.columns
        # if it's a MultiIndex, level 0=X, level1=Y
        if isinstance(cols, pd.MultiIndex):
            xs = cols.get_level_values(0).astype(float)
            ys = cols.get_level_values(1).astype(float)
        else:
            coords = [Processing.extract_witec_coordinates(name) for name in cols]
            mi = pd.MultiIndex.from_tuples(coords, names=("X","Y"))
            intens_sel.columns = mi
            xs = mi.get_level_values(0).astype(float)
            ys = mi.get_level_values(1).astype(float)
            
        # DEBUG: tell me what X/Y values you actually have
        print(">>> compute_average available X:", np.unique(xs))
        print(">>> compute_average available Y:", np.unique(ys))

        # 4) Trim columns by X/Y limits
        col_mask = np.ones(len(xs), dtype=bool)
        if lower_x is not None:
            col_mask &= (xs >= lower_x)
        if upper_x is not None:
            col_mask &= (xs <= upper_x)
        if lower_y is not None:
            col_mask &= (ys >= lower_y)
        if upper_y is not None:
            col_mask &= (ys <= upper_y)

        if not col_mask.any():
            raise ValueError(
                f"No pixels in X×Y ranges "
                f"[{lower_x},{upper_x}]×[{lower_y},{upper_y}]"
            )
        intens_sel = intens_sel.iloc[:, col_mask]

        # 5) Finally, average across the selected columns
        y_avg = intens_sel.mean(axis=1).to_numpy()
        x_avg = shifts_sel
        
        # y_avg = intens_sel.mean(axis=1).astype(float).values
        # x_avg = shifts_sel.astype(float).values

        return x_avg, y_avg
    
    # def compute_1d_fit(df, components, shift_min=None, shift_max=None):
    #     """
    #     Fit each spectrum (row) in the given 1D Raman DataFrame with a defined model.
    
    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         2D Raman map, index = scan coordinate, columns = Raman shift, values = intensity
    #     components : list of dict
    #         Model components from the main window
    #     shift_min : float or None
    #         Lower Raman shift limit
    #     shift_max : float or None
    #         Upper Raman shift limit
    
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Fitted parameters for each row, indexed by the scan coordinate
    #     """
    
    #     results = []
    
    #     for idx, row in df.iterrows():
    #         x = df.columns.values.astype(float)
    #         y = row.values.astype(float)
    
    #         # Restrict to fitting window
    #         mask = np.ones_like(x, dtype=bool)
    #         if shift_min is not None:
    #             mask &= x >= shift_min
    #         if shift_max is not None:
    #             mask &= x <= shift_max
    #         x_fit = x[mask]
    #         y_fit = y[mask]
    
    #         if len(x_fit) < 5:
    #             print(f"⚠️ Not enough points for fitting at index {idx}")
    #             continue
    
    #         # Composite model
    #         def composite(x, *params):
    #             i = 0
    #             y_model = np.zeros_like(x)
    #             for comp in components:
    #                 model = comp['model']
    #                 if model in ('gaussian', 'lorentzian'):
    #                     amp, ctr, sig = params[i:i+3]
    #                     if model == 'gaussian':
    #                         y_model += amp * np.exp(-(x - ctr)**2 / (2 * sig**2))
    #                     elif model == 'lorentzian':
    #                         y_model += amp * (0.5*sig)**2 / ((x - ctr)**2 + (0.5*sig)**2)
    #                     i += 3
    #             return y_model
    
    #         # Initial params and bounds
    #         p0 = []
    #         bounds_lower, bounds_upper = [], []
    #         for comp in components:
    #             p0.extend([
    #                 comp['params']['amplitude'],
    #                 comp['params']['center'],
    #                 comp['params']['sigma']
    #             ])
    #             bounds_lower.extend([
    #                 comp['bounds']['amplitude'][0],
    #                 comp['bounds']['center'][0],
    #                 comp['bounds']['sigma'][0]
    #             ])
    #             bounds_upper.extend([
    #                 comp['bounds']['amplitude'][1],
    #                 comp['bounds']['center'][1],
    #                 comp['bounds']['sigma'][1]
    #             ])
    
    #         try:
    #             popt, _ = curve_fit(composite, x_fit, y_fit, p0=p0, bounds=(bounds_lower, bounds_upper))
    #             result = {'index': idx}
    #             for i, comp in enumerate(components):
    #                 base = i * 3
    #                 result[f"{comp['label']}_amp"] = popt[base]
    #                 result[f"{comp['label']}_ctr"] = popt[base+1]
    #                 result[f"{comp['label']}_sig"] = popt[base+2]
    #             results.append(result)
    
    #         except Exception as e:
    #             print(f"⚠️ Fit failed at index {idx}: {e}")
    
    #     df_result = pd.DataFrame(results)
    #     df_result.set_index('index', inplace=True)
    #     df_result.index.name = df.index.name
    #     return df_result

    # def compute_1d_fit(df, components, model_dict, shift_min=None, shift_max=None):
    #     """
    #     Fit each row of 1D Raman data with the user-defined model.
        
    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Raman map: index = scan coordinate, columns = Raman shifts, values = intensity
    #     components : list of dict
    #         User-defined components from the GUI (with model_name, params, bounds, etc.)
    #     model_dict : dict
    #         Dictionary of models and their parametric forms (same as in 2D)
    #     shift_min, shift_max : float or None
    #         Raman shift fitting window
        
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Fitting results indexed by scan coordinate
    #     """
        
    #     results = []
    #     x_full = df.columns.values.astype(float)
    
    #     # Pre-generate parameter structure
    #     param_names = []
    #     def composite_model(x, *params):
    #         y_fit = np.zeros_like(x)
    #         i = 0
    #         for comp in components:
    #             model = comp["model_name"]
    #             fn    = model_dict[model]["func"]
    #             pnames = model_dict[model]["params"]
    #             n = len(pnames)
    #             pset = {name: params[i + j] for j, name in enumerate(pnames)}
    #             y_fit += fn(x, **pset)
    #             i += n
    #         return y_fit
    
    #     # Prepare p0 and bounds globally (same for every row)
    #     p0_list, lb_list, ub_list = [], [], []
    #     for comp in components:
    #         model = comp["model_name"]
    #         for pn in model_dict[model]["params"]:
    #             p0_list.append(comp["params"][pn])
    #             lo_b, hi_b = comp["bounds"][pn]
    #             lb_list.append(-np.inf if lo_b is None else lo_b)
    #             ub_list.append( np.inf if hi_b is None else hi_b)
    #             param_names.append(f"{comp['label']}_{pn}")
    #     p0 = np.array(p0_list, dtype=float)
    #     bounds = (np.array(lb_list, dtype=float), np.array(ub_list, dtype=float))
    
    #     for idx, row in df.iterrows():
    #         y = row.values.astype(float)
    
    #         # Apply shift filtering
    #         mask = np.ones_like(x_full, dtype=bool)
    #         if shift_min is not None:
    #             mask &= x_full >= float(shift_min)
    #         if shift_max is not None:
    #             mask &= x_full <= float(shift_max)
    
    #         x = x_full[mask]
    #         y = y[mask]
    
    #         if len(x) < len(p0):
    #             print(f"⚠️ Not enough points to fit at {idx}")
    #             continue
    
    #         try:
    #             popt, _ = curve_fit(composite_model, x, y, p0=p0, bounds=bounds)
    #             result = dict(zip(param_names, popt))
    #             result["index"] = idx
    #             results.append(result)
    #         except Exception as e:
    #             print(f"⚠️ Fit failed at {idx}: {e}")
    #             continue
    
    #     df_result = pd.DataFrame(results)
    #     if not df_result.empty:
    #         df_result.set_index("index", inplace=True)
    #         df_result.index.name = df.index.name
    
    #     return df_result
    
    @staticmethod
    def compute_1d_fit(df_raw, components, lower_shift=None, upper_shift=None):
        """
        For each row (spectrum) in df_raw:
         - clip to lower/upper shift
         - fit the composite model defined by `components`
         - return a DataFrame with index = scan axis, and each fitted param as columns
        """
    
        # 1) Get shift vector from columns
        shift = df_raw.columns.astype(float).to_numpy()
    
        # 2) Clip by shift
        lo = shift.min() if lower_shift is None else lower_shift
        hi = shift.max() if upper_shift is None else upper_shift
        mask = (shift >= lo) & (shift <= hi)
        x = shift[mask]
    
        # 3) Prepare global p0 and bounds
        p0_list, lb_list, ub_list = [], [], []
        for comp in components:
            model_name = comp["model_name"]
            pnames = model_dict[model_name]["params"]
            for pn in pnames:
                p0_list.append(comp["params"][pn])
                lo_b, hi_b = comp["bounds"][pn]
                lb_list.append(-np.inf if lo_b is None else lo_b)
                ub_list.append(np.inf if hi_b is None else hi_b)
        p0 = np.array(p0_list, dtype=float)
        bounds = (np.array(lb_list, dtype=float), np.array(ub_list, dtype=float))
    
        # 4) Fit each row in df_raw
        records = []
        for idx, row in df_raw.iterrows():
            y = row.values[mask]
    
            if len(y) < len(p0):
                continue
    
            try:
                popt, _ = curve_fit(
                    lambda xx, *pp: Processing._make_composite(xx, pp, components),
                    x, y, p0=p0, bounds=bounds
                )
            except Exception:
                continue
    
            # Assemble result
            rec = {"index": idx}
            i = 0
            for comp in components:
                model_name = comp["model_name"]
                pnames = model_dict[model_name]["params"]
                label = comp.get("label", model_name)
                for pn in pnames:
                    rec[f"{label}_{pn}"] = popt[i]
                    i += 1
            records.append(rec)
    
        # Return as indexed dataframe
        df_result = pd.DataFrame.from_records(records)
        if not df_result.empty:
            df_result.set_index("index", inplace=True)
            df_result.index.name = df_raw.index.name
    
        return df_result

class Plotting:
    def __init__(self, data):
        self.data = data
    
    def update_plot(fig, data, components, filtered_data):

        fig.clear()

        x = data['X'].values
        y = data['Y'].values
        
        x_dense   = np.linspace(x.min(), x.max(), len(x)*10)
        composite = np.zeros_like(x_dense)
        composite_at_x = np.zeros_like(x)
        counter = 0

        # accumulate each component into composite
        for comp in components:
            name   = comp['model_name']
            func   = model_dict[name]['func']
            pnames = model_dict[name]['params']
            pvals  = [comp['params'].get(pn, 0.0) for pn in pnames]
            y_comp = func(x_dense, *pvals)
            composite += y_comp
            
            y_comp_x = func(x, *pvals)
            composite_at_x += y_comp_x

        if not np.allclose(composite, 0.0):
            # — full plot: individual components + composite & residual spectrum
            gs = GridSpec(nrows=2, ncols=1, height_ratios=[2, 8], hspace=0.05, figure=fig)
            ax_bott_left  = fig.add_subplot(gs[1,0])
            ax_bott_right = ax_bott_left.twinx()
            for comp in components:
                name   = comp['model_name']
                label  = comp.get("label", comp["model_name"])
                func   = model_dict[name]['func']
                pnames = model_dict[name]['params']
                pvals  = [comp['params'].get(pn, 0.0) for pn in pnames]
                y_comp = func(x_dense, *pvals)

                if not np.allclose(y_comp, 0.0):
                    if name in ('Linear','Sigmoid'):
                        ax_bott_left.plot(x_dense, y_comp, '--', c='gray', label=label, alpha=0.8)
                    else:
                        counter = counter + 1
                        if name == func:
                            ax_bott_right.plot(x_dense, y_comp, '--', label=f'{name}{counter}', alpha=0.8)
                        else:
                            ax_bott_right.plot(x_dense, y_comp, '--', label=label, alpha=0.8)
            ax_bott_right.set_ylabel('Intensity (arb. u.)')
            if filtered_data is not None:
                ax_bott_left.plot(x, y, linestyle = 'dotted', color='black', label='Data (raw)')
                ax_bott_left.plot(filtered_data["X"].values, filtered_data["Y"].values, color='blue', label='Filtered data')
            else:
                ax_bott_left.scatter(x, y, s=5, color='blue', label='Data (raw)')
            ax_bott_left.plot(x_dense, composite, '-', color='red', label='Composite')
            
            if y.max() > composite.max():
                y_max = y.max()
            else:
                y_max = composite.max()
                
            if y.min() < composite.min():
                y_min = y.min()
            else:
                y_min = composite.min()
                
            diff = y_max - y_min
            ax_bott_right.set_ylim(0-0.05*diff, diff + 0.05*diff)
            ax_bott_left.set_ylim(y_min-0.05*diff, y_max + 0.05*diff)
            ax_bott_right.set_ylabel('Intensity (arb. u.)') 
            ax_bott_left.legend(loc='upper left')
            ax_bott_right.legend(loc='upper right')
            ax_bott_left.set_xlabel(r'Raman shift (cm$^{-1}$)')
            ax_bott_left.set_ylabel('Intensity (arb. u.)')
            
            if filtered_data is not None:
                residual = filtered_data["Y"] - composite_at_x
            else:
                residual = y - composite_at_x
            
            ax_top = fig.add_subplot(gs[0,0])
            ax_top.plot(x, residual, 'k-')
            ax_top.axhline(0, color='red', linestyle='--', linewidth=0.8)
            ax_top.set_ylabel("Residual")
            ax_top.tick_params(axis='x', labelbottom=False) 
        else:
            ax_left  = fig.add_subplot(111)
            if filtered_data is not None:
                ax_left.plot(x, y, linestyle = 'dotted', color = 'black', label='Raw data')
                ax_left.plot(filtered_data["X"].values, filtered_data["Y"].values, color='blue', label='Filtered data')
            else:
                ax_left.plot(x, y, color = 'blue', label='Raw data')
            ax_left.legend(loc='best')
            ax_left.set_xlabel(r'Raman shift (cm$^{-1}$)')
            ax_left.set_ylabel('Intensity (arb. u.)')
            
    def lin_bcg(row):
        xs = row.index.to_numpy().astype(float)
        ys = row.to_numpy().astype(float)
        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        intercept = ys[0] - slope * xs[0]
        return slope*xs-intercept
    
    @staticmethod
    def plot_1d_overlay(ax,
                        df,
                        *,
                        lower_shift=None,
                        upper_shift=None,
                        lower_index=None,
                        upper_index=None,
                        offset_percent=20,
                        cmap_name="Viridis",
                        correction="None",
                        mode="SEC"):
        """
        Draw each row of `df` as a line‐stacked overlay with a color bar.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        df : pd.DataFrame
            Index = scan key, Columns = Raman shifts (must be numeric).
        lower_shift, upper_shift : float or None
            Clip the X‐axis to [lower_shift, upper_shift], default = full range.
        lower_index, upper_index : int/float or None
            Clip the Y‐axis (row indices) to [lower_index, upper_index], default = full range.
        offset_percent : float
            Vertical offset between spectra = (offset_percent/100) × max_intensity.
        cmap_name : str
            Name of a Matplotlib colormap for mapping scan‐key → color.
        correction : {"None","Zero","Linear"}
            Row‐wise baseline correction before plotting.
        mode : {"SEC","TIMESCAN","LINESCAN"}
            Used only to label the colorbar.
        title : str, optional
            Plot title.
        """

        # 0) grab the figure & clear everything
        fig = ax.get_figure()
        fig.clear()
        ax  = fig.add_subplot(111)

        # 1) derive numeric shift‐axis from column names
        shifts = df.columns.to_numpy().astype(float)

        # 2) clamp shift bounds
        low_sh  = shifts.min() if lower_shift  is None else max(lower_shift, shifts.min())
        high_sh = shifts.max() if upper_shift  is None else min(upper_shift, shifts.max())

        # 3) clamp row‐index bounds
        low_ix  = df.index.min() if lower_index is None else max(lower_index, df.index.min())
        high_ix = df.index.max() if upper_index is None else min(upper_index, df.index.max())

        # 4) slice the DataFrame
        col_mask = (shifts >= low_sh) & (shifts <= high_sh)
        cols     = df.columns[col_mask]
        Data_df  = df.loc[low_ix:high_ix, cols]

        # 5) optional baseline correction
        if correction == "Zero":
            Data_df = Data_df.apply(lambda r: r - r.min(), axis=1)
        elif correction == "Linear":
            Data_df = Data_df.apply(lambda r: r - Plotting.lin_bcg(r), axis=1)
            Data_df = Data_df.apply(lambda r: r - r.min(), axis=1)
        # else correction=="None" → leave Data_df as is

        # 6) compute offset step
        max_val     = Data_df.values.max()
        offset_step = offset_percent/100 * max_val

        # 7) prepare for color‐mapping by row‐index
        distances = Data_df.index.to_numpy().astype(float)
        norm_dist = (distances - distances.min()) / (distances.max() - distances.min())
        cmap      = plt.get_cmap(cmap_name)

        # 8) plot each spectrum with its offset
        for i, (_, row) in enumerate(Data_df.iterrows()):
            y = row.to_numpy().astype(float) + i * offset_step
            ax.plot(shifts[col_mask], y, color=cmap(norm_dist[i]))

        # 9) add a colorbar showing scan‐key → color
        norm = Normalize(vmin=distances.min(), vmax=distances.max())
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label({
            "TIMESCAN":"Time (s)",
            "LINESCAN":r"Distance (μm)",
            "SEC":"Potential (mV)"
        }[mode.upper()])

        # 10) decorate axes
        ax.set_xlim(low_sh, high_sh)
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")

        # 11) tighten up
        fig.tight_layout()
        return ax, Data_df
    
    @staticmethod
    def plot_1D_map(ax, df: pd.DataFrame, 
                    lower_shift, upper_shift,
                    lower_index, upper_index,
                    cmap_name, correction,
                    orientation, mode,
                    interpolation):

        shifts = df.columns.to_numpy().astype(float)
                    
        # Find the columns (shifts) within the specified range
        filtered_columns = (shifts > lower_shift) & (shifts <= upper_shift)
        Data_df = df.loc[lower_index:upper_index, filtered_columns]

        if correction == "Zero":
            Data_df = Data_df.apply(lambda row: row - row.min(), axis=1)
        elif correction == "Linear":
            Data_df = Data_df.apply(lambda row: row - Plotting.lin_bcg(row), axis=1)
            Data_df = Data_df.apply(lambda row: row - row.min(), axis=1)
        
        fig = ax.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        # 1) remove any extra axes (e.g. old colorbars)
        #    collect all axes except the main one
        extra_axes = [a for a in fig.axes if a is not ax]
        for a in extra_axes:
            fig.delaxes(a)
        
        # 2) clear the main axes
        ax.clear()
        extent = [lower_shift, upper_shift, lower_index, upper_index]
            
        if orientation == "descending":
            img = ax.imshow(
                Data_df.values,
                aspect='auto',
                extent=extent,
                origin="lower",
                cmap=cmap_name,
                interpolation=interpolation
            )
            ax.invert_yaxis()
        else:
            img = ax.imshow(
                Data_df.values,
                aspect='auto',
                extent=extent,
                origin="lower",
                cmap=cmap_name,
                interpolation=interpolation
            )
        
        # --- labels & title ---
        ylabels = {
            "Timescan": "Time (s)",
            "Linescan": r"Distance ($\mu$m)",
            "SEC": "Potential (mV)"
        }
        ax.set_ylabel(ylabels.get(mode, ""))
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        
        fig.colorbar(img, ax=ax, label="Intensity (a.u.)")
        
        # # 5) Create a fixed-position colorbar using a divider
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # ax.figure.colorbar(img, cax=cax, label="Intensity (a.u.)")
        return ax, Data_df
        
    @staticmethod
    def plot_2d(ax, df, meta_data, source,
                       cmap_name, interp,
                       low_x, upp_x,
                       low_y, upp_y,
                       range_min, range_max):

        if low_x  is not None: df = df[df['X'] >= low_x]
        if upp_x  is not None: df = df[df['X'] <= upp_x]
        if low_y  is not None: df = df[df['Y'] >= low_y]
        if upp_y  is not None: df = df[df['Y'] <= upp_y]
        
        if df.empty:
            raise ValueError("No data in that X/Y region")
        
        heat_map = df.pivot(index="Y", columns="X", values=source)
        
        # get the “real” pixel coordinates
        xs = heat_map.columns.astype(float).to_numpy()
        ys = heat_map.index.astype(float).to_numpy()
        
        lo = range_min if range_min is not None else float(heat_map.min().min())
        hi = range_max if range_max is not None else float(heat_map.max().max())
        
        fig = ax.get_figure()
        fig.clear()
        ax  = fig.add_subplot(111)

        im = ax.imshow(
            heat_map.values,
            origin='lower',
            aspect='auto',
            cmap=cmap_name,
            interpolation=interp,
            vmin=lo,
            vmax=hi,
            extent=[xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5]
        )

        # colorbar
        cb = fig.colorbar(im, ax=ax, orientation='vertical')
        cb.set_label(source)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.tight_layout()
        return ax
        
class Filtering:
    @staticmethod
    def savitzky_golay(x, y, window_length: int, polyorder: int) -> np.ndarray:
        """
        Savitzky–Golay smoothing. Requires odd window_length >= polyorder+2.
        """
        # Ensure window_length is odd and <= len(y)
        wl = max(3, int(window_length) | 1)  # force odd, at least 3
        if wl > len(y):
            wl = len(y) if len(y) % 2 == 1 else len(y) - 1
        po = min(wl - 1, max(0, int(polyorder)))
        return savgol_filter(y, window_length=wl, polyorder=po)

    @staticmethod
    def moving_average(x, y, window_size: int) -> np.ndarray:
        """
        Simple moving average (boxcar). Pads endpoints with edge values.
        """
        w = int(window_size)
        if w < 1:
            return y.copy()
        kernel = np.ones(w) / w
        # Use 'nearest' padding via np.pad so length stays constant
        y_padded = np.pad(y, (w//2, w//2), mode="edge")
        return np.convolve(y_padded, kernel, mode="valid")[: len(y)]

    @staticmethod
    def median_filter(x, y, window_size: int) -> np.ndarray:
        """
        Median filter to remove spikes. Requires odd window_size.
        """
        w = int(window_size) | 1  # force odd
        return medfilt(y, kernel_size=w)
    
    @staticmethod
    def fft_lowpass(x, y, cutoff):
        """
        Simple Fourier‐domain low‐pass filter.
    
        Parameters:
        -----------
        x : array‐like (uniformly spaced)
        y : array‐like
        cutoff : float
            Frequency cutoff (in 1/(x unit)).
    
        Returns:
        --------
        y_smooth : np.ndarray
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 2:
            return y.copy()
    
        dx = np.mean(np.diff(x))
        Yf = np.fft.fft(y)
        freqs = np.fft.fftfreq(n, d=dx)
        Yf[np.abs(freqs) > cutoff] = 0
        y_smooth = np.fft.ifft(Yf).real
        return y_smooth