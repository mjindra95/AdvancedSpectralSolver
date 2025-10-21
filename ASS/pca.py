
"""
PCA setup & reconstruction window for ASS
Author: Martin Jindra
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ASS.logic import Loading
from glob import glob

# --- Global storage for the most recent PCA model ---
LAST_PCA_MODEL = None

def run_pca_calculation(loader_func, input_folder, output_folder, n_components=10, use_scaler=False): 
    """Perform PCA and save results (scores, loadings, scree, etc.).""" 
    global LAST_PCA_MODEL 
    
    all_files = sorted(glob(os.path.join(input_folder, "*.txt"))) 
    
    if not all_files: 
        raise FileNotFoundError(f"No files found in {input_folder}") 
        
    # --- Load spectra --- 
    x_common = None
    y_list, col_names = [], [] 
    seen = {} 
    def unique_name(name): 
        base = os.path.splitext(name)[0] 
        if base not in seen: 
            seen[base] = 0 
            return base 
        seen[base] += 1 
        return f"{base}_{seen[base]}" 
    
    for i, f in enumerate(all_files): 
        x, y, fname = loader_func(f) 
        ok = np.isfinite(x) & np.isfinite(y) 
        x, y = x[ok], y[ok] 
        if i == 0: 
            x_common = x.copy() 
        else: 
            if (len(x) != len(x_common)) or (not np.array_equal(x, x_common)):
                raise ValueError(f"X mismatch in {fname}") 
        
        y_list.append(y) 
        col_names.append(unique_name(fname)) 
    spectra = np.vstack(y_list) 
    n_samples, n_features = spectra.shape 
    n_comp_eff = min(n_components, min(n_samples, n_features)) 
    
    # --- PCA --- 
    scaler = StandardScaler(with_mean=True, with_std=True) if use_scaler else None 
    X_proc = scaler.fit_transform(spectra) if scaler else spectra 
    
    pca = PCA(n_components=n_comp_eff, svd_solver="full") 
    scores = pca.fit_transform(X_proc) 
    print("Scores:") 
    print(scores) 
    loadings = pca.components_ 
    print("Loadings:") 
    print(loadings) 
    explained_var = pca.explained_variance_ratio_ 
    cum_var = np.cumsum(explained_var) * 100
    
    # --- Save outputs --- 
    os.makedirs(output_folder, exist_ok=True)
    # Scree plot 
    fig, ax1 = plt.subplots(figsize=(6, 4)) 
    ax1.plot(range(1, n_comp_eff + 1), explained_var * 100, 'o-', label="Individual Var") 
    ax2 = ax1.twinx() 
    ax2.plot(range(1, n_comp_eff + 1), cum_var, 's--', color='red', label="Cumulative") 
    ax1.set_xlabel("Principal Component") 
    ax1.set_ylabel("Explained Variance (%)") 
    ax2.set_ylabel("Cumulative (%)") 
    fig.tight_layout() 
    fig.savefig(os.path.join(output_folder, "scree_plot.png"), dpi=200) 
    plt.close(fig) 
    
    # Scores 
    score_dir = os.path.join(output_folder, "scores") 
    os.makedirs(score_dir, exist_ok=True) 
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp_eff)]) 
    scores_df.insert(0, "file", col_names) 
    scores_df.to_excel(os.path.join(score_dir, "scores.xlsx"), index=False) 
    for i, j in combinations(range(n_comp_eff), 2): 
        fig = plt.figure(figsize=(5, 4)) 
        plt.scatter(scores[:, i], scores[:, j], s=15) 
        plt.xlabel(f"PC{i+1} ({explained_var[i]*100:.1f}%)") 
        plt.ylabel(f"PC{j+1} ({explained_var[j]*100:.1f}%)") 
        plt.tight_layout() 
        fig.savefig(os.path.join(score_dir, f"scores_PC{i+1}_PC{j+1}.png"), dpi=200) 
        plt.close(fig) 
        
    # Loadings 
    load_dir = os.path.join(output_folder, "loadings") 
    os.makedirs(load_dir, exist_ok=True) 
    for i in range(n_comp_eff): 
        np.savetxt(os.path.join(load_dir, f"PC{i+1}_loading.txt"), np.column_stack([x_common, loadings[i]]), fmt=["%.6f", "%.8f"], delimiter="\t", comments="") 
        fig = plt.figure(figsize=(7, 4)) 
        plt.plot(x_common, loadings[i], label=f"PC{i+1}") 
        plt.xlabel("Raman Shift (cm$^{-1}$)") 
        plt.ylabel("Loading") 
        plt.legend() 
        plt.tight_layout() 
        fig.savefig(os.path.join(load_dir, f"loading_PC{i+1}.png"), dpi=200) 
        plt.close(fig) 
        
    np.savez(os.path.join(output_folder, "pca_model.npz"), 
             x_common=x_common, 
             mean_vector=pca.mean_, 
             scores=scores, 
             loadings=loadings, 
             explained_var=explained_var, 
             cum_var=cum_var, 
             col_names=col_names, 
             use_scaler=use_scaler, 
             scaler_mean=(scaler.mean_ if scaler else None), 
             scaler_std=(scaler.scale_ if scaler else None)) 
    
    # Cache globally 
    LAST_PCA_MODEL = { "x": x_common, 
                      "spectra": spectra, 
                      "pca": pca, 
                      "scores": scores, 
                      "loadings": loadings, 
                      "scaler": scaler, 
                      "col_names": col_names } 
    return output_folder

def _as_bool(x):
    # x is a 0-d numpy array(bool) → return a python bool
    return bool(np.array(x).item())

def _maybe_none(arr):
    # convert 0-d object arrays back to Python None
    return arr.item() if (isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ()) else arr

def load_pca_model_npz(model_dir):
    """Load PCA bits from pca_model.npz with correct types."""
    npz = np.load(os.path.join(model_dir, "pca_model.npz"), allow_pickle=True)
    x = npz["x_common"]
    scores = npz["scores"]
    loadings = npz["loadings"]
    mean_vec = npz["mean_vector"]
    explained_var = npz["explained_var"]
    cum_var = npz["cum_var"]
    col_names = npz["col_names"].tolist()

    use_scaler = _as_bool(npz["use_scaler"])
    # centering = _as_bool(npz["centering"])
    scaler_mean = _maybe_none(npz["scaler_mean"])
    scaler_std  = _maybe_none(npz["scaler_std"])
            
    scaler = None 
    if use_scaler: 
        scaler = StandardScaler(with_mean=True, with_std=True) 
        # Attach learned params 
        scaler.mean_ = np.asarray(scaler_mean, dtype=float) 
        scaler.scale_ = np.asarray(scaler_std, dtype=float)

    return {
        "x": x,
        "scores": scores,
        "loadings": loadings,
        "mean": mean_vec,
        "explained_var": explained_var,
        "cum_var": cum_var,
        "col_names": col_names,
        "scaler": scaler,
        "use_scaler": use_scaler,
    }


def reconstruct_from_components(scores, loadings, mean_vector, comps_1based, scaler=None, add_mean=True):
    """
    scores:    (n_samples, n_components)
    loadings:  (n_components, n_features)
    mean_vector: (n_features,), in the SAME space as scores/loadings
    comps_1based: iterable of ints like [1,2,4]
    """
    idx = [int(c)-1 for c in comps_1based if 0 <= int(c)-1 < loadings.shape[0]]
    if not idx:
        raise ValueError("No valid PCs selected.")

    # 1) combine only selected PCs in PCA input space
    X_in = scores[:, idx] @ loadings[idx, :]

    # 2) add PCA mean (same space)
    if add_mean:
        X_in += mean_vector

    # 3) inverse-scale to raw units if scaler was used
    if scaler is not None:
        X_out = scaler.inverse_transform(X_in)
    else:
        X_out = X_in

    return X_out

def run_reconstruction(pc_list, save_folder, model_path=None, force_from_disk=False, add_mean=True):
    global LAST_PCA_MODEL

    if force_from_disk:
        model = load_pca_model_npz(model_path)
    else:
        if LAST_PCA_MODEL is not None:
            # build dict compatible with loader result
            model = {
                "x": LAST_PCA_MODEL["x"],
                "scores": LAST_PCA_MODEL["scores"],
                "loadings": LAST_PCA_MODEL["loadings"],
                "mean": getattr(LAST_PCA_MODEL["pca"], "mean_", 
                np.zeros(LAST_PCA_MODEL["loadings"].shape[1])),
                "col_names": LAST_PCA_MODEL["col_names"],
                "scaler": LAST_PCA_MODEL["scaler"],
                "use_scaler": LAST_PCA_MODEL["scaler"] is not None,
            }
        elif model_path:
            model = load_pca_model_npz(model_path)
        else:
            raise RuntimeError("No PCA model available (cache empty and no model_path).")

    # print("Readed list:")
    # print(pc_list)    
    comps = [int(c.strip()) for c in pc_list.split(",") if c.strip().isdigit()]
    
    # print("Extracted components:")
    # print(comps)
    
    recon = reconstruct_from_components(
        model["scores"], model["loadings"], 
        model["mean"], comps, model["scaler"],
        add_mean = add_mean
    )

    os.makedirs(save_folder, exist_ok=True)
    for name, y in zip(model["col_names"], recon):
        np.savetxt(os.path.join(save_folder, f"{name}_recon.txt"),
                   np.column_stack([model["x"], y]),
                   fmt=["%.6f", "%.6f"], delimiter="\t", comments="")
    return len(comps)

# ==============================================================
# Tkinter window
# ==============================================================
class PCA_Setup(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("PCA – Analysis & Reconstruction")
        self.resizable(False, False)

        # Variables
        self.data_type_var = tk.StringVar(value="Horiba")
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.n_comp_var = tk.IntVar(value=10)
        self.scaler_var = tk.BooleanVar(value=False)
        #self.centering_var = tk.BooleanVar(value=True)
        self.pc_list_var = tk.StringVar(value="1,2,3")
        self.model_dir_var = tk.StringVar()
        self.full_recon_var = tk.BooleanVar(value=True)

        pad = {'padx': 8, 'pady': 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        # --- PCA CALCULATION SECTION ---
        ttk.Label(frm, text="Data type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.data_type_var,
                     values=["Horiba", "Witec", "Default", "User"],
                     state="readonly").grid(row=0, column=1, columnspan=2, sticky="we")

        ttk.Label(frm, text="Input folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_dir_var, width=40).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._choose_input).grid(row=1, column=2)

        ttk.Label(frm, text="Output folder:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_dir_var, width=40).grid(row=2, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._choose_output).grid(row=2, column=2)

        ttk.Label(frm, text="# Components:").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frm, from_=1, to=500, textvariable=self.n_comp_var, width=8).grid(row=3, column=1, sticky="w")

        ttk.Checkbutton(frm, text="Scaling", variable=self.scaler_var).grid(row=4, column=0, columnspan=1, sticky="w")
        
        #ttk.Checkbutton(frm, text="Centering", variable=self.centering_var).grid(row=4, column=1, columnspan=1, sticky="w")

        ttk.Button(frm, text="Run PCA", command=self._run_pca).grid(row=4, column=2, pady=(5,10), sticky="e")

        # --- SEPARATOR ---
        ttk.Separator(frm, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky="we", pady=10)

        # --- RECONSTRUCTION SECTION ---
        ttk.Label(frm, text="Reconstruction PCs (e.g. 1,2,4):").grid(row=6, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.pc_list_var, width=20).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Full reconstruction (add mean)",
                variable=self.full_recon_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=(5,0))
        ttk.Button(frm, text="Reconstruct", command=self._run_reconstruction).grid(row=7, column=2, sticky="e")

        for i in range(3):
            frm.columnconfigure(i, weight=1)

        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _choose_input(self):
        d = filedialog.askdirectory(title="Select input folder")
        if d:
            self.input_dir_var.set(d)

    def _choose_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir_var.set(d)

    def _run_pca(self):
        input_folder = self.input_dir_var.get().strip()
        output_folder = self.output_dir_var.get().strip() or input_folder
        if not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Invalid input folder.")
            return

        loader_func = {
            "Horiba": Loading.load_horiba,
            "Witec": Loading.load_witec,
            "Default": Loading.load_default,
            "User": Loading.load_user
        }[self.data_type_var.get()]

        try:
            out = run_pca_calculation(loader_func, input_folder, output_folder,
                                      n_components=self.n_comp_var.get(),
                                      use_scaler=self.scaler_var.get()
                                      #centering=self.centering_var
                                      )
            messagebox.showinfo("Done", f"PCA completed.\nResults saved in:\n{out}")
        except Exception as e:
            messagebox.showerror("PCA error", str(e))

    def _run_reconstruction(self):
        pcs = self.pc_list_var.get().strip()
        save_dir = filedialog.askdirectory(title="Select folder for reconstructed spectra")
        if not save_dir:
            return
        try:
            n_used = run_reconstruction(pcs, save_dir, model_path=self.output_dir_var.get(), add_mean=self.full_recon_var.get())
            messagebox.showinfo("Reconstruction", f"Reconstruction done using PCs: {pcs}\nSaved to:\n{save_dir} and {n_used} actually used")
        except Exception as e:
            messagebox.showerror("Reconstruction error", str(e))


# ==============================================================
# Entry point for GUI
# ==============================================================
def open_pca_window(master=None):
    PCA_Setup(master)
