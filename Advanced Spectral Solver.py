# -*- coding: utf-8 -*-
"""
ASS stater
Author: Martin Jindra
"""
import os, json, base64, requests
from pathlib import Path
from datetime import datetime, UTC, timedelta
import tkinter as tk
from tkinter import messagebox, filedialog
from cryptography.hazmat.primitives import serialization
from ASS.gui import MainWindow

# ==============================================================
# 🧩 Embedded data
# ==============================================================

# # Embedded Ed25519 public key (base64 only, no PEM header/footer)
# PUBLIC_KEY_B64 = b"MCowBQYDK2VwAyEAV9MBq2WFk4ARF1kg3VeEYd2gLkZH2hckEsODVLn9Xrc="

# Default fallback "Free" licence (signed)
FALLBACK_LICENCE = {
    "payload": {
        "user": "General User",
        "licence_type": "Free",
        "valid_until": "3025-12-30",
        "created_at": "2025-11-06T13:25:42.382157Z",
    },
    "signature": "1dO+KxBuiVGnNCXyPcFv3iNXF5OmzV9J6xD26zCs5fGsCCBuCXPf9IzxO76H8BjGN2NcdSNoirTnzMEfcAXFDg=="
}

DEFAULT_PLOT_CONFIG = {
    "X axis label": None,
    "Y axis label": None,
    "Raw spectrum label": None,
    "Raw spectrum color": "blue",
    "Compare spectrum offset": 0,
    "Compare spectrum label": None,
    "Compare spectrum color": "green",
    "Plot title": None,
    "Composite model color": "red",
    "Residual plot visibility": True,
}

DEFAULT_USER_LOADER = {
    "file_type": "txt",
    "separator": "\\s+",
    "skip_rows": 40,
    "usecols": [0, 1],
    "encoding": "ansi",
}

# ==============================================================
# 🔍 Licence Verification
# ==============================================================

def load_embedded_public_key():
    pem = (
        b"-----BEGIN PUBLIC KEY-----\n"
        b"MCowBQYDK2VwAyEAV9MBq2WFk4ARF1kg3VeEYd2gLkZH2hckEsODVLn9Xrc=\n"
        b"-----END PUBLIC KEY-----\n"
    )
    return serialization.load_pem_public_key(pem)

def verify_license_object(licence: dict) -> dict:
    """
    Verify the structure, signature, and expiry of the licence.
    Returns the payload if valid; raises ValueError otherwise.
    """
    if "payload" not in licence or "signature" not in licence:
        raise ValueError("Invalid licence structure.")

    payload = licence["payload"]
    signature = base64.b64decode(licence["signature"])

    # Prepare canonical JSON bytes for signature verification
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    pk = load_embedded_public_key()
    try:
        pk.verify(signature, payload_bytes)
    except Exception:
        raise ValueError("Licence signature invalid (file tampered or corrupted).")
    
    # --- Expiry check ---
    val_str = payload["valid_until"].rstrip("Z")
    valid_until = datetime.fromisoformat(val_str)
    now_utc = datetime.now(UTC)
    
    days_left = (valid_until.date() - now_utc.date()).days
    
    if days_left < 0:
        raise ValueError(f"Licence expired on {payload['valid_until']}")
    elif days_left < 30:
        messagebox.showinfo(
            "Licence Expiry Notice",
            f"Your licence will expire in {days_left} day{'s' if days_left != 1 else ''} "
            f"({payload['valid_until']}).\n\n"
            "Please consider renewing your licence soon."
        )

    return payload


SERVER_URL = "https://ass-licence-server.onrender.com/check"  # your live server
TIMEOUT = 120  # seconds

# def verify_with_server(licence_data):
#     """Check licence signature on the online activation server."""
#     sig = licence_data.get("signature")
#     if not sig:
#         return {"status": "invalid", "message": "Missing signature."}
#     try:
#         r = requests.post(SERVER_URL, json={"signature": sig}, timeout=TIMEOUT)
#         if r.status_code == 200:
#             return r.json()
#         return {"status": "error", "message": f"HTTP {r.status_code}"}
#     except requests.exceptions.RequestException:
#         return None  # offline fallback

def verify_with_server(licence_data, fallback_licence):
    """
    Verifies the licence against the remote server.
    Returns either the verified licence_data or FALLBACK_LICENCE["payload"].
    Also displays appropriate user messages.
    """
    sig = licence_data.get("signature")
    if not sig:
        messagebox.showerror("Invalid Licence", "Licence is missing its signature field.")
        return fallback_licence["payload"]

    try:
        r = requests.post(SERVER_URL, json={"signature": sig}, timeout=TIMEOUT)
        if r.status_code != 200:
            messagebox.showerror("Server Error", f"Unexpected response (HTTP {r.status_code}).")
            return fallback_licence["payload"]

        resp = r.json()
        status = resp.get("status", "error")

        if status == "valid":
            messagebox.showinfo("Licence Activated", "Licence verified online and activated.")
            return licence_data["payload"]

        elif status == "already_used":
            messagebox.showerror("Licence Already Used",
                                 "This licence has already been activated.\nStarting with Free licence.")
            return fallback_licence["payload"]

        elif status == "invalid":
            messagebox.showerror("Invalid Licence",
                                 "This licence is not recognised by the activation server.")
            return fallback_licence["payload"]

        else:
            messagebox.showerror("Server Error", f"Unexpected response: {resp}")
            return fallback_licence["payload"]

    except requests.exceptions.RequestException:
        messagebox.showwarning("Offline",
                               "For licence activation you need to be online.\nStarting with Free licence.")
        return fallback_licence["payload"]


# ==============================================================
# 🧠 Licence Logic
# ==============================================================

def check_licence():
    """
    Main licence validation routine.
    Ensures that the app can start with a valid licence
    (either loaded or fallback Free).
    """
    root = tk.Tk()
    root.withdraw()  # hide root until the GUI is ready

    config_dir = Path(os.getenv("APPDATA", Path.home())) / "ASS"
    config_dir.mkdir(parents=True, exist_ok=True)
    licence_path = config_dir / "licence.json"

    licence_data = None
    payload = None

    # ----------------------------------------------------------
    # CASE 1: Licence exists → verify it
    # ----------------------------------------------------------
    if licence_path.exists():
        try:
            with open(licence_path, "r", encoding="utf-8") as f:
                licence_data = json.load(f)
            payload = verify_license_object(licence_data)

            # Valid licence
            if payload["licence_type"] == "Free":
                if messagebox.askyesno(
                    "Free Licence Detected",
                    "You are using the Free licence.\nWould you like to load a Pro licence?\n(server verification could take some time)"
                ):
                    fn = filedialog.askopenfilename(
                        title="Select Licence File",
                        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
                    )
                    if fn:
                        try:
                            with open(fn, "r", encoding="utf-8") as f:
                                new_lic = json.load(f)
                            new_payload = verify_license_object(new_lic)
                            
                            # Add the check with server here - if it passes save the licence, if not inform user and continue with free licence
                            
                            new_payload = verify_with_server(new_lic, FALLBACK_LICENCE)
                            
                            # # Save new licence
                            # with open(licence_path, "w", encoding="utf-8") as out:
                            #     json.dump(new_lic, out, indent=2, ensure_ascii=False)
                            # messagebox.showinfo("Licence Updated", f"Licence activated for: {new_payload['user']}")
                            # payload = new_payload
                            
                            if new_payload == FALLBACK_LICENCE["payload"]:
                                payload = FALLBACK_LICENCE["payload"]
                            else:
                                with open(licence_path, "w", encoding="utf-8") as out:
                                    json.dump(new_lic, out, indent=2, ensure_ascii=False)
                                messagebox.showinfo("Licence Updated", f"Licence activated for: {new_payload['user']}")
                                payload = new_payload
                                
                        except Exception as e:
                            messagebox.showwarning("Invalid Licence", f"{e}\nContinuing with Free licence.")
            else:
                print(f"✅ Valid licence for {payload['user']} ({payload['licence_type']})")

        except Exception as e:
            messagebox.showwarning("Licence Error", f"{e}\nThe licence file will be replaced with a Free licence.")
            licence_data = FALLBACK_LICENCE
            with open(licence_path, "w", encoding="utf-8") as f:
                json.dump(licence_data, f, indent=2, ensure_ascii=False)
            payload = licence_data["payload"]

    # ----------------------------------------------------------
    # CASE 2: Licence file missing → ask user
    # ----------------------------------------------------------
    else:
        if messagebox.askyesno("Licence Missing", "No licence found.\nWould you like to load one now?"):
            fn = filedialog.askopenfilename(
                title="Select Licence File",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
            )
            if fn:
                try:
                    with open(fn, "r", encoding="utf-8") as f:
                        new_lic = json.load(f)
                    payload = verify_license_object(new_lic)
                    
                    # Add the check with server here - if it passes save the licence, if not inform user and continue with free licence
                    
                    payload = verify_with_server(new_lic, FALLBACK_LICENCE)

                    # Save whichever licence is active (real or fallback)
                    if payload == FALLBACK_LICENCE["payload"]:
                        with open(licence_path, "w", encoding="utf-8") as f:
                            json.dump(FALLBACK_LICENCE, f, indent=2, ensure_ascii=False)
                    else:
                        with open(licence_path, "w", encoding="utf-8") as out:
                            json.dump(new_lic, out, indent=2, ensure_ascii=False)
                        messagebox.showinfo("Licence Activated", f"Licence loaded for: {payload['user']}")
                    
                    # with open(licence_path, "w", encoding="utf-8") as out:
                    #     json.dump(new_lic, out, indent=2, ensure_ascii=False)
                    # messagebox.showinfo("Licence Activated", f"Licence loaded for: {payload['user']}")
                except Exception as e:
                    messagebox.showwarning("Invalid Licence", f"{e}\nStarting with Free licence.")
                    licence_data = FALLBACK_LICENCE
                    with open(licence_path, "w", encoding="utf-8") as f:
                        json.dump(licence_data, f, indent=2, ensure_ascii=False)
                    payload = licence_data["payload"]
            else:
                messagebox.showinfo("Licence Required", "No licence selected. Starting with Free licence.")
                with open(licence_path, "w", encoding="utf-8") as f:
                    json.dump(FALLBACK_LICENCE, f, indent=2, ensure_ascii=False)
                payload = FALLBACK_LICENCE["payload"]
        else:
            # messagebox.showerror("Licence Required", "Application cannot start without a licence.")
            # root.destroy()
            # raise SystemExit(1)
            messagebox.showinfo("Licence Required", "No licence selected. Starting with Free licence.")
            with open(licence_path, "w", encoding="utf-8") as f:
                json.dump(FALLBACK_LICENCE, f, indent=2, ensure_ascii=False)
            payload = FALLBACK_LICENCE["payload"]

    print(f"🎫 Using licence: {payload['licence_type']} ({payload['user']}) valid until {payload['valid_until']}")
    root.destroy()
    return payload


# ==============================================================
# ⚙️ Config Initialization
# ==============================================================

def ensure_user_config():
    # 1. Where configs live
    config_dir = Path(os.getenv("APPDATA", Path.home())) / "ASS"
    config_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define file paths
    plot_path = config_dir / "plot_config.json"
    loader_path = config_dir / "user_loader.json"

    # 3. If missing, create them
    if not plot_path.exists():
        with open(plot_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PLOT_CONFIG, f, indent=4)
        print(f"🆕 Created default {plot_path}")

    if not loader_path.exists():
        with open(loader_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_USER_LOADER, f, indent=4)
        print(f"🆕 Created default {loader_path}")

    print(f"✅ Config folder ready: {config_dir}")
    
if __name__ == "__main__":
    licence_payload = check_licence()
    ensure_user_config()
    app = MainWindow(licence_payload)  # you can pass licence info to control features
    # app = MainWindow()
    app.run()