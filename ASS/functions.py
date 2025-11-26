# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:28:41 2025

@author: Martin Jindra
"""
import numpy as np
from scipy.special import voigt_profile

def lorentzian(x, intensity, center, fwhm):
    """
    Lorentzian peak function.

    Parameters:
    x : array-like
        Independent variable (e.g., wavenumber, frequency).
    intensity : float
        Area under the curve (proportional to real intensity).
    center : float
        Peak position (center of the Lorentzian).
    fwhm : float
        Full Width at Half Maximum, controlling the peak's width.
    """
    width = fwhm/2
    return intensity/(np.pi * width * (1+((x-center)/width)**2))

def gaussian(x, intensity, center, fwhm):
    """
    Gaussian peak function.

    Parameters:
    x : array-like
        Independent variable (e.g., wavenumber, frequency).
    intensity : float
        Area under the curve (proportional to real intensity).
    center : float
        Peak position (mean of the Gaussian).
    fwhm : float
        Full Width at Half Maximum, used to calculate standard deviation.
    """
    stddev = fwhm/(2*np.sqrt(2*np.log(2)))
    amplitude = intensity / (stddev * np.sqrt(2 * np.pi))
    return amplitude * np.exp(-((x - center) ** 2) / (2 * stddev ** 2))

def voigt(x, intensity, center, sigma, gamma):
    """
    Voigt peak function (convolution of Gaussian and Lorentzian).

    Parameters:
    x : array-like
        Independent variable (e.g., wavenumber, frequency).
    intensity : float
        Area under the curve (proportional to real intensity).
    center : float
        Peak position (center of the Voigt profile).
    sigma : float
        Standard deviation of the Gaussian contribution.
    gamma : float
        Half-width at half-maximum (HWHM) of the Lorentzian contribution.
    """
    return intensity * voigt_profile(x - center, sigma, gamma)

def asym_lorentzian(x, intensity, center, fwhm, alpha):
    """
    Asymmetrical Lorentzian peak function.

    Parameters:
    x : array-like
        Independent variable.
    intensity : float
        Area under the curve (real intensity).
    center : float
        Peak position.
    fwhm : float
        Full Width at Half Maximum (symmetric case when alpha=0).
    alpha : float
        Asymmetry parameter (0 = symmetric Lorentzian).
    """
    width = (fwhm / 2) * (1 + alpha * (x - center))
    width = np.maximum(width, 1e-10)  # Prevent division by zero
    return intensity / (np.pi * width * (1 + ((x - center) / width) ** 2))

# def fano(x, intensity, center, fwhm, q):
#     """
#     Fano resonance profile.

#     Parameters:
#     x : array-like
#         Independent variable.
#     intensity : float
#         Area under the curve (real intensity).
#     center : float
#         Resonance peak position.
#     fwhm : float
#         Full Width at Half Maximum (controls resonance width).
#     q : float
#         Fano asymmetry parameter (q=0 gives symmetric dip).
#     """
#     epsilon = (x - center) / (fwhm / 2)
#     return intensity * ((q + epsilon) ** 2) / (1 + epsilon ** 2)

# def fano(x, intensity, center, fwhm, q):
#     """
#     Breit–Wigner–Fano (BWF) resonance profile.

#     Parameters
#     ----------
#     x : array-like
#         Independent variable (e.g., frequency or Raman shift).
#     amplitude/intenisty : float
#         Peak amplitude scaling factor.
#     center : float
#         Resonance position (ω₀).
#     fwhm : float
#         Full width at half maximum (Γ).
#     q : float
#         Fano asymmetry parameter (q → ∞ → Lorentzian).
#     """
#     epsilon = 2 * (x - center) / fwhm
#     return intensity * ((q + epsilon)**2 / (1 + epsilon**2))

def fano(x, intensity, center, fwhm, Q):
    """
    Breit–Wigner–Fano (BWF) resonance profile.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., frequency or Raman shift).
    amplitude/intenisty : float
        Peak amplitude scaling factor.
    center : float
        Resonance position (ω₀).
    fwhm : float
        Full width at half maximum (Γ).
    Q : float
        Really 1/q parameter based on publication - https://doi.org/10.1103/PhysRevB.90.245140
        Q (1/q) is used as fitting parameter instead of q
            otherwise the value was on completely other scale then the rest 
            and the fitting procedure was not able to optimize it properly
        Fano asymmetry parameter (q → ∞ (1/q → 0) → Lorentzian).
    """
    s = (x - center) / fwhm
    return intensity * ((1+s*Q)**2)/(1+s**2)


def double_lorentzian(x, intensity1, center1, intensity2, center2, fwhm):
    """
    Double-Lorentzian peak function.

    Parameters:
    x : array-like
        Independent variable (e.g., wavenumber, frequency).
    int1 : float
        Area of the fist orsilator.
    cen1 : float
        Position of the fist orsilator. (center of the Lorentzian).
    int2 : float
        Area of the second orsilator.
    cen2 : float
        Position of the second orsilator. (center of the Lorentzian).
    fwhm : float
        Shared Full Width at Half Maximum, controlling the peak's width.
    """
    width = fwhm/2
    return intensity1/(np.pi * width * (1+((x-center1)/width)**2)) + intensity2/(np.pi * width * (1+((x-center2)/width)**2))

def linear(x, slope, intercept):
    """
    Linear function for baseline correction or trend modeling.

    Parameters:
    x : array-like
        Independent variable.
    slope : float
        The slope of the linear function (rate of change).
    intercept : float
        The y-intercept, where the line crosses the y-axis.
    """
    return slope*x+intercept

def sigmoid(x, amplitude, center, steepness, baseline):
    """
    Sigmoid function for asymmetric backgrounds.

    Parameters:
    x : array-like
        Independent variable (e.g., binding energy).
    amplitude : float
        Step height (difference between two baseline levels).
    center : float
        Transition midpoint (where the step occurs).
    steepness : float
        Controls the sharpness of the transition (small = sharp step, large = smooth transition).
    baseline : float
        Baseline offset (lower background level).
    """
    return amplitude / (1 + np.exp(-(x - center) / steepness)) + baseline

# Model dictionary
model_dict = {
    "Gaussian": {
        "func": gaussian,
        "params": ["intensity", "center", "fwhm"],
    },
    "Lorentzian": {
        "func": lorentzian,
        "params": ["intensity", "center", "fwhm"],
    },
    "Voigt": {
        "func": voigt,
        "params": ["intensity", "center", "sigma", "gamma"],
    },
    "Asym_Lorentzian": {
        "func": asym_lorentzian,
        "params": ["intensity", "center", "fwhm", "alpha"],
    },
    "Fano": {
        "func": fano,
        "params": ["intensity", "center", "fwhm", "1/q"],
    },
    "Double Lorentz": {
        "func": double_lorentzian,
        "params": ["intensity #1", "center #1", "intensity #2", "center #2", "fwhm"],
    },
    "Linear": {
        "func": linear,
        "params": ["slope", "intercept"],
    },
    "Sigmoid": {
        "func": sigmoid,
        "params": ["amplitude", "center", "steepness", "baseline"],
    }
}