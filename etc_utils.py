
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from joblib import Parallel, delayed

import sys
import math

def safe_value(x):
    """Return x if it is finite, otherwise return the maximum float."""
    return x if math.isfinite(x) else sys.float_info.max

def gaussian_broaden(
    wavelengths, spectrum, resolution_array, window=5, n_jobs=-1
):
    """
    NaN-resistant, parallel Gaussian broadening on non-uniform grid with resolution input.
    Pads the edges by extending the last spectrum value to ensure proper broadening even at low resolution.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array (1D, non-uniform allowed).
    spectrum : ndarray
        Flux values (1D), may contain NaNs.
    resolution_array : ndarray
        Resolving power R = lam / dlam at each wavelength.
    window : float
        Width of convolution window in sigma (default 5).
    n_jobs : int
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    broadened_spectrum : ndarray
        Broadened 1D spectrum.
    """
    N = len(wavelengths)
    assert N == len(spectrum) == len(resolution_array)

    # Estimate maximum sigma to compute required padding in wavelength units
    fwhm_max = np.max(wavelengths / resolution_array)
    sigma_max = fwhm_max / 2.3548
    delta_pad = window * sigma_max

    # Pad on both sides
    lam_start, lam_end = wavelengths[0], wavelengths[-1]
    dlam = np.median(np.diff(wavelengths))

    # Estimate how many points to add on each side
    n_pad = int(np.ceil(delta_pad / dlam))
    left_pad_wl = lam_start - dlam * np.arange(n_pad, 0, -1)
    right_pad_wl = lam_end + dlam * np.arange(1, n_pad + 1)

    # Create extended arrays
    wl_pad = np.concatenate([left_pad_wl, wavelengths, right_pad_wl])
    sp_pad = np.concatenate([
        np.full(n_pad, spectrum[np.where(np.isfinite(spectrum))[0][0]]),  # Extend first valid value
        spectrum,
        np.full(n_pad, spectrum[np.where(np.isfinite(spectrum))[0][-1]])  # Extend last valid value
    ])

    wl_pad_diff = np.diff(wl_pad, prepend=2 * wl_pad[0] - wl_pad[1])


    def broaden_at_index(i):
        lam_i = wavelengths[i]
        R_i = resolution_array[i]
        fwhm_i = lam_i / R_i
        sigma_i = fwhm_i / 2.3548
        delta = window * sigma_i

        mask = np.abs(wl_pad - lam_i) < delta
        wl_local = wl_pad[mask]
        wl_local_diff = wl_pad_diff[mask]
        sp_local = sp_pad[mask]
        valid = np.isfinite(sp_local)


        if not np.any(valid):
            return np.nan

        weights = 1/np.sqrt(2*np.pi*sigma_i**2)*np.exp(-0.5 * ((wl_local - lam_i) / sigma_i) ** 2)*wl_local_diff
        weights = weights[valid]

        return np.sum(weights * sp_local[valid])

    broadened = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(broaden_at_index)(i) for i in range(N)
    )
    return np.array(broadened)


def subtract_continuum_envelop(wavelength, albedo, n_control=15, penalty=10):
    """
    Fit and subtract the continuum from a reflected light spectrum using an
    asymmetric loss function that penalizes continuum estimates below the data.

    Parameters
    ----------
    wavelength : np.ndarray
        1D array of wavelength values (must be sorted in increasing order).
    albedo : np.ndarray
        1D array of albedo values at the corresponding wavelengths.
    n_control : int, optional
        Number of control points for the cubic spline fitting the continuum.
        A smaller number results in a smoother continuum. Default is 15.
    penalty : float, optional
        weight penalizing the chi2 of the part of the data below the model compared to the data above the model. Default is 1e6.

    Returns
    -------
    continuum : np.ndarray
        The fitted continuum spectrum (upper envelope).
    albedo_cont_subtracted : np.ndarray
        The original albedo with the continuum subtracted.
    x_knots : np.ndarray
        Wavelengths of the spline control points.
    y_fit : np.ndarray
        Fitted albedo values at the control points (for plotting/debugging).
    """
    # Define control point locations evenly in wavelength space
    x_knots = np.linspace(wavelength.min(), wavelength.max(), n_control)

    # Initial guess: interpolate data onto knots
    initial_y = np.interp(x_knots, wavelength, albedo)

    # Asymmetric loss function
    def loss_func(y_control):
        spline = CubicSpline(x_knots, y_control, bc_type='natural')
        model = spline(wavelength)
        residuals = model - albedo

        # Penalize only positive residuals (model above data), infinite cost otherwise
        if np.any(model < albedo):
            out = penalty*np.sum(residuals[residuals < 0]) ** 2 + np.sum((residuals[residuals > 0]) ** 2)
            return safe_value(out)
        return np.sum((residuals)**2)

    # Bounds: continuum should not fall below data at control points
    bounds = [(max(a, 0), 1.0) for a in np.interp(x_knots, wavelength, albedo)]

    # Fit the continuum
    result = minimize(loss_func, initial_y, method='L-BFGS-B', bounds=bounds)
    y_fit = result.x

    # Evaluate the final spline
    spline = CubicSpline(x_knots, y_fit, bc_type='natural')
    envelop = spline(wavelength)
    albedo_cont_subtracted = albedo - envelop

    return envelop, albedo_cont_subtracted, x_knots, y_fit