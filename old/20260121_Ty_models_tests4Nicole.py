import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from EXOSIMS.OpticalSystem.MHRS import broaden_and_resample,broaden
import os

def merge_regular_with_original(nc_wavelength_nm, nc_reflectance, delta_lambda_nm,tol_factor=0.25):
    """
    Merge a regularly sampled, interpolated spectrum with the original irregular spectrum.

    Parameters
    ----------
    nc_wavelength_nm : array-like
        Original wavelength array in nm (1D). Can be unsorted and non-uniform.
    nc_reflectance : array-like
        Reflectance values corresponding to `nc_wavelength_nm` (1D).
    delta_lambda_nm : float
        Step size (in nm) for the regular grid.
    tol_factor : float, optional
        Tolerance for considering a regular-grid wavelength a duplicate of an original point,
        expressed as a fraction of delta_lambda_nm. Default 0.25.

    Returns
    -------
    w_out : np.ndarray
        Merged wavelength array (increasing, 1D).
    r_out : np.ndarray
        Reflectance values aligned with `w_out`.
    """
    # --- Input checks ---
    w = np.asarray(nc_wavelength_nm, dtype=float).ravel()
    r = np.asarray(nc_reflectance, dtype=float).ravel()
    if w.size != r.size:
        raise ValueError("nc_wavelength_nm and nc_reflectance must have the same length.")
    if delta_lambda_nm <= 0:
        raise ValueError("delta_lambda_nm must be positive.")

    # --- Clean & sort original data, drop NaNs and duplicate wavelengths (keep first) ---
    m = np.isfinite(w) & np.isfinite(r)
    w, r = w[m], r[m]
    # order = np.argsort(w)
    # w, r = w[order], r[order]
    # Deduplicate exact duplicate wavelengths to ensure strictly increasing
    # uniq_w, uniq_idx = np.unique(w, return_index=True)
    # w, r = uniq_w, r[uniq_idx]

    # --- Build regular grid covering [min, max] inclusive ---
    wmin, wmax = w[0], w[-1]
    # Add a small epsilon to include wmax if it lands exactly on a step
    eps = np.finfo(float).eps
    w_reg = np.arange(wmin+delta_lambda_nm, wmax + delta_lambda_nm * (1 - 1e-12) + eps, delta_lambda_nm)

    # --- Interpolate original spectrum onto the regular grid ---
    # np.interp clamps to the edge values outside [wmin, wmax], which is OK since w_reg lies within.
    r_reg = np.interp(w_reg, w, r)
    # return w_reg, r_reg

    # --- Decide which regular-grid points to keep (avoid near-duplicates of original points) ---
    # Use searchsorted to find nearest original wavelength for each w_reg
    idx = np.searchsorted(w, w_reg)
    # Compute distance to nearest neighbor in original grid
    left_dist = np.abs(w_reg - w[np.clip(idx - 1, 0, len(w) - 1)])
    right_dist = np.abs(w_reg - w[np.clip(idx, 0, len(w) - 1)])
    min_dist = np.minimum(left_dist, right_dist)

    tol = tol_factor * delta_lambda_nm
    keep_reg = min_dist > tol

    # --- Merge: original points first (preserve exact originals), then selected regular points ---
    w_merged = np.concatenate([w, w_reg[keep_reg]])
    r_merged = np.concatenate([r, r_reg[keep_reg]])

    # --- Final sort (should already be nearly sorted) ---
    order = np.argsort(w_merged)
    w_out = w_merged[order]
    r_out = r_merged[order]

    return w_out, r_out


if __name__ == "__main__":
    ##### Parameters to update #####
    # Directory of Ty models
    dir2models = "/fast/jruffio/data/exosims/model_Ty/"
    R= 150000 # Broaden Ty's model to this spectral resolution
    lmin,lmax = 750,775 # nm, wavelength range of interest
    wv0 = np.arange(lmin,lmax,lmin/(4*R)) * u.nm # Define output wavelength grid
    # clouds_list = ["highcloud","lowcloud","clearsky"]
    clouds_list = ["highcloud"]
    ######

    # Run once to broaden and resample all models. This saves the models in csv files.
    if 1:
        for clouds in clouds_list:

            #### Model with all molecules ####
            fname = os.path.join(dir2models,"earth_maxres","earth_icrccm_hitran2020_"+clouds+"_50_100000cm-1_toa.rad")
            nc_data = np.loadtxt(fname, comments='#')
            nc_where_wvs = np.where((nc_data[:, 0] > lmin/1000.) * (nc_data[:, 0] < lmax/1000.))
            nc_data = nc_data[nc_where_wvs[0], :]

            # Split into named columns
            nc_wavelength_nm = nc_data[::-1, 0]*1000  * u.nm # column 1
            nc_wavenumber_cm1 = nc_data[::-1, 1]  # column 2
            nc_solar_flux_1au = nc_data[::-1, 2]  # column 3
            nc_earth_flux_toa = nc_data[::-1, 3]  # column 4
            nc_radiance_streams = nc_data[::-1, 4:]  # columns 5-8
            nc_reflectance = nc_earth_flux_toa / nc_solar_flux_1au  # Reflectance spectrum

            # plt.scatter(nc_wavelength_nm, nc_reflectance, label=f"Ty {clouds} (Original)",s=30,c="red",marker="o")

            nc_wavelength_nm2, nc_reflectance2 = merge_regular_with_original(nc_wavelength_nm.to(u.nm).value, nc_reflectance, lmin/(10*R), tol_factor=lmin/(10*R)/2.)
            nc_wavelength_nm2 = nc_wavelength_nm2 * u.nm
            nc_reflectance_R_all = broaden_and_resample(wv0, nc_wavelength_nm2, nc_reflectance2, R, n_jobs=8, broaden_pixel=False)

            # Save broadened spectrum in csv file
            np.savetxt(fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad"), np.column_stack((wv0.to(u.nm).value, nc_reflectance_R_all)), fmt="%.9e",)

            # plt.plot(wv0, nc_reflectance_R_all, label=f"Ty {clouds} All (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

            #### Model with only H2O ####
            fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_"+clouds+"_h2o_50_100000cm-1_toa.rad"
            nc_data = np.loadtxt(fname, comments='#')
            nc_where_wvs = np.where((nc_data[:, 0] > lmin/1000.) * (nc_data[:, 0] < lmax/1000.))
            nc_data = nc_data[nc_where_wvs[0], :]

            # Split into named columns
            nc_wavelength_nm = nc_data[::-1, 0]*1000  * u.nm # column 1
            nc_wavenumber_cm1 = nc_data[::-1, 1]  # column 2
            nc_solar_flux_1au = nc_data[::-1, 2]  # column 3
            nc_earth_flux_toa = nc_data[::-1, 3]  # column 4
            nc_radiance_streams = nc_data[::-1, 4:]  # columns 5-8
            nc_reflectance = nc_earth_flux_toa / nc_solar_flux_1au  # Reflectance spectrum

            # plt.scatter(nc_wavelength_nm, nc_reflectance, label=f"Ty {clouds} H2O (Original)",s=10)

            nc_wavelength_nm2, nc_reflectance2 = merge_regular_with_original(nc_wavelength_nm.to(u.nm).value, nc_reflectance, lmin/(10*R), tol_factor=lmin/(10*R)/2.)
            nc_wavelength_nm2 = nc_wavelength_nm2 * u.nm
            nc_reflectance_R_h2o = broaden_and_resample(wv0, nc_wavelength_nm2, nc_reflectance2, R, n_jobs=8, broaden_pixel=False)

            # Save broadened spectrum in csv file
            np.savetxt(fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad"), np.column_stack((wv0.to(u.nm).value, nc_reflectance_R_h2o)), fmt="%.9e",)

            # plt.plot(wv0, nc_reflectance_R_h2o, label=f"Ty {clouds} H2O (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

            ### Model with only O2 ####
            fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_"+clouds+"_o2_50_100000cm-1_toa.rad"
            nc_data = np.loadtxt(fname, comments='#')
            nc_where_wvs = np.where((nc_data[:, 0] > lmin/1000.) * (nc_data[:, 0] < lmax/1000.))
            nc_data = nc_data[nc_where_wvs[0], :]

            # Split into named columns
            nc_wavelength_nm = nc_data[::-1, 0]*1000  * u.nm # column 1
            nc_wavenumber_cm1 = nc_data[::-1, 1]  # column 2
            nc_solar_flux_1au = nc_data[::-1, 2]  # column 3
            nc_earth_flux_toa = nc_data[::-1, 3]  # column 4
            nc_radiance_streams = nc_data[::-1, 4:]  # columns 5-8
            nc_reflectance = nc_earth_flux_toa / nc_solar_flux_1au  # Reflectance spectrum

            # plt.scatter(nc_wavelength_nm, nc_reflectance, label=f"Ty {clouds} O2 (Original)",s=10)

            nc_wavelength_nm2, nc_reflectance2 = merge_regular_with_original(nc_wavelength_nm.to(u.nm).value, nc_reflectance, lmin/(10*R), tol_factor=lmin/(10*R)/2.)
            nc_wavelength_nm2 = nc_wavelength_nm2 * u.nm
            nc_reflectance_R_o2 = broaden_and_resample(wv0, nc_wavelength_nm2, nc_reflectance2, R, n_jobs=8, broaden_pixel=False)

            # Save broadened spectrum in csv file
            np.savetxt(fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad"), np.column_stack((wv0.to(u.nm).value, nc_reflectance_R_o2)), fmt="%.9e",)

            # plt.plot(wv0, nc_reflectance_R_o2, label="Ty no cloud O2 (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

    # After the models have been broadened and saved, plot the spectra directly from the csv files
    for clouds in clouds_list:
        fname = os.path.join(dir2models,"earth_maxres","earth_icrccm_hitran2020_"+clouds+"_50_100000cm-1_toa.rad")
        fname_broadened = fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad")
        data = np.loadtxt(fname_broadened)
        wvs = data[:,0]
        spec = data[:,1]
        plt.plot(wvs, spec, label=f"Ty {clouds} All (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

        fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_"+clouds+"_h2o_50_100000cm-1_toa.rad"
        fname_broadened = fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad")
        data = np.loadtxt(fname_broadened)
        wvs = data[:,0]
        spec = data[:,1]
        plt.plot(wvs, spec, label=f"Ty {clouds} H2O (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

        fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_"+clouds+"_o2_50_100000cm-1_toa.rad"
        fname_broadened = fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad")
        data = np.loadtxt(fname_broadened)
        wvs = data[:,0]
        spec = data[:,1]
        plt.plot(wvs, spec, label=f"Ty {clouds} O2 (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

    ##### Also compare to the calibrated Earth spectrum
    filename = os.path.join(dir2models,"earth_quadrature_R140.csv")  # replace with your file path
    data = np.genfromtxt(filename, delimiter=",", names=True)
    print(data.dtype.names)
    # Access columns by name
    lam = data['lam_um']*1000  # Convert um to nm
    dlam = data['dlam_um']
    Aapp = data['Aapp']
    AgPhi = data['AgPhi']
    plt.plot(lam, AgPhi, label="Earth calibrated R=140",linestyle="-",alpha=1,linewidth=2)

    plt.xlim([lmin,lmax])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend(loc="upper left")


    #### Example computing template matching SNR for O2 detection ####
    SNR_per_bin = 0.1  # Example SNR per spectral bin
    fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_"+clouds+"_o2_50_100000cm-1_toa.rad"
    fname_broadened = fname.replace(".rad",f"_{lmin}_{lmax}_R{R}.rad")
    data = np.loadtxt(fname_broadened)
    wvs = data[:,0]
    spec_O2 = data[:,1]
    max_template = np.nanmax(spec_O2)
    sigma_bin = max_template / SNR_per_bin

    # Template matching "mean" SNR.
    SNR_template = np.sqrt(np.nansum(spec_O2**2))/sigma_bin
    print(f'Template matching SNR for O2 detection at R={R} and "SNR per bin"={SNR_per_bin} is {SNR_template:.2f}')


    plt.show()