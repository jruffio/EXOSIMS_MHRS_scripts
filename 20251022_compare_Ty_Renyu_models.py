import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from EXOSIMS.OpticalSystem.MHRS import broaden_and_resample,broaden


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

# plt.figure()
# plt.subplot(1, 2, 1)
phi = 0.318
R= 150000
lmin,lmax = 400,1500
wv0 = np.arange(lmin,lmax,lmin/(4*R)) * u.nm

if 0: # high clouds
    # index 0: Wavelength
    # index 1: All
    # index 2: H2O
    # index 3: CO2
    # index 4: N2O
    # index 5: CH4
    # index 6: O2
    # index 7: O3
    # pl_template_name = ["all","H2O","CO2","N2O","CH4","O2","O3"]
    ## Load the albedo spectral model
    filename_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_HighCloud_UltraRes.dat'
    hc_data_broad = np.loadtxt(filename_broad, dtype=float)
    hc_where_wvs = np.where((hc_data_broad[:,0]>200)*(hc_data_broad[:,0]<1500))
    hc_data_broad = hc_data_broad[hc_where_wvs[0],:]
    hc_waves = hc_data_broad[:, 0] * u.nm  # or u.AA, u.nm, etc.
    hc_all_template = hc_data_broad[:, 1]  # unitless reflectance
    hc_H2O_template = hc_data_broad[:, 2] - hc_data_broad[:, 4]  # unitless reflectance
    hc_O2_template = hc_data_broad[:, 6] - hc_data_broad[:, 4]  # unitless reflectance

    hc_all_template = broaden_and_resample(wv0, hc_waves, hc_all_template, R, n_jobs=8, broaden_pixel=False)

    plt.plot(wv0,hc_all_template*phi,label="Renyu high clouds",linestyle="--",alpha=0.5,linewidth=1)#0.2/0.556*
    # plt.plot(hc_waves,hc_H2O_template,label="Renyu high clouds H2O")
    # plt.plot(hc_waves,hc_O2_template,label="Renyu high clouds O2")

if 0:  # low clouds
    filename_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_LowCloud_UltraRes1.dat'
    lc_data_broad = np.loadtxt(filename_broad, dtype=float)
    lc_where_wvs = np.where((lc_data_broad[:, 0] > 200) * (lc_data_broad[:, 0] < 1500))
    lc_data_broad = lc_data_broad[lc_where_wvs[0], :]
    lc_waves = lc_data_broad[:, 0] * u.nm  # or u.AA, u.nm, etc.
    lc_all_template = lc_data_broad[:, 1]  # unitless reflectance
    lc_H2O_template = lc_data_broad[:, 2] - lc_data_broad[:, 4]  # unitless reflectance
    lc_O2_template = lc_data_broad[:, 6] - lc_data_broad[:, 4]  # unitless reflectance

    lc_all_template = broaden_and_resample(wv0, lc_waves, lc_all_template, R, n_jobs=8, broaden_pixel=False)

    plt.plot(wv0, lc_all_template*phi, label="Renyu low clouds",linestyle="--",alpha=0.5,linewidth=1)
    # plt.plot(lc_waves, lc_H2O_template, label="Renyu low clouds H2O")
    # plt.plot(lc_waves, lc_O2_template, label="Renyu low clouds O2")

if 0:  # No clouds
    filename_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_NoCloud_UltraRes.dat'
    nc_data_broad = np.loadtxt(filename_broad, dtype=float)
    nc_where_wvs = np.where((nc_data_broad[:, 0] > 200) * (nc_data_broad[:, 0] < 1500))
    nc_data_broad = nc_data_broad[nc_where_wvs[0], :]
    nc_waves = nc_data_broad[:, 0] * u.nm  # or u.AA, u.nm, etc.
    nc_all_template = nc_data_broad[:, 1]  # unitless reflectance
    nc_H2O_template = nc_data_broad[:, 2] - nc_data_broad[:, 4]  # unitless reflectance
    nc_O2_template = nc_data_broad[:, 6] - nc_data_broad[:, 4]  # unitless reflectance

    nc_all_template = broaden_and_resample(wv0, nc_waves, nc_all_template, R, n_jobs=8, broaden_pixel=False)

    plt.plot(wv0, nc_all_template*phi, label="Renyu no clouds")
    # plt.plot(nc_waves, nc_H2O_template, label="Renyu no clouds H2O")
    # plt.plot(nc_waves, nc_O2_template, label="Renyu no clouds O2")

if 0: #mixed model
    mixed_model_Renyu = 0.25*hc_all_template+0.25*lc_all_template+0.5*nc_all_template
    plt.plot(wv0, mixed_model_Renyu*phi, label="Renyu mixed model",linestyle="-",alpha=1,linewidth=2)

    plt.xlim([600,1000])
    plt.ylim([0,0.35])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend(loc="upper left")
    plt.subplot(1, 2, 2)


if 0:
    # From Ty Robinson:
    # - A first spectrum is done, and I've placed it here. It would be good to get eyes on this before I generate too many more spectra.
    # - Details  clearsky Earth; gases: H2O, CO2, O3, N2O, CO, CH4, O2; surface reflectance is ocean-like
    # - Columns are:
    # - 1 - wavelength (um)
    # - 2 - wavenumber (cm**-1)
    # - 3 - incident solar flux at 1 au (w m**-2 um**-1)
    # - 4 - top-of-atmosphere Earth flux (w m**-2 um**-1)
    # - 5/8 - radiance streams (w m**-2 um**-1 sr**-1)
    # -
    # - To get a reflectance spectrum, divide column 4 by column 3.
    # - To get Fp/Fs, do: Fp/Fs = (col 4) / (col 3) x (Re / 1 au)**2
    # - Let me know how it goes or if you need anything else! Cloudy cases are running now, but will take longer.

    # - Ag is the geometric albedo, Phi is the phase function, and Ag x
    #   Phi is the phase-dependent reflectivity. Planet to star flux ratio would then
    #   be,
    # -
    # - Fp/Fs = Ag x Phi x (Rp/a)**2 .
    # -
    # - There are ways to mimic phase-dependent brightnesses with the data
    #   I shared. They are "representative" of quadrature, and a Lambert
    #   phase function (relative to quadrature) would get first-order effects.
    # -
    # - Note that, a V-band, Earth should have Ag x Phi of about 0.06 at
    #   quadrature. That can help you dial-in the right cloud mixture.

    # Path to your file
    fname = "/fast/jruffio/data/exosims/model_Ty/earth_hires/earth_icrccm_hitran2020_highcloud_50_100000cm-1_toa.rad"
    # fname = "/fast/jruffio/data/exosims/model_Ty/earth_hires/earth_icrccm_hitran2020_clearsky_50_100000cm-1_toa.rad"
    # Load the data, skipping comment lines starting with '#'
    hc_data = np.loadtxt(fname, comments='#')
    hc_where_wvs = np.where((hc_data[:, 0] > 200/1000.) * (hc_data[:, 0] < 1500/1000.))
    hc_data = hc_data[hc_where_wvs[0], :]

    # Split into named columns
    hc_wavelength_nm = hc_data[::-1, 0]*1000  * u.nm # column 1
    hc_wavenumber_cm1 = hc_data[::-1, 1]  # column 2
    hc_solar_flux_1au = hc_data[::-1, 2]  # column 3
    hc_earth_flux_toa = hc_data[::-1, 3]  # column 4
    hc_radiance_streams = hc_data[::-1, 4:]  # columns 5-8
    hc_reflectance = hc_earth_flux_toa / hc_solar_flux_1au  # Reflectance spectrum

    hc_reflectance = broaden_and_resample(wv0, hc_wavelength_nm, hc_reflectance, R, n_jobs=8, broaden_pixel=False)

    plt.plot(wv0, hc_reflectance, label="Ty high cloud Reflectance",linestyle="--",alpha=0.5,linewidth=1)

if 0: # low clouds
    # Path to your file
    fname = "/fast/jruffio/data/exosims/model_Ty/earth_hires/earth_icrccm_hitran2020_lowcloud_50_100000cm-1_toa.rad"
    lc_data = np.loadtxt(fname, comments='#')
    lc_where_wvs = np.where((lc_data[:, 0] > 200/1000.) * (lc_data[:, 0] < 1500/1000.))
    lc_data = lc_data[lc_where_wvs[0], :]

    # Split into named columns
    lc_wavelength_nm = lc_data[::-1, 0]*1000  * u.nm # column 1
    lc_wavenumber_cm1 = lc_data[::-1, 1]  # column 2
    lc_solar_flux_1au = lc_data[::-1, 2]  # column 3
    lc_earth_flux_toa = lc_data[::-1, 3]  # column 4
    lc_radiance_streams = lc_data[::-1, 4:]  # columns 5-8
    lc_reflectance = lc_earth_flux_toa / lc_solar_flux_1au  # Reflectance spectrum

    lc_reflectance = broaden_and_resample(wv0, lc_wavelength_nm, lc_reflectance, R, n_jobs=8, broaden_pixel=False)

    plt.plot(wv0, lc_reflectance, label="Ty low cloud Reflectance",linestyle="--",alpha=0.5,linewidth=1)

if 1: #no clouds
    # Path to your file
    for clouds in ["highcloud","lowcloud","clearsky"]:
        #/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_lowcloud_50_100000cm-1_toa.rad
        #/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_clearsky_50_100000cm-1_toa.rad
        #
        # fname = "/fast/jruffio/data/exosims/model_Ty/earth_hires/earth_icrccm_hitran2020_clearsky_50_100000cm-1_toa.rad"
        fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_"+clouds+"_50_100000cm-1_toa.rad"
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
        # plt.scatter(nc_wavelength_nm, nc_reflectance, label="Ty no cloud (Original)",s=30,c="red",marker="o")

        nc_wavelength_nm2, nc_reflectance2 = merge_regular_with_original(nc_wavelength_nm.to(u.nm).value, nc_reflectance, lmin/(10*R), tol_factor=lmin/(10*R)/2.)
        nc_wavelength_nm2 = nc_wavelength_nm2 * u.nm
        # plt.scatter(nc_wavelength_nm2, nc_reflectance2, label="Ty no cloud (Original2)",s=10,marker="x",c="blue")
        nc_reflectance_R_all = broaden_and_resample(wv0, nc_wavelength_nm2, nc_reflectance2, R, n_jobs=8, broaden_pixel=False)
        np.savetxt(fname.replace(".rad","_R{0}.rad".format(R)), np.column_stack((wv0.to(u.nm).value, nc_reflectance_R_all)), fmt="%.9e",)

        plt.plot(wv0, nc_reflectance_R_all, label="Ty no cloud (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)
        # plt.show()

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
        # plt.scatter(nc_wavelength_nm, nc_reflectance, label="Ty no cloud H2O (Original)")

        nc_wavelength_nm2, nc_reflectance2 = merge_regular_with_original(nc_wavelength_nm.to(u.nm).value, nc_reflectance, lmin/(10*R), tol_factor=lmin/(10*R)/2.)
        nc_wavelength_nm2 = nc_wavelength_nm2 * u.nm
        nc_reflectance_R_h2o = broaden_and_resample(wv0, nc_wavelength_nm2, nc_reflectance2, R, n_jobs=8, broaden_pixel=False)
        np.savetxt(fname.replace(".rad","_R{0}.rad".format(R)), np.column_stack((wv0.to(u.nm).value, nc_reflectance_R_h2o)), fmt="%.9e",)

        plt.plot(wv0, nc_reflectance_R_h2o, label="Ty no cloud H2O (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

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
        # plt.scatter(nc_wavelength_nm, nc_reflectance, label="Ty no cloud O2 (Original)",alpha=1)

        nc_wavelength_nm2, nc_reflectance2 = merge_regular_with_original(nc_wavelength_nm.to(u.nm).value, nc_reflectance, lmin/(10*R), tol_factor=lmin/(10*R)/2.)
        nc_wavelength_nm2 = nc_wavelength_nm2 * u.nm
        nc_reflectance_R_o2 = broaden_and_resample(wv0, nc_wavelength_nm2, nc_reflectance2, R, n_jobs=8, broaden_pixel=False)
        np.savetxt(fname.replace(".rad","_R{0}.rad".format(R)), np.column_stack((wv0.to(u.nm).value, nc_reflectance_R_o2)), fmt="%.9e",)

        plt.plot(wv0, nc_reflectance_R_o2, label="Ty no cloud O2 (R={0})".format(R),linestyle="--",alpha=1,linewidth=2)

if 0: #mixed model
    mixed_model_Ty = 0.10*hc_reflectance+0.10*lc_reflectance+0.8*nc_reflectance
    plt.plot(wv0, mixed_model_Ty, label="Ty mixed model",linestyle="-",alpha=1,linewidth=2)



if 1:
    # Read the CSV file
    filename = "/fast/jruffio/data/exosims/model_Ty/earth_quadrature_R140.csv"  # replace with your file path
    data = np.genfromtxt(filename, delimiter=",", names=True)
    print(data.dtype.names)
    # Access columns by name
    lam = data['lam_um']*1000  # Convert um to nm
    dlam = data['dlam_um']
    Aapp = data['Aapp']
    AgPhi = data['AgPhi']

    # plt.plot(lam, Aapp, label="Earth calibrated R=140",linestyle="-",alpha=1,linewidth=2)
    plt.plot(lam, AgPhi, label="Earth calibrated R=140",linestyle="-",alpha=1,linewidth=2)

plt.xlim([lmin,lmax])
# plt.ylim([0,0.1])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend(loc="upper left")
plt.show()