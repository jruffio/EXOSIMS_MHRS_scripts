
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import os


def print_fits_structure(filename):
    """
    Print the structure of a FITS file including HDU types, data shape, and columns if present.

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    """
    with fits.open(filename) as hdul:
        print("Begin"+"/"*40)
        print(f"File: {filename}")
        print(f"Number of HDUs: {len(hdul)}\n")
        for i, hdu in enumerate(hdul):
            hdu_type = type(hdu).__name__
            print(f"HDU {i}: {hdu_type}")
            print(f"  Name       : {hdu.name}")
            print(f"  Data type  : {type(hdu.data).__name__}")
            print(f"  Data shape : {None if hdu.data is None else hdu.data.shape}")
            if hdu.header:
                print(f"  Header keys: {len(hdu.header)} total")
            if hdu_type == 'BinTableHDU' or hdu_type == 'TableHDU':
                print(f"  Columns:")
                for col in hdu.columns:
                    print(f"    - {col.name} ({col.format})")
            print(hdu.data)
            print("-" * 40)
        print("End"+"/"*40)


fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

core_mean_intensity= "/fast/jruffio/data/exosims/exosims_samples/USORT_OVC/core_mean_intensity.fits"
core_thruput = "/fast/jruffio/data/exosims/exosims_samples/USORT_OVC/core_thruput.fits"
occ_trans = "/fast/jruffio/data/exosims/exosims_samples/USORT_OVC/occ_trans.fits"
# # Example usage
# print_fits_structure(core_mean_intensity)
# print_fits_structure(core_thruput)
# print_fits_structure(occ_trans)

with fits.open(occ_trans) as hdul:
    dat = hdul[0].data.T
    WA, D = dat[0].astype(float), dat[1].astype(float)
    print(WA)
    print(D)

    WA = WA*(1000*u.nm/ (7.87*u.m) *u.rad).to(u.arcsec)

    plt.figure(figsize=(5,4))
    plt.plot(WA.to_value(u.arcsec),D,label="Occulter")

with fits.open(core_thruput) as hdul:
    dat = hdul[0].data.T
    WA, D = dat[0].astype(float), dat[1].astype(float)
    print(WA)
    print(D)

    WA = WA*(1000*u.nm/ (7.87*u.m) *u.rad).to(u.arcsec)

    # plt.figure()
    plt.plot(WA.to_value(u.arcsec),D,label="Core")
    plt.xlabel(r"Separation (arcsec)",fontsize=12)
    plt.ylabel("Transmission",fontsize=12)
    plt.gca().tick_params(axis='y', labelsize=12)
    plt.gca().tick_params(axis='x', labelsize=12)
    # plt.gca().grid(True)
    plt.legend(loc='center right', fontsize=12, frameon=True)
    plt.tight_layout()

    out_filename = os.path.join(fig_dir, "core_n_occ_thruput.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

with fits.open(core_mean_intensity) as hdul:
    dat = hdul[0].data.T
    WA, D = dat[0].astype(float), dat[1:].astype(float)
    print(WA)
    print(D)

    WA = WA*(1000*u.nm/ (7.87*u.m) *u.rad).to(u.arcsec)

    plt.figure()
    for k in range(D.shape[0]):
        plt.plot(WA.to_value(u.arcsec),D[k],label="{0}".format(k))
    plt.yscale("log")
    plt.legend()
    plt.show()
