# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:56:11 2024

@author: Gerard Garcia
"""

import os
from pathlib import Path
import calibration as calib

import warnings
warnings.filterwarnings("ignore")

calib.init()
                   
def change_keyword(path, keyword, change_from, change_to):
    from astropy.io import fits
    
    if os.path.isdir(path):
        list_of_files = os.listdir(path)
    else:
        list_of_files = [path]
    
    for fname in list_of_files:
        if fname.endswith(".fits"):
            if len(list_of_files)>1:
                fullpath = os.path.join(path, fname)
            else:
                fullpath = fname

            with fits.open(fullpath, mode="update") as hdul:
                hdr = hdul[0].header

                if hdr.get(keyword) == change_from:
                    hdr[keyword] = change_to
                    hdul.flush()
                    
def check_images(dir):
    
    from ccdproc import ImageFileCollection
    
    #CHECK IMAGES AVAILABLE
    selected_keywords = [ #fits header parameters to check
        'IMAGETYP', 'NAXIS1', 'NAXIS2', 'OBJECT' , 
            'INSAPDY', 'INSGRID', 'INSGRNAM', 'INSGRROT', 'EXPTIME' , 'INSFLID', 'DATE-OBS'
    ]
    
    directory=Path(dir)
    ifc_all = ImageFileCollection(
        location=directory,
        glob_include='caf*.fits',
        keywords=selected_keywords
    )
    
    ifc_all.summary.pprint(max_width=-1, max_lines=-1)

def general_calibrations():
    print('*************************')
    print('MASTER BIAS')
    print('*************************')
    calib.apply_master_bias(plot=True)
    print('Master BIAS applied')
    
    print('*************************')
    print('MASTER FLAT')
    print('*************************')
    calib.apply_master_flat(plot=True)
    print('Master FLAT applied')
    
    print('*************************')
    print('WAVELENGTH CALIBRATIONS')
    print('*************************')
    calib.wavelength_calibration(plot=True)

def science():
    print('*************************')
    print('SKY SUBTRACTION')
    print('*************************')
    calib.sky_substraction()
    
    print('*************************')
    print('ALIGNMENT')
    print('*************************')
    x_min, x_max, y_min, y_max = calib.spec_align()
    
    print('*************************')
    print('SPECTRUM EXTRACTION')
    print('*************************')
    calib.spec_extract(x_min, x_max, y_min, y_max)
    
    
def flux_calibration(std_flux_mags=False):
    print('*************************')
    print('FLUX CALIBRATION')
    print('*************************')
    calib.flux_calib(std_flux_mags=std_flux_mags)