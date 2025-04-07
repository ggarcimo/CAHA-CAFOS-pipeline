# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:57:30 2024

@author: Gerard Garcia
"""

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from ccdproc import ImageFileCollection #conda install -c astropy ccdproc
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from tea_utils import tea_avoid_astropy_warnings
from tea_utils import tea_imshow
from tea_utils import tea_ifc_statsummary
from tea_utils import SliceRegion2D
import numpy as np
import matplotlib.patches as patches
from astropy.nddata import CCDData
import ccdproc
from astropy.stats import mad_std
from tea_utils import tea_statsummary
from tqdm.notebook import tqdm
from astropy.nddata.nduncertainty import StdDevUncertainty
from tea_wavecal import TeaWaveCalibration
from tea_utils import SliceRegion1D
import astropy.units as u
from tea_wavecal import apply_wavecal_ccddata
from tea_wavecal import fit_sdistortion
from tea_wavecal import polfit_residuals_with_sigma_rejection
import os, glob

tea_avoid_astropy_warnings(True)

from astropy.table import Table
from astropy import stats
from scipy.ndimage.filters import gaussian_filter1d
from astropy import constants as cnt


'''
INSGRID    INSGRNAM
--------   ----------
GRISM-11   FREE
GRISM- 9   green-200
GRISM- 2   green-100

SLIT    INSAPDY
-----   ------------
1''     89.0 micron
1.5''   129.0 micron
'''

# Set directories as a global variable
# DATADIR='test'
# directory=Path(DATADIR)
# PLOTDIR=os.path.join(DATADIR, "plots")

def init():
    global DATADIR, directory, PLOTDIR, imge_code
    DATADIR = input(' * Input directory where your data are: ')
    directory = Path(DATADIR)
    PLOTDIR = os.path.join(DATADIR, "plots")
    if not os.path.isdir(PLOTDIR):
        os.makedirs(PLOTDIR)
    imge_code = input(' * What letter/word do all your input files begin with? (end input with *, exemple: caf*): ')
    # plt.ion()
    print(' * Functions used for the reduction')
    print('   - general_calibrations(): bias, flats, and wave calibrations')
    print('   - science(): sky substraction, alignment, and spectrum extraction')
    print('   - flux_calibration(): flux calibration of reduced images')

def read_images(img_code='*', plot=False):
    ##DATADIR='test'
    #CHECK IMAGES AVAILABLE
    selected_keywords = [ #fits header parameters to check
        'IMAGETYP', 'NAXIS1', 'NAXIS2', 'OBJECT' , 
            'INSAPDY', 'INSGRID', 'INSGRNAM', 'INSGRROT', 'EXPTIME' , 'INSFLID', 'DATE-OBS', 'AIRMASS'
    ]
    
    #directory=Path(DATADIR)
    ifc_all = ImageFileCollection(
        location=directory,
        glob_include=f'{img_code}.fits',
        keywords=selected_keywords
    )
    
    summary_all = tea_ifc_statsummary(ifc_all, directory)
    
    #The plots directory is created inside the DATADIR.
        
    #Show a few images:
    if plot is True:
        for example_image in glob.glob(os.path.join(DATADIR,f"{img_code}.fits")):
            image_name = os.path.basename(example_image).split(".")[0]
            data = fits.getdata(example_image)
            header = fits.getheader(example_image)
            vmin, vmax = np.percentile(data, [40, 99]) #Play with these values to avoid saturation of the image.
            fig, ax = plt.subplots(figsize=(25, 15))
            tea_imshow(fig, ax, data, vmin=vmin, vmax=vmax)
            ax.set_title('{1} ({0})'.format( os.path.basename(example_image),header['OBJECT']))
            plt.savefig(os.path.join(PLOTDIR, image_name+".png"), dpi=80)
            plt.close()

    return ifc_all, summary_all, summary_all.to_pandas()

'''
BIAS
'''
def check_bias(ifc_all, summary_all):
    #BIAS IMAGES
    #summary_all = tea_ifc_statsummary(ifc_all, directory)
    #summary_all.show_in_notebook(display_length=50)
    matches_bias= ['bias' in object_name.lower() for object_name in ifc_all.summary['OBJECT']]
    summary_bias = summary_all[matches_bias]
    
    return summary_bias, summary_bias.to_pandas()
    
def create_master_bias(summary_bias, plot = False):
    list_bias = []
    for filename in summary_bias['file']:
        filepath = directory / filename
    
        print(f'Reading {filepath}')
        data = fits.getdata(filepath)
        header = fits.getheader(filepath)
        
        # compute and subtract median from underscan & overscan regions
        """
        med_underscan = np.median(data[region_underscan.python])
        med_overscan = np.median(data[region_overscan.python])
        med_underoverscan = (med_underscan + med_overscan) / 2
        data = data - med_underoverscan
        """
        # create CCDData instance and append to list
        list_bias.append(CCDData(data=data, header=header, unit='adu'))
        
    num_bias = len(list_bias)
    print(f'Number of BIAS exposures: {num_bias}')
    
    master_bias = ccdproc.combine(
        img_list=list_bias,
        method='average',
        sigma_clip=True, sigma_clip_low_thres=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std
    )
    
    #Write info and save master_bias
    #There are two parameters weird, better remove them
    list_keywords_to_be_removed = ['CCDXPIXE', 'CCDYPIXE']
    for keyword in list_keywords_to_be_removed:
        if keyword in master_bias.header:
            del master_bias.header[keyword]
            
    num_bias = len(list_bias)
    
    # Insert FILENAME keyword and include additional information
    filename_master_bias = 'master_bias.fits'
    master_bias.header['FILENAME'] = filename_master_bias
    master_bias.header['HISTORY'] = '-------------------'
    master_bias.header['HISTORY'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    master_bias.header['HISTORY'] = f'master BIAS combining {num_bias} BIAS images'
    #master_bias.header['HISTORY'] = 'after subtracting median from under/overscan regions'
    #master_bias.header['HISTORY'] = f'underscan (FITS): {region_underscan.fits}'
    #master_bias.header['HISTORY'] = f'overscan  (FITS): {region_overscan.fits}'
    master_bias.header['HISTORY'] = 'using ccdproc.combine, with average and sigma clipping'
    for filename in summary_bias['file']:
        master_bias.header['HISTORY'] = f"+ {filename}"
    
    #Now I trim because biases are different shape. You don't need to!
    #region_useful = SliceRegion2D(np.s_[1:1100, 1:1600], mode='fits')
    #fits_section = region_useful.fits_section
    #master_bias = ccdproc.trim_image(master_bias, fits_section=fits_section)
    
    # save result
    master_bias.write(directory / filename_master_bias, overwrite='yes')      
    median_bias = np.median(master_bias.data)
    gain = master_bias.header['CCDSENS']
    readout_noise = np.mean(summary_bias['robust_std'])
    
    if plot is True:
        #Chack that the resulting master_bias.
        _ = tea_statsummary(master_bias.data)
        fig, ax = plt.subplots(figsize=(15, 5))
        vmin, vmax = np.percentile(data, [5, 95])
        tea_imshow(fig, ax, master_bias, vmin=vmin, vmax=vmax)
        ax.set_title('Master BIAS')
        plt.savefig(os.path.join(PLOTDIR, "master_bias.pdf"), dpi=80, format='pdf')
        plt.close()
        
    return master_bias, median_bias, gain, readout_noise

def apply_master_bias(plot=False):
    
    ifc_all, summary_all,_ = read_images(img_code=imge_code, plot=False)
    summary_bias,_ = check_bias(ifc_all, summary_all)
    master_bias, median_bias, gain, readout_noise = create_master_bias(summary_bias, plot=plot)
    
    #We apply the masterbias to all other images
    
    no_bias= summary_all['IMAGETYP'] != 'bias'
    summary_nobias = summary_all[no_bias]
    
    for filename in tqdm(summary_nobias['file']):
        filepath = directory / filename
        
        # read data and header from file
        data = fits.getdata(filepath)
        header = fits.getheader(filepath)
        print(filepath,header['OBJECT'])
        
        # remove non-standard keywords from header
        list_keywords_to_be_removed = ['CCDXPIXE', 'CCDYPIXE']
        for keyword in list_keywords_to_be_removed:
            if keyword in header:
                del header[keyword]
        
        # compute and subtract median from underscan & overscan regions
        """
        med_underscan = np.median(data[region_underscan.python])
        med_overscan = np.median(data[region_overscan.python])
        med_underoverscan = (med_underscan + med_overscan) / 2
        data = data - med_underoverscan
        """
        # create associated uncertainty
        uncertainty1 = (data - median_bias) / gain
        uncertainty1[uncertainty1 < 0] = 0.0   # remove unrealistic negative estimates
        uncertainty2 = readout_noise**2
        uncertainty = np.sqrt(uncertainty1 + uncertainty2)
        
        # create CCDData instance
        ccdimage = CCDData(
            data=data,
            header=header,
            uncertainty=StdDevUncertainty(uncertainty),
            unit='adu'
        )
        
        # subtract master_bias
        ccdimage_bias_subtracted = ccdproc.subtract_bias(ccdimage, master_bias)
        
        # rotate primary HDU and extensions
        ccdimage_bias_subtracted.data = np.rot90(ccdimage_bias_subtracted.data, 3)
        ccdimage_bias_subtracted.mask = np.rot90(ccdimage_bias_subtracted.mask, 3)
        ccdimage_bias_subtracted.uncertainty.array = np.rot90(ccdimage_bias_subtracted.uncertainty.array, 3)
    
        # update HISTORY in header
        ccdimage_bias_subtracted.header['HISTORY'] = '-------------------'
        ccdimage_bias_subtracted.header['HISTORY'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ccdimage_bias_subtracted.header['HISTORY'] = 'rotating image using np.rot90(array, 3)'
    
        # update FILENAME keyword with output file name
        output_filename = f'z_{filename}'
        ccdimage_bias_subtracted.header['FILENAME'] = output_filename
        # update HISTORY in header
        ccdimage_bias_subtracted.header['HISTORY'] = '-------------------'
        ccdimage_bias_subtracted.header['HISTORY'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ccdimage_bias_subtracted.header['HISTORY'] = 'using ccdproc.subtract_bias'
        ccdimage_bias_subtracted.header['HISTORY'] = 'master BIAS file: master_bias.fits'
        
        # save result
        ccdimage_bias_subtracted.write(directory / output_filename, overwrite='yes')
    

'''
FLATS
'''

def create_master_flat(plot = False):
    #We create a list of flats.
    selected_keywords = ['IMAGETYP', 'NAXIS1', 'NAXIS2', 'OBJECT' , 
            'INSAPDY', 'INSGRID', 'INSGRNAM', 'INSGRROT', 'EXPTIME' , 'INSFLID', 'DATE-OBS'
    ]
    
    ifc_all = ImageFileCollection(
        location=directory,
        glob_include='z_*.fits',
        keywords=selected_keywords
    )
    
    summary_all = tea_ifc_statsummary(ifc_all, directory)
    
    matches_flat = ['flat' in object_name.lower() for object_name in summary_all['OBJECT']]
    summary_flat = summary_all[matches_flat]
    
    grisms = set(summary_flat['INSGRNAM'])
    slits = set(summary_flat['INSAPDY'])
    
    #Check each of the flats
    if plot is True:
        for filename, objname, exptime, median, robust_std in summary_flat[
            'file', 'OBJECT', 'EXPTIME', 'median', 'robust_std'
        ]:
            data = fits.getdata(directory / filename)
            vmin = median - 2 * robust_std
            vmax = median + 2 * robust_std
            vmin, vmax = np.percentile(data, [2, 99.9])
            fig, ax = plt.subplots(figsize=(15, 5))
            tea_imshow(fig, ax, data, vmin=vmin, vmax=vmax)
            ax.set_title(f'{filename}  [{objname}]  EXPTIME={exptime} s')
            plt.savefig(os.path.join(PLOTDIR, f"flat_{filename}.pdf"), dpi=80, format='pdf')
            plt.close()
        
    for grism in grisms:
        for slit in slits:
            mask = (summary_flat['INSGRNAM'] == grism) & (summary_flat['INSAPDY'] == slit)
            if not any(mask):
                continue
            print(f'Grism/Slit combination: {grism}, {slit} micron')
            
            # list of CCDData instances (normalized images)
            list_flats = []
            for filename in tqdm(summary_flat['file'][mask]):
                filepath = directory / filename
                
                # CCDData instance from file
                ccdimage = CCDData.read(filepath)
                
                # normalize image dividing by the median spectrum
                ccdimage = ccdimage.divide(np.median(ccdimage.data, axis=0), handle_meta='first_found')
                
                # append to list
                list_flats.append(ccdimage)
                
            num_flats = len(list_flats)
            print(f'Number of FLATS exposures: {num_flats}')
            
            # combine images
            print(f'Combining {num_flats} images')
            for ccdimage in list_flats:
                print(f"- Normalized {ccdimage.header['FILENAME']}")
            
            master_flat = ccdproc.combine(
                img_list=list_flats,
                method='average',
                sigma_clip=True, sigma_clip_low_thres=5, sigma_clip_high_thres=5,
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std
            )
            
            # update header keywords
            output_filename = f'N1_master_flat_{grism}_{slit}.fits'
            master_flat.header['FILENAME'] = output_filename
            master_flat.header['HISTORY'] = '-------------------'
            master_flat.header['HISTORY'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            master_flat.header['HISTORY'] = f'master FLAT combining {num_flats} FLAT images'
            master_flat.header['HISTORY'] = 'using ccdproc.combine, with average and sigma clipping'
            for ccdimage in list_flats:
                master_flat.header['HISTORY'] = f"+ {ccdimage.header['FILENAME']}"
            
            print(f'Saving {output_filename}')
            
            # save result
            master_flat.write(directory / output_filename, overwrite='yes')
            
            # display combined image
            if plot is True:
                naxis2, naxis1 = master_flat.data.shape
                
                fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
                vmin, vmax = 0.85, 1.15
                vmin, vmax = np.percentile(master_flat.data, [5, 99])
                tea_imshow(fig, ax1, master_flat.data, vmin=vmin, vmax=vmax)
                ax.set_title(f'Master flat ({grism}, slit: {slit})')
                plt.savefig(os.path.join(PLOTDIR, f"master_flat_{grism}_{slit}.pdf"), dpi=80, format='pdf')
                plt.close()        
    
    return ifc_all, summary_all

def apply_master_flat(plot = False):
    
    ifc_all, summary_all = create_master_flat()
    
    # Correcting all images using the master flat
    matches_no_flat = summary_all['IMAGETYP'] != 'flat'
    matches_no_flat = matches_no_flat & (summary_all['IMAGETYP'] != 'sky')
    matches_no_flat = matches_no_flat & (summary_all['IMAGETYP'] != 'bias')
    matches_no_flat = matches_no_flat & (summary_all['IMAGETYP'] != 'dark')
    summary_no_flat = summary_all[matches_no_flat]
    
    for i, filename in enumerate(tqdm(summary_no_flat['file'])):
        print('---')
        print(f'Flatfielding image: {filename}  --> File {i+1} / {len(summary_no_flat)}')
        
        mask = summary_no_flat['file']==filename
        grism = summary_no_flat['INSGRNAM'][mask]
        slit = summary_no_flat['INSAPDY'][mask]
        
        grism = grism[0]
        slit = slit[0]
        
        filename_master_flat = f'N1_master_flat_{grism}_{slit}.fits'
        try:
            master_flat = CCDData.read(directory / filename_master_flat)
        except:
            print(f'No master flat with this grism/slit combination: {grism}/{slit}')
            continue
        
        # generate CCDData instance
        ccdimage = CCDData.read(directory / filename)
    
        # divide science exposure by master flat
        ccdimage_after_flat = ccdproc.flat_correct(ccdimage, master_flat)
        
        # update FILENAME keyword with output file name
        output_filename = f'f{filename}'
        ccdimage_after_flat.header['FILENAME'] = output_filename
        
        # update HISTORY in header
        ccdimage_after_flat.header['HISTORY']  = '-------------------'
        ccdimage_after_flat.header['HISTORY']  = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ccdimage_after_flat.header['HISTORY'] = 'using ccdproc.flat_correct'
        ccdimage_after_flat.header['HISTORY'] = f'master Flat Field file: {filename_master_flat}'
        
        # save result
        ccdimage_after_flat.write(directory / output_filename, overwrite='yes')
        print(f'Output file name.: {output_filename}')
        
        if plot is True:
            # display image before and after flatfielding
            mean, median, std = sigma_clipped_stats(ccdimage_after_flat.data, sigma=3.0)
            vmin = median - 2 * std
            vmax = median + 2 * std
            vmin, vmax = np.percentile(ccdimage_after_flat.data, [5, 99.])
            fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
            title = f"{filename}\n{ccdimage.header['OBJECT']}"
            tea_imshow(fig, axarr[0], ccdimage.data, vmin=vmin, vmax=vmax, title=title)
            title = f"{output_filename}\n{ccdimage.header['OBJECT']}"
            tea_imshow(fig, axarr[1], ccdimage_after_flat.data, vmin=vmin, vmax=vmax, title=title)
        
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTDIR, f"img_flatfielded_{grism}_{slit}.pdf"), dpi=80, format='pdf')
            # plt.show()
            plt.close()

'''
Wavelength calibrations
'''
def wavelength_calibration(plot=False):
    
    def find_best_match(observed_peaks, reference_peaks):
        max_matches = 0
        id_matches = []
        
        # Iterate over a range of potential differences
        for difference in np.arange(np.min(reference_peaks)-np.mean(observed_peaks),
                                    np.max(reference_peaks)-np.mean(observed_peaks)):  
            id_matches_obs, id_matches_ref = find_peak_matches(observed_peaks, reference_peaks, difference)
            num_matches = len(id_matches_obs)
            if num_matches > max_matches:
                # Found a new best match
                max_matches = num_matches
                id_matches = (id_matches_obs, id_matches_ref)
        
        return id_matches
    
    def find_peak_matches(observed_peaks, reference_peaks, difference, tol=10):
        reference_peaks_shifted = [peak - difference for peak in reference_peaks]
        
        # Find the matches between observed and shifted reference peaks within (tol) pix
        id_matches_obs = [np.argmin(np.abs(observed_peaks-reference_peaks_shifted[p])) for p in range(len(reference_peaks_shifted)) if np.min(np.abs(observed_peaks-reference_peaks_shifted[p]))<tol]
        id_matches_ref = [p for p in range(len(reference_peaks_shifted)) if np.min(np.abs(observed_peaks-reference_peaks_shifted[p]))<tol]
        
        return id_matches_obs, id_matches_ref
    
    # Arc images
    selected_keywords = [ #fits header parameters to check
        'IMAGETYP', 'NAXIS1', 'NAXIS2', 'OBJECT' , 
            'INSAPDY', 'INSGRID', 'INSGRNAM', 'INSGRROT', 'EXPTIME' , 'INSFLID', 'DATE-OBS', 'AIRMASS'
    ]
    
    ifc_all = ImageFileCollection(
        location=directory ,
        glob_include='fz_*.fits',
        keywords=selected_keywords
    )
    summary_all = ifc_all.summary
        
    matches_arc = summary_all['IMAGETYP'] == 'arc'
    summary_arc = summary_all[matches_arc]
    
    grisms = set(summary_arc['INSGRNAM'])
    slits = set(summary_arc['INSAPDY'])
    
    if plot is True:
        for filename, objname in summary_arc['file', 'OBJECT']:
            data = fits.getdata(directory / filename)
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            vmin = median - 2 * std
            vmax = median + 8 * std
            fig, ax = plt.subplots(figsize=(15, 5))
            tea_imshow(fig, ax, data, vmin=vmin, vmax=vmax, 
                       title=f'{filename}, {objname}')
            plt.savefig(os.path.join(PLOTDIR, f"arc_{filename}_{objname}.pdf"), dpi=80, format='pdf')            
            # plt.show()
            plt.close()
    
    for grism in grisms:
        for slit in slits:
            mask = (summary_arc['INSGRNAM'] == grism) & (summary_arc['INSAPDY'] == slit)
            if not any(mask):
                continue
            print(f'Grism/Slit combination: {grism}, {slit} micron')
            arc_file=summary_arc['file'][mask][0]
            input_filename = directory / arc_file
            data = fits.getdata(input_filename)
            
            #we use a slice between 120-130 to find the peaks
            wavecalib = TeaWaveCalibration()
            ns_range = SliceRegion1D(np.s_[600:700], mode='fits')
            
            xpeaks_reference_initial, ixpeaks_reference, spectrum_reference = wavecalib.compute_xpeaks_reference(
                data=data,
                sigma_smooth=2,
                delta_flux=10,
                ns_range=ns_range,
                plot_spectrum=True,
                plot_peaks=True,
                pdf_output=os.path.join(PLOTDIR, f"xpeaks_reference_{grism}_{slit}.pdf"),
                pdf_only=True
            )
            
            # reference lines and pixels for Hg+He+Rb lamps, grism g-200
            comparison_reference_wav = np.array([4046.56, 4358.33, 4678.15, 4799.91, 5085.82, 5460.74,
                                                 6438.47, 7800.268, 7947.603, 8521.162, 10139.8])
            comparison_reference_pix = np.array([1475,1400,1320,1295,1225,1140,925,635,605,485,125])
            
            # revert x-axis
            comparison_reference_pix = 2000-comparison_reference_pix
            
            
            # Find the best match
            observed_peaks = xpeaks_reference_initial.data.tolist()
            reference_peaks = comparison_reference_pix
            id_matches_obs, id_matches_ref = find_best_match(observed_peaks, reference_peaks)
            id_matches_obs, id_matches_ref = np.array(id_matches_obs), np.array(id_matches_ref)
            
            print("Number of matches:", len(id_matches_obs))
            print("Indexes of matches:", id_matches_obs+1)
            print("Corresponding wavelengths:", comparison_reference_wav[id_matches_ref])
            
            
            # Apply peak wavelength assignment
            xpeaks_reference = np.array(ixpeaks_reference)[id_matches_obs] * u.pixel
            wavelengths_reference = comparison_reference_wav[id_matches_ref] * u.Angstrom
            wavecalib.define_peak_wavelengths(
                xpeaks=xpeaks_reference,
                wavelengths=wavelengths_reference
            )
            
            #we fit a 5 degree pol
            poly_fits_wav, residual_std_wav, poly_fits_pix, residual_std_pix, \
            crval1_linear, cdelt1_linear, crmax1_linear = wavecalib.fit_xpeaks_wavelengths(
                xpeaks=xpeaks_reference,
                degree_wavecalib=5,
                debug=True,
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"fit_xpeaks_{grism}_{slit}.pdf"), 
                pdf_only=True
            )
            
            crval1 = crval1_linear
            cdelt1 = cdelt1_linear
            
            wavecalib.compute_xpeaks_image(
                data=data, 
                xpeaks_reference=xpeaks_reference, 
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"xpeaks_image_{grism}_{slit}.pdf"),
                pdf_only=True,
                title=input_filename
            )
            
            wavecalib.reset_image()
             
            ns_range1 = SliceRegion1D(np.s_[300:400], mode='fits')
        
            wavecalib.compute_xpeaks_image(
                data=data,
                xpeaks_reference=xpeaks_reference,
                ns_range=ns_range1,
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"slice1_xpeaks_image_{grism}_{slit}.pdf"),
                pdf_only=True,
                title=input_filename,
                disable_tqdm=False
            )
            
            ns_range2 = SliceRegion1D(np.s_[400:800], mode='fits')
        
            wavecalib.compute_xpeaks_image(
                data=data,
                xpeaks_reference=xpeaks_reference,
                ns_range=ns_range2,
                direction='down',
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"slice2_xpeaks_image_{grism}_{slit}.pdf"),
                pdf_only=True,
                title=input_filename,
                disable_tqdm=False
            )
            
            ns_range3 = SliceRegion1D(np.s_[800:1000], mode='fits')
        
            wavecalib.compute_xpeaks_image(
                data=data,
                xpeaks_reference=xpeaks_reference,
                ns_range=ns_range3,
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"slice3_xpeaks_image_{grism}_{slit}.pdf"),
                pdf_only=True,
                title=input_filename,
                disable_tqdm=False
            )
            
            #We fit C distorsion:
            wavecalib.fit_cdistortion(
                degree_cdistortion=2, 
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"fit_cdistortion_{grism}_{slit}.pdf"),
                pdf_only=True,
                title=input_filename
            )
            
            wavecalib.plot_cdistortion(data, title=input_filename,
                                       pdf_output=os.path.join(PLOTDIR, f"cdistortion_{grism}_{slit}.pdf"),
                                       pdf_only=True)
            
            # define additional information for the image header
            history_list = [
                'Wavelength calibration',
                f'Input file: {input_filename}'
            ]
            
            # name of the auxiliary FITS file
            wavecal_filename =  f'{directory}/wavecal_{grism}_{slit}_{arc_file}'
            
            # compute wavelength calibration and save result in auxiliary FITS file
            wavecalib.fit_wavelengths(
                output_filename= wavecal_filename,
                history_list=history_list,
                plots=True,
                pdf_output=os.path.join(PLOTDIR, f"fit_wavelengths_{grism}_{slit}.pdf"),
                pdf_only=True,
                title=input_filename,
                disable_tqdm=False,
            )
            
            data_wavecalib = wavecalib.apply(
                data=data, 
                crval1=crval1, 
                cdelt1=cdelt1, 
                disable_tqdm=False
            )
            
            wavecalib.plot_data_comparison(
                data_before=data,
                data_after=data_wavecalib,
                crval1=crval1,
                cdelt1=cdelt1,
                title=f'Wavelength calibration of {input_filename}',
                semi_window=25,
                pdf_output=os.path.join(PLOTDIR, f"data_comparison_{grism}_{slit}.pdf"),
                pdf_only=True,
            )
            
            matches_run = (summary_all['IMAGETYP'] == 'science')
            summary_science = summary_all[matches_run]
            matches_run = (summary_science['EXPTIME']>20)
            summary_science = summary_science[matches_run]
            matches_run = (summary_science['INSGRNAM'] == grism) & (summary_science['INSAPDY'] == slit)
            
            
            print(f'Total number of science files ...: {len(summary_science)}')
            
            for i, (filename) in enumerate(tqdm(summary_science['file'])):
                input_filename = directory / Path(filename).name
                output_filename = directory / f'w{Path(filename).name}'
                apply_wavecal_ccddata(
                    infile=input_filename,
                    wcalibfile=wavecal_filename,
                    outfile=output_filename,
                    crval1=crval1,
                    cdelt1=cdelt1
                )
                
# def wavelength_calibration(grism='green-200', plot=False):
    
#     # Arc images
#     selected_keywords = [ #fits header parameters to check
#         'IMAGETYP', 'NAXIS1', 'NAXIS2', 'OBJECT' , 
#             'INSAPDY', 'INSGRID', 'INSGRNAM', 'INSGRROT', 'EXPTIME' , 'INSFLID', 'DATE-OBS', 'AIRMASS'
#     ]
    
#     ifc_all = ImageFileCollection(
#         location=directory ,
#         glob_include='fz_*.fits',
#         keywords=selected_keywords
#     )
#     summary_all = ifc_all.summary
        
#     matches_arc = summary_all['IMAGETYP'] == 'arc'
#     summary_arc = summary_all[matches_arc]
    
#     if plot is True:
#         for filename, objname in summary_arc['file', 'OBJECT']:
#             data = fits.getdata(directory / filename)
#             mean, median, std = sigma_clipped_stats(data, sigma=3.0)
#             vmin = median - 2 * std
#             vmax = median + 8 * std
#             fig, ax = plt.subplots(figsize=(15, 5))
#             tea_imshow(fig, ax, data, vmin=vmin, vmax=vmax, 
#                        title=f'{filename}, {objname}')
#             plt.savefig(os.path.join(PLOTDIR, f"arc_{filename}_{objname}.png"), dpi=80)            
#             plt.show()
    
#     arc_file=summary_arc['file'][summary_arc['INSGRNAM'] == grism][0]
#     input_filename = directory / arc_file
#     data = fits.getdata(input_filename)
    
#     #we use a slice between 120-130 to find the peaks
#     wavecalib = TeaWaveCalibration()
#     ns_range = SliceRegion1D(np.s_[600:700], mode='fits')
    
#     xpeaks_reference_initial, ixpeaks_reference, spectrum_reference = wavecalib.compute_xpeaks_reference(
#         data=data,
#         sigma_smooth=2,
#         delta_flux=10,
#         ns_range=ns_range,
#         plot_spectrum=True,
#         plot_peaks=True
#     )
    
#     # reference lines and pixels for Hg+He+Rb lamps, grism g-200
#     comparison_reference_wav = np.array([4046.56, 4358.33, 4678.15, 4799.91, 5085.82, 5460.74,
#                                          6438.47, 7800.268, 7947.603, 8521.162, 10139.8])
#     comparison_reference_pix = np.array([1475,1400,1320,1295,1225,1140,925,635,605,485,125])
    
#     # revert x-axis
#     comparison_reference_pix = 2000-comparison_reference_pix
    
#     def find_best_match(observed_peaks, reference_peaks):
#         max_matches = 0
#         id_matches = []
        
#         # Iterate over a range of potential differences
#         for difference in np.arange(np.min(reference_peaks)-np.mean(observed_peaks),
#                                     np.max(reference_peaks)-np.mean(observed_peaks)):  
#             id_matches_obs, id_matches_ref = find_peak_matches(observed_peaks, reference_peaks, difference)
#             num_matches = len(id_matches_obs)
#             if num_matches > max_matches:
#                 # Found a new best match
#                 max_matches = num_matches
#                 id_matches = (id_matches_obs, id_matches_ref)
        
#         return id_matches
    
#     def find_peak_matches(observed_peaks, reference_peaks, difference, tol=10):
#         reference_peaks_shifted = [peak - difference for peak in reference_peaks]
        
#         # Find the matches between observed and shifted reference peaks within (tol) pix
#         id_matches_obs = [np.argmin(np.abs(observed_peaks-reference_peaks_shifted[p])) for p in range(len(reference_peaks_shifted)) if np.min(np.abs(observed_peaks-reference_peaks_shifted[p]))<tol]
#         id_matches_ref = [p for p in range(len(reference_peaks_shifted)) if np.min(np.abs(observed_peaks-reference_peaks_shifted[p]))<tol]
        
#         return id_matches_obs, id_matches_ref
    
#     # Find the best match
#     observed_peaks = xpeaks_reference_initial.data.tolist()
#     reference_peaks = comparison_reference_pix
#     id_matches_obs, id_matches_ref = find_best_match(observed_peaks, reference_peaks)
#     id_matches_obs, id_matches_ref = np.array(id_matches_obs), np.array(id_matches_ref)
    
#     print("Number of matches:", len(id_matches_obs))
#     print("Indexes of matches:", id_matches_obs+1)
#     print("Corresponding wavelengths:", comparison_reference_wav[id_matches_ref])
    
    
#     # Apply peak wavelength assignment
#     xpeaks_reference = np.array(ixpeaks_reference)[id_matches_obs] * u.pixel
#     wavelengths_reference = comparison_reference_wav[id_matches_ref] * u.Angstrom
#     wavecalib.define_peak_wavelengths(
#         xpeaks=xpeaks_reference,
#         wavelengths=wavelengths_reference
#     )
    
#     #we fit a 5 degree pol
#     poly_fits_wav, residual_std_wav, poly_fits_pix, residual_std_pix, \
#     crval1_linear, cdelt1_linear, crmax1_linear = wavecalib.fit_xpeaks_wavelengths(
#         xpeaks=xpeaks_reference,
#         degree_wavecalib=5,
#         debug=True,
#         plots=True
#     )
    
#     crval1 = crval1_linear
#     cdelt1 = cdelt1_linear
    
#     wavecalib.compute_xpeaks_image(
#         data=data, 
#         xpeaks_reference=xpeaks_reference, 
#         plots=True,
#         title=input_filename
#     )
    
#     wavecalib.reset_image()
     
#     ns_range1 = SliceRegion1D(np.s_[300:400], mode='fits')

#     wavecalib.compute_xpeaks_image(
#         data=data,
#         xpeaks_reference=xpeaks_reference,
#         ns_range=ns_range1,
#         plots=True,
#         title=input_filename,
#         disable_tqdm=False
#     )
    
#     ns_range2 = SliceRegion1D(np.s_[400:800], mode='fits')

#     wavecalib.compute_xpeaks_image(
#         data=data,
#         xpeaks_reference=xpeaks_reference,
#         ns_range=ns_range2,
#         direction='down',
#         plots=True,
#         title=input_filename,
#         disable_tqdm=False
#     )
    
#     ns_range3 = SliceRegion1D(np.s_[800:1000], mode='fits')

#     wavecalib.compute_xpeaks_image(
#         data=data,
#         xpeaks_reference=xpeaks_reference,
#         ns_range=ns_range3,
#         plots=True,
#         title=input_filename,
#         disable_tqdm=False
#     )
    
#     #We fit C distorsion:
#     wavecalib.fit_cdistortion(
#         degree_cdistortion=2, 
#         plots=True,
#         title=input_filename
#     )
    
#     wavecalib.plot_cdistortion(data, title=input_filename)
    
#     # define additional information for the image header
#     history_list = [
#         'Wavelength calibration',
#         f'Input file: {input_filename}'
#     ]
    
#     # name of the auxiliary FITS file
#     wavecal_filename =  f'{directory}/wavecal_{arc_file}'
    
#     # compute wavelength calibration and save result in auxiliary FITS file
#     wavecalib.fit_wavelengths(
#         output_filename= wavecal_filename,
#         history_list=history_list,
#         plots=True,
#         title=input_filename,
#         disable_tqdm=False,
#     )
    
#     data_wavecalib = wavecalib.apply(
#         data=data, 
#         crval1=crval1, 
#         cdelt1=cdelt1, 
#         disable_tqdm=False
#     )
    
#     wavecalib.plot_data_comparison(
#         data_before=data,
#         data_after=data_wavecalib,
#         crval1=crval1,
#         cdelt1=cdelt1,
#         title=f'Wavelength calibration of {input_filename}',
#         semi_window=25
#     )
    
#     matches_run = (summary_all['IMAGETYP'] == 'science')
#     summary_science = summary_all[matches_run]
#     matches_run = (summary_science['EXPTIME']>20)
#     summary_science = summary_science[matches_run]
    
    
#     print(f'Total number of science files ...: {len(summary_science)}')
    
#     for i, (filename) in enumerate(tqdm(summary_science['file'])):
#         input_filename = directory / Path(filename).name
#         output_filename = directory / f'w{Path(filename).name}'
#         apply_wavecal_ccddata(
#             infile=input_filename,
#             wcalibfile=wavecal_filename,
#             outfile=output_filename,
#             crval1=crval1,
#             cdelt1=cdelt1
#         )

'''
Sky substraction
'''
    
def sky_substraction():
    
    ifc_all, summary_all, _ = read_images(img_code='wfz*', plot=False)
    
    matches_run = (summary_all['IMAGETYP'] == 'science')
    summary_science = summary_all[matches_run]
    matches_run = (summary_science['EXPTIME']>20)
    summary_science = summary_science[matches_run]
    
    print('---')
    print('Index  //  filename  //  objectname')
    print('---')
    for i, (filename, obj) in enumerate(zip(summary_science['file'], summary_science['OBJECT'])):
        print(i, ' // ', filename, ' // ', obj)
    print('---')
    
    cont = False
    while cont is False:
        file_index = input(f'Choose file (select index from 0 to {len(summary_science)-1}) or "all": ')
        if file_index == 'all':
            indexes = list(range(0, len(summary_science)))
            cont = True
        else:
            indexes = [int(file_index)]
            if indexes[0] > (len(summary_science)-1):
                print(f'Incorrect index. Choose an index between 0 and {len(summary_science)-1}.')
            else:
                cont = True
    
    for index in indexes:
        science_file = summary_science['file'][index]
        input_filename = directory / science_file
        
        print('---')
        data = fits.getdata(input_filename)
        naxis2, naxis1 = data.shape
        print(f'NAXIS1={naxis1}')
        print(f'NAXIS2={naxis2}')
        header = fits.getheader(input_filename)
        
        cunit1 = 1 * u.Unit(header['cunit1'])
        crpix1 = header['crpix1'] * u.pixel
        crval1 = header['crval1'] * u.Unit(cunit1)
        cdelt1 = header['cdelt1'] * u.Unit(cunit1) / u.pixel
        print(f'crpix1: {crpix1}')
        print(f'crval1: {crval1}')
        print(f'cdelt1: {cdelt1}')
        print(f'cunit1: {cunit1}')
        
        # matplotlib.use('Qt5Agg')
        plt.ion()
        vmin, vmax = np.percentile(data, [5, 99])
        for iplot in range(2):
            if iplot == 0:
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                tea_imshow(fig1, ax1, data, vmin=vmin, vmax=vmax, title=input_filename, cmap='gray',
                            crpix1=crpix1, crval1=crval1, cdelt1=cdelt1, cunit1=cunit1)
                ax1.tick_params(axis='both', labelsize=25)
                fig1.canvas.draw()  # Force update
                fig1.canvas.flush_events()
                plt.pause(0.1)
            else:
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                tea_imshow(fig2, ax2, data, vmin=vmin, vmax=vmax, title=input_filename, cmap='gray', aspect='auto')
                ax2.tick_params(axis='both', labelsize=25)
                fig2.canvas.draw()  # Force update
                fig2.canvas.flush_events()
                plt.pause(0.1)
        # plt.tight_layout()
        # fig1.canvas.draw()  # Force update
        # fig1.canvas.flush_events()
        # plt.pause(0.1)
        
        print('---')
        print('Select two empty background regions from the 2D image. They need to be at each side of the main object trace. They should not contain other traces. INTEGERS ONLY.')
        
        cont = False
        first = True
        while cont is False:
            print('---')
            skyregion1_min = int(input('Skyregion1 y axis MIN: '))
            skyregion1_max = int(input('Skyregion1 y axis MAX: '))
            skyregion2_min = int(input('Skyregion2 y axis MIN: '))
            skyregion2_max = int(input('Skyregion1 y axis MAX: '))
            if first:
                plt.close(fig1)
                plt.close(fig2)
            else:
                plt.close(fig)
            
            # Define good sky regions (iterate with next window)
            skyregion1 = (skyregion1_min, skyregion1_max)
            skyregion2 = (skyregion2_min, skyregion2_max)
            
            indices_sky = np.r_[skyregion1[0]:(skyregion1[1]+1), 
                                skyregion2[0]:(skyregion2[1]+1)]
            
            #We fit the sky level
            fig, ax = plt.subplots(figsize=(12, 6))
            xp = np.arange(naxis2)
            yp = np.mean(data[:, 300:1001], axis=1)
            ax.plot(xp, yp, label='mean spatial profile')
            ax.set_xlabel('Y axis (array index)')
            ax.set_ylabel('Number of counts')
            ax.set_title(input_filename)
            
            # fit of the sky level
            xfit = indices_sky
            yfit = yp[indices_sky]
            poly_funct, yres, reject = polfit_residuals_with_sigma_rejection(
                x=xfit,
                y=yfit,
                deg=1,# you can change the order of the fit
                times_sigma_reject=3
            )
            xpredict = np.arange(naxis2)
            ypredict = poly_funct(xpredict)
            ax.plot(xpredict, ypredict, color='C2', ls='-', label='fitted sky level')
            ax.plot(xfit[reject], yfit[reject], 'rx', label='rejected points')
            ax.legend()
            
            # display sky regions
            ymin, ymax = ax.get_ylim()
            for r in [skyregion1, skyregion2]:
                ax.axvline(r[0], lw=1, color='C1')
                ax.axvline(r[1], lw=1, color='C1')
                rect = patches.Rectangle((r[0], ymin), r[1]-r[0], ymax-ymin, facecolor='C1', alpha=0.2)
                ax.add_patch(rect)
            
            fig.canvas.draw()  # Force update
            fig.canvas.flush_events()
            plt.pause(0.1)
            
            quest = input('Are you okay with the background selection? (Type "yes" if so): ')
            if quest in ['Yes', 'yes', 'Y', 'y']:
                cont = True
                plt.close(fig)
            else:
                first = False
            
        data_sky = data[indices_sky]
        # initialize 2D full frame sky image
        data_sky_fullframe = np.zeros((naxis2, naxis1))
        
        # fit each image column
        xfit = indices_sky
        xpredict = np.arange(naxis2)
        
        for i in tqdm(range(naxis1)):
            yfit = data_sky[:, i]
            poly_funct, yres, reject = polfit_residuals_with_sigma_rejection(
                x=xfit,
                y=yfit,
                deg=1,
                times_sigma_reject=3
            )
            data_sky_fullframe[:, i] = poly_funct(xpredict)
            
        for data_, title in zip([data, data_sky_fullframe],
                               ['data', 'data_sky_fullframe']):
            fig, ax = plt.subplots(figsize=(15, 5))
            tea_imshow(fig, ax, data_, vmin=vmin, vmax=vmax, title=title, cmap='gray',
                       crpix1=crpix1, crval1=crval1, cdelt1=cdelt1, cunit1=cunit1)
            plt.tight_layout()
            plt.show(block=False)
            # fig.canvas.draw()  # Force update
            # fig.canvas.flush_events()
            plt.pause(5)
            plt.close(fig)
            
        data_sky_subtracted = data - data_sky_fullframe
        
        hdu = fits.PrimaryHDU(data=data_sky_subtracted, header=header)
        hdu.writeto(f'{directory}/s{science_file}', overwrite=True)
 
    
'''
Aligning the spectra
'''

def spec_align():
    
    ifc_all, summary_all, _ = read_images(img_code='swfz*', plot=False)
    
    matches_run = (summary_all['IMAGETYP'] == 'science')
    summary_science = summary_all[matches_run]
    
    print('---')
    print('Index  //  filename  //  objectname')
    print('---')
    for i, (filename, obj) in enumerate(zip(summary_science['file'], summary_science['OBJECT'])):
        print(i, ' // ', filename, ' // ', obj)
    print('---')
    
    cont = False
    while cont is False:
        file_index = input(f'Choose file (select index from 0 to {len(summary_science)-1}) or "all": ')
        if file_index == 'all':
            indexes = list(range(0, len(summary_science)))
            cont = True
        else:
            indexes = [int(file_index)]
            if indexes[0] > (len(summary_science)-1):
                print(f'Incorrect index. Choose an index between 0 and {len(summary_science)-1}.')
            else:
                cont = True
    
    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []
    for index in indexes:
        science_file = summary_science['file'][index]
        input_filename = science_file
        
        data = fits.getdata(directory / input_filename)
        header = fits.getheader(directory / input_filename)
        # Remove nans from data
        data[np.isnan(data)] = 0
        
        cunit1 = 1 * u.Unit(header['cunit1'])
        crpix1 = header['crpix1'] * u.pixel
        crval1 = header['crval1'] * u.Unit(cunit1)
        cdelt1 = header['cdelt1'] * u.Unit(cunit1) / u.pixel
        ctype1 = header['ctype1']
        ax_max = header['NAXIS1']-1
        print('---')
        print(f'crpix1: {crpix1}')
        print(f'crval1: {crval1}')
        print(f'cdelt1: {cdelt1}')
        print(f'cunit1: {cunit1}')
        print(f'ctype1: {ctype1}')
        print(f'X axis max: {ax_max}')
        print('---')
        
        vmin, vmax = np.percentile(data, [5, 99])
        
        fig, ax = plt.subplots(figsize=(15, 5))
        tea_imshow(fig, ax, data, vmin=vmin, vmax=vmax, title=input_filename,
                   cmap='gray', aspect='auto')
        ax.tick_params(axis='both', labelsize=25)
        # plt.tight_layout()
        # plt.show()
        fig.canvas.draw()  # Force update
        fig.canvas.flush_events()
        plt.pause(0.1)
        
        print('Choose the x and the y ranges with the spectral trace IN the window, to align the spectum. INTEGERS ONLY')
            
        cont = False
        while cont is False:
            print('---')
            x_min = int(input('x min: '))
            x_max = int(input('x max: '))
            y_min = int(input('y min: '))
            y_max = int(input('y max: '))
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(15, 5))
            tea_imshow(fig, ax, data, vmin=vmin, vmax=vmax, title=input_filename,
                       cmap='gray', aspect='auto')
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.tick_params(axis='both', labelsize=25)
            # plt.tight_layout()
            # plt.show()
            fig.canvas.draw()  # Force update
            fig.canvas.flush_events()
            plt.pause(0.1)
            
            quest = input('Are you okay with the selected ranges? (Type "yes" if so): ')
            if quest in ['Yes', 'yes', 'Y', 'y']:
                cont = True
                plt.close(fig)
        
        
        data_straight = fit_sdistortion(
            data=data,
            ns_min=y_min,
            ns_max=y_max,
            nc_min=x_min,
            nc_max=x_max,
            median_size=(1, 51),
            plots=True
        )
        
        hdu = fits.PrimaryHDU(data=data_straight, header=header)
        hdu.writeto(f'{directory}/d{input_filename}', overwrite=True)
    
        xmin_list.append(x_min)
        xmax_list.append(x_max)
        ymin_list.append(y_min)
        ymax_list.append(y_max)
        
    return xmin_list, xmax_list, ymin_list, ymax_list


'''
Extraction of the spectra
'''

def spec_extract(initial_xmin, initial_xmax, initial_ymin, initial_ymax):
    
    ifc_all, summary_all, _ = read_images(img_code='dswfz*', plot=False)
    
    matches_run = (summary_all['IMAGETYP'] == 'science')
    summary_science = summary_all[matches_run]
    
    print('---')
    print('Index  //  filename  //  objectname')
    print('---')
    for i, (filename, obj) in enumerate(zip(summary_science['file'], summary_science['OBJECT'])):
        print(i, ' // ', filename, ' // ', obj)
    print('---')
    
    cont = False
    while cont is False:
        file_index = input(f'Choose file (select index from 0 to {len(summary_science)-1}) or "all": ')
        if file_index == 'all':
            indexes = list(range(0, len(summary_science)))
            cont = True
        else:
            indexes = [int(file_index)]
            if indexes[0] > (len(summary_science)-1):
                print(f'Incorrect index. Choose an index between 0 and {len(summary_science)-1}.')
            else:
                cont = True
                
    for k, index in enumerate(indexes):      
        science_file = summary_science['file'][index]
        input_filename = science_file
        
        data = fits.getdata(directory / input_filename)
        header = fits.getheader(directory / input_filename)
        
        cunit1 = 1 * u.Unit(header['cunit1'])
        crpix1 = header['crpix1'] * u.pixel
        crval1 = header['crval1'] * u.Unit(cunit1)
        cdelt1 = header['cdelt1'] * u.Unit(cunit1) / u.pixel
        naxis2, naxis1 = data.shape
    
        vmin, vmax = np.percentile(data, [5, 99])
        
        kymin = initial_ymin[k]
        kymax = initial_ymax[k]
        kxmin = initial_xmin[k]
        kxmax = initial_xmax[k]
    
        for iplot in range(2):
            if iplot == 0:
                fig1, ax1 = plt.subplots(figsize=(15, 5))
                tea_imshow(fig1, ax1, data, vmin=0, vmax=500, title=input_filename,
                           cmap='gray', aspect='auto')
                fig1.canvas.draw()  # Force update
                fig1.canvas.flush_events()
                plt.pause(0.1)
            if iplot == 1:
                fig2, ax2 = plt.subplots(figsize=(15, 5))
                tea_imshow(fig2, ax2, data, vmin=0, vmax=500, title=input_filename,
                           cmap='gray', aspect='auto')
                ax2.set_ylim(kymin, kymax)
                ax2.set_xlim(kxmin, kxmax)
                fig2.canvas.draw()  # Force update
                fig2.canvas.flush_events()
                plt.pause(0.1)
            # plt.tight_layout()
            # plt.show()
    
        print('Choose the y range to extract the spectum. INTEGERS ONLY')
            
        cont = False
        first = True
        while cont is False:
            print('---')
            ns1 = int(input('y min: '))
            ns2 = int(input('y max: '))
            if first:
                plt.close(fig1)
                plt.close(fig2)
            else:
                plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(15, 5))
            tea_imshow(fig, ax, data, vmin=0, vmax=500, title=input_filename,
                       cmap='gray', aspect='auto')
            ax.set_ylim(ns1, ns2)
            ax.set_xlim(kxmin, kxmax)
            ax.tick_params(axis='both', labelsize=25)
            # plt.tight_layout()
            # plt.show()
            fig.canvas.draw()  # Force update
            fig.canvas.flush_events()
            plt.pause(0.1)
            
            quest = input('Are you okay with the selected range? (Type "yes" if so): ')
            if quest in ['Yes', 'yes', 'Y', 'y']:
                cont = True
                plt.close(fig)
            else:
                first = False
                
        spectrum = np.sum(data[ns1:(ns2+1), :], axis=0)
        
        wavelength = crval1 + ((np.arange(naxis1) + 1)*u.pixel - crpix1) * cdelt1
        
        wavelength.to(u.Angstrom)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(wavelength.to(u.Angstrom), spectrum)
        #ax.set_xlim(4000, 10000)
        # ax.set_ylim(0, 1500)
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Flux (number of counts)')
        plt.title (input_filename)
           
        hdu = fits.PrimaryHDU(data=spectrum[np.newaxis, :], header=header)
        hdu.writeto(f'{directory}/spectra1D_{input_filename}', overwrite=True)
        print(f'[CONGRATS] 1D spectrum succesfully saved to: {directory}/spectra1D_{input_filename}')
        plt.ioff()
        plt.show()
        if file_index == 'all':
            plt.close()

'''
Flux calibrations
'''

def flux_calib(std_flux_mags=False):
    
    def from_mAB_2_fluxlam(fileabflux, plot=False):
        '''
        flieabflux. Flux of the standard star in AB magnitudes.
        The routine converts it to the usual units of erg/s/cm2/A.
        '''
        t = Table.read(fileabflux, format="ascii")
        wl = t['col1']
        flux_AB = t['col2']
        #fnu is in J and we convert it to cgs
        fnu = 10**((8.9 - flux_AB)/2.5)*(1e-23 * u.erg*u.s**-1 *u.Hz**-1 * u.cm**-2) 
        flam = (fnu * cnt.c / (wl*u.AA)**2).to(u.erg * u.cm**-2 * u.s**-1 * u.AA**-1)
        
        if plot:
            plt.plot(wl, flam, "o-", label="AB to flux")
            plt.ylabel("erg/cm2/s/A")
            plt.xlabel("Wavelength [$\\AA$]")
            
        return Table(np.array([wl, flam]).T, names=["wavelength", "flux"])

    def sigma_clip(x, y, w=15, sigma=10):
        '''
        With a sliding window, of width of w Amstrongs
        it scans the spectrum and discards points that are 
        sigma standard deviations from the median value in the windown.
        '''
        
        yo = np.copy(y)
        wpix = w #np.ceil( 0.5 * w / (x[1]-x[0]))
        mask = np.repeat(False, len(yo))
        for i in np.arange(wpix, len(x[:-wpix])):
            m = np.median(yo[i-wpix:i+wpix])
            s = stats.funcs.median_absolute_deviation(yo[i-wpix:i+wpix])
            clip = np.abs(y[i] - m) > sigma*s
            mask[i-1:i+1] = mask[i-1:i+1] | clip
        if (np.count_nonzero(mask)>0):
            yo[mask] = np.interp(x[mask], x[~mask], yo[~mask])
        return yo

    def read_spec_from_fits(fitsfile, wmin=None, wmax=None):
        '''
        Reads a spectrum from a fits file between wavelengths min and max.
        Returns an astropy table with the columns wavelength and flux.
        '''
        raw_science = fits.open(fitsfile)
    
        if len(raw_science) == 1:
            f = raw_science[0]
            flux = f.data[0]
            init = f.header["CRVAL1"]*1e10
            refpix = f.header["CRPIX1"]
            delta = f.header["CDELT1"]*1e10
            npix = f.header["NAXIS1"]
            exptime = f.header["EXPTIME"]
            wl = init + (np.arange(npix)+1. - refpix)*delta
            
            #create the table with the wavelength and raw flux in counts
            raw_science = Table(np.array([wl, flux]).T, names=["wavelength", "flux"])
            #Divide by total exposure time to convert to rates per second
            raw_science["flux"] = raw_science["flux"] / exptime
            
            #Sigma clip to remove cosmic rays
            #raw_science["flux"] = sigma_clip(raw_science["flux"], raw_science["flux"], sigma=20)
            
        elif len(raw_science) ==2:
            try:
                raw_science = Table(raw_science)
            except:
                raw_science = Table.read(raw_science_file, format="ascii")
    
        print(raw_science.colnames)
        w = raw_science["wavelength"]
        if (not wmin is None) and (not wmax is None):
            mask = (w>wmin)*(w<wmax)
            raw_science = raw_science[mask]
        
        return raw_science

    def apply_flux_calibration(raw_science_file, raw_std_file, abs_flux_std_file, std_flux_mags=std_flux_mags, wmin=3600, wmax=9500, plot=True):
        '''
        The code applies the flux calibration using the following steps.
        
         The code creates a ratio of abs_flux/raw_Std to create a sensitivity "spectrum" that is "flux per counts".
         Then it multiplies that by raw_science to obtain the absolute flux calibration for the science spectrum.
    
    
        "raw_science_file: an uncalibrated (i.e. unit of "counts" or similar) spectrum  
         raw_std_file: an uncalibrated standard star with the same observation setup
         abs_flux_std_file: a calibrated spectrum (i.e. in flux units). 
    
        '''
        
        telluric = [[6860, 6910], [7150, 7300], [7590, 7670]]
        
        #name_obj = raw_science_file.split(".")[0] <-- Changed this
        name_obj = os.path.basename(raw_science_file).split(".")[0]
        
        plotdir = os.path.join(os.path.dirname(raw_science_file), "plots")
        if not os.path.isdir(plotdir):
            print ("Plots will be sotred in {}".format(plotdir))
            os.makedirs(plotdir)
        
        #Read the absolute flux from the standard star from a file.
        #It can be in AB magnitudes or in ergs/s/cm2/A
        if std_flux_mags:
            fluxstd = from_mAB_2_fluxlam(abs_flux_std_file)
        else:
            fluxstd = Table.read(abs_flux_std_file, format="ascii", names=('wavelength', 'flux', 'milli-Jy', 'bandpass'))
            #Flux are in untis of 'ergs/cm2/s/A*10**-16'
            fluxstd["flux"] = fluxstd["flux"]*1e-16
    
        print (fluxstd.colnames)
        wmin = np.maximum(np.min(fluxstd["wavelength"]), wmin)
        wmax = np.minimum(np.max(fluxstd["wavelength"]), wmax)
        print ("Consider between these wavelengths", wmin, wmax)
    
        #Read the raw science files from the fits file.
        if raw_science_file.endswith(".fits"):
            raw_science = read_spec_from_fits(raw_science_file, wmin, wmax)
        else:
            raw_science = Table.read(raw_science_file)
            
    
        #Read the raw STD files from the fits file.
        raw_std = read_spec_from_fits(raw_std_file, wmin, wmax)
    
            
        '''raw_std = fits.open(raw_std_file)
    
        if len(raw_std) == 1:
            f = raw_std[0]
            flux = f.data[0]
            init = f.header["CRVAL1"]*1e10
            refpix = f.header["CRPIX1"]
            delta = f.header["CDELT1"]*1e10
            npix = f.header["NAXIS1"]
            exptime = f.header["EXPTIME"]
            
            #create the table with the wavelength and raw flux in counts
            wl = init + (np.arange(npix)+1. - refpix)*delta
            #Divide by total exposure time to convert to rates per second
            raw_std = Table(np.array([wl, flux]).T, names=["wavelength", "flux"])
            raw_std["flux"] = raw_std["flux"]/exptime
            
            for ti in telluric:
                raw_std["flux"]
            
        elif len(raw_std) ==2:
            try:
                f = raw_std[1]
                raw_std = f.data
                exptime = f.header["EXPTIME"]
                raw_std = Table(raw_std)
                print(raw_std.colnames)
            except:
                raw_std = Table.read(raw_science_file, format="ascii")
                
        w = raw_std["wavelength"]
        mask = (w>wmin)*(w<wmax)
        raw_std = raw_std[mask]
        '''
    
        #Interpolate the raw_std in the same wavelength bins as the absolute flux spectrum. 
        abs_flux_std = np.interp(raw_std["wavelength"], fluxstd["wavelength"], fluxstd["flux"])
    
        sigma = 50 #Smoothing. Larger values may not be good to get the initial and final fluxes.
        ams_per_pix = np.abs(np.median(raw_std["wavelength"][1:]-raw_std["wavelength"][0:-1]))
        sigma = sigma / ams_per_pix
        abs_flux_std_smooth = gaussian_filter1d(abs_flux_std, sigma)
        raw_flux_smooth = gaussian_filter1d(raw_std["flux"], sigma)
    
        #Compute the factor to transform counts to flux units.
        factor =  abs_flux_std_smooth / raw_flux_smooth
        factor_sci = np.interp(raw_science["wavelength"], raw_std["wavelength"], factor)
        
        #The true flux is the raw counts multiplied by the factor
        science_abs = raw_science["flux"] * factor_sci
    
        
        #Write the calibrated flux to file
        mask = (raw_science["wavelength"]>wmin) * (raw_science["wavelength"] < wmax)
        wave_store = raw_science["wavelength"][mask]
        flux_calib_store = raw_science["flux"][mask] * factor_sci[mask]
        t = Table(np.array([wave_store, flux_calib_store]).T, names=["wavelength", "flux"])
        
        
        t.write(os.path.join(os.path.dirname(raw_science_file), \
                             os.path.basename(raw_science_file).replace(".fits", ".txt")), format="ascii", overwrite=True)
        
        
        #Store the header parameters and the table in a fits file.  
        '''header = fits.open(raw_science_file)[0].header
        header['BITPIX'] = 16
        header['NAXIS1'] = len(t)
        header['NAXIS2'] = 2
    
        hdu = fits.PrimaryHDU()
        hdu.data = t
        hdu.header = header
        
        foutname = os.path.join(os.path.dirname(raw_science_file), os.path.basename(raw_science_file).replace(".fits", "_fluxcalib.fits"))
        hdu.write(foutname, overwrite=True)'''
        
        if plot:
            #Plot the raw, standard raw and standard absolute.
            plt.plot(raw_science["wavelength"], raw_science["flux"]/np.median(raw_science["flux"]), label="norm raw science counts/s")
            plt.plot(raw_std["wavelength"], raw_std["flux"]/np.median(raw_std["flux"]), label="norm raw std counts/s")
            plt.plot(fluxstd["wavelength"], fluxstd["flux"]/np.median(fluxstd["flux"]), label="norm abs std flux")
            plt.yscale("log")
            plt.xlabel("Wavelength")
            plt.ylabel("Scaled flux units")
            plt.legend()
            plt.savefig(os.path.join(plotdir, f"{name_obj}_raw_abs_flux.png"))
        
            #Plot the factor
            plt.figure()
            plt.plot(raw_science["wavelength"], factor_sci)
            plt.xlabel("Wavelength")
            plt.ylabel("Factor")
            plt.savefig(os.path.join(plotdir, f"{name_obj}_factor.png"))
    
    
            plt.figure()
            plt.plot(raw_std["wavelength"], raw_std["flux"] * factor, label="raw * factor")
            plt.plot(raw_std["wavelength"], abs_flux_std, label="abs flux")
            plt.plot(raw_std["wavelength"], abs_flux_std_smooth, label="abs smooth")
            ymin, ymax = plt.ylim()
            #Mark Balmer lines
            plt.vlines(6532, ymin, ymax, color="k", linewidth=0.2)
            plt.vlines(4861.33, ymin, ymax, color="k", linewidth=0.2)
            plt.vlines(4340.47, ymin, ymax, color="k", linewidth=0.2)
            plt.vlines(4101.76,ymin, ymax, color="k", linewidth=0.2)
            plt.legend()
            plt.yscale("log")
            plt.xlabel("Wavelength")
            plt.ylabel('ergs/cm2/s/A*10$^{-16}$')
            plt.savefig(os.path.join(plotdir, f"{name_obj}_std_flux.png"))
    
    
            #Plot Halpha in the STD star to verify wavelength calibration
            radius=100
            plt.figure()
            mask = np.abs(raw_std["wavelength"] - 6563)<radius
            plt.plot(raw_std["wavelength"][mask], raw_std["flux"][mask], label="raw_std")
            #plt.xlim(6532*(1+z)-100, 6532*(1+z)+100)
            ymin, ymax = plt.ylim()
            plt.vlines(6532, ymin, ymax, color="k", linewidth=0.2)
            plt.xlabel("Wavelength")
            plt.ylabel('ergs/cm2/s/A*10$^{-16}$')
            plt.title("Halpha region")
            plt.savefig(os.path.join(plotdir, f"{name_obj}_science_flux_halpha.png"))
    
    
    
            plt.figure()
            plt.plot(raw_science["wavelength"], raw_science["flux"] * factor_sci, label="raw * factor")
            ymin = np.min(raw_science["flux"] * factor_sci)
            ymax = np.max(raw_science["flux"] * factor_sci)
            plt.vlines(6563, ymin, ymax, color="k", linewidth=0.2)
            plt.vlines(4861.33, ymin, ymax, color="k", linewidth=0.2)
            plt.vlines(4340.47, ymin, ymax, color="k", linewidth=0.2)
            plt.vlines(4101.76, ymin, ymax, color="k", linewidth=0.2)
            plt.legend()
            #plt.yscale("log")
            plt.xlabel("Wavelength")
            plt.ylabel('ergs/cm2/s/$\\AA$')
            plt.savefig(os.path.join(plotdir, f"{name_obj}_science_flux_1.png"))
    
    # Set the raw_science to the name of the reduced fits file containt the science object.
    # The raw_std to the reduced file containing the standard star (same grism and slit as the target).
    # The abs_flux_std to the path to the file wite the absolute flux calibration.
    
    ifc_all, summary_all, _ = read_images(img_code='spectra1D*', plot=False)
    
    matches_run = (summary_all['IMAGETYP'] == 'science')
    summary_science = summary_all[matches_run]
    
    print('---')
    print('Index  //  filename  //  objectname')
    print('---')
    for i, (filename, obj) in enumerate(zip(summary_science['file'], summary_science['OBJECT'])):
        print(i, ' // ', filename, ' // ', obj)
    print('---')
    
    cont = False
    while cont is False:
        file_index = int(input(f'Choose a file to flux calibrate (select index from 0 to {len(summary_science)-1}): '))
        if file_index > (len(summary_science)-1):
            print(f'Incorrect index. Choose an index between 0 and {len(summary_science)-1}.')
        else:
            cont = True       
    science_file = summary_science['file'][file_index]
    
    cont = False
    while cont is False:
        file_index = int(input(f'Choose a file with the observed standard star data (select index from 0 to {len(summary_science)-1}): '))
        if file_index > (len(summary_science)-1):
            print(f'Incorrect index. Choose an index between 0 and {len(summary_science)-1}.')
        else:
            cont = True       
    standard_file = summary_science['file'][file_index]
    
    absolute_standards = os.listdir('Standard stars')
    
    print('---')
    print('Index  //  filename  //  objectname')
    print('---')
    for i, filename in enumerate(absolute_standards):
        obj_name = filename[1:-4].upper().replace('_', '-')
        print(i, ' // ', filename, ' // ', obj_name)
    print('---')
    
    cont = False
    while cont is False:
        file_index = int(input(f'Choose a file with the absolute flux calibration (select index from 0 to {len(absolute_standards)-1}): '))
        if file_index > (len(absolute_standards)-1):
            print(f'Incorrect index. Choose an index between 0 and {len(absolute_standards)-1}.')
        else:
            cont = True
    abs_standard_file = absolute_standards[file_index]
    
    raw_science = os.path.join(DATADIR, science_file)
    
    raw_std = os.path.join(DATADIR, standard_file)
    
    abs_flux_std = os.path.join('Standard stars', abs_standard_file)
    
    apply_flux_calibration(raw_science, raw_std, abs_flux_std, std_flux_mags=False, wmin=3800, wmax=10000)
    
    name = os.path.join(os.path.dirname(raw_science), \
                         os.path.basename(raw_science).replace(".fits", ".txt"))
    print(f'Flux calibrated spectrum saved to: {name}')
