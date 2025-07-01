import multiprocessing
import numpy as np
import os
import platform
import re
import subprocess as sp
import time
from astropy.io import fits, ascii
from astropy.table import Table
from itertools import repeat
from PIL import Image


#==============================================================================
#
#
#
#==============================================================================

# NAMES
SEXTRACTOR_STR = "source-extractor"


# DIRECTORIES
DIR_STARS = "sext"
DIR_SCRIPT = os.getcwd()
DIR_PROJ_CAT_RPI = "{}/Catalog/RPi/Projected".format(DIR_SCRIPT)
DIR_PROJ_CAT_STEREO = "{}/Catalog/STEREO/Projected".format(DIR_SCRIPT)
DIR_NORMAL_CAT_RPI = "{}/Catalog/RPi/Normal".format(DIR_SCRIPT)
DIR_NORMAL_CAT_STEREO = "{}/Catalog/STEREO/Normal".format(DIR_SCRIPT)
NEW_PROJ_CAT = 'new_cat'


# ALGORITHM PARAMETERS
SEXTRACTOR_MAX_STARS = 40
X_PIX = 512                 # x_center, half image width  (image width  = 1024)
Y_PIX = 512                 # y_center, half image height (image height = 1024)
PIX2MM_RPI    = 0.002695    # Raspberry Pi camera pixel size  (mm)
PIX2MM_STEREO = 0.027021    # STEREO camera pixel size        (mm)
LIS_MAX_ITER = 3            # 3 matching iterations
FOCAL_LEN_MM_RPI = 3.04     # RPi camera lens focal length     (mm)
FOCAL_LEN_MM_STEREO = 78.46 # STEREO camera lens focal length  (mm)


# MATCH PARAMETERS
PARAM1 = "trirad=0.002 nobj=15 max_iter=1 matchrad=0.1 scale=1"
PARAM2 = "trirad=0.002 nobj=20 max_iter=5 matchrad=0.01 scale=1"
# REGULAR EXPRESSIONS
MATCH_STD = re.compile(r"sig=(-*\d\.\d+e...) Nr=(-*\d+) Nm=(-*\d+) sx=(-*\d\.\d+e...) sy=(-*\d\.\d+e...)")
MATCH_NUMBERS = re.compile(r"a=(-*\d\.\d+e...) b=(-*\d\.\d+e...) c=(-*\d\.\d+e...) "
                           r"d=(-*\d\.\d+e...) e=(-*\d\.\d+e...) f=(-*\d\.\d+e...)")


#==============================================================================
#
#
#
#==============================================================================

def jpg2fits(full_dir_img, fits_name_of_image):
    """
    Generate .fits from .jpg image.
    """
    image = Image.open(full_dir_img)

    # "L": Grayscale (8-bit pixels, black and white).
    # "RGB": True Color (3x8-bit pixels, red, green, blue).
    # "RGBA": True Color with Alpha (4x8-bit pixels, red, green, blue, alpha transparency).
    # "P": Paletted (8-bit pixels, mapped to a color palette).
    # "1": Black and White (1-bit pixels, black or white).
    image_bw = image.convert('L')       # grayscale

    x_size, y_size = image_bw.size

    print(f'x_size =  {x_size}    y_size =  {y_size}\n')

    fits1 = image_bw.getdata()
    if platform.machine().endswith('64'):
        fits2 = np.array(fits1, dtype=np.int32)
    else:
        fits2 = np.array(fits1)
    
    #print(f'fits2.size =  {fits2.size}\n')

    fits3 = fits2.reshape(y_size, x_size)
    
    #print(f'fits3.size =  {fits3.size}\n')

    #flip the up/down order
    # In FITS images, the row direction typically increases upwards, meaning
    # the first pixel (row, column) is in the bottom-left corner, with rows
    # increasing towards the top
    #
    # JPG images, on the other hand, generally have a row direction that
    # increases downwards, similar to how they are displayed in web browsers
    # or image viewers. 
    #
    fits4 = np.flipud(fits3)

    fits5 = fits.PrimaryHDU(data=fits4)
    fits5.writeto(fits_name_of_image, overwrite=True)

    #print(f'fits5 =  {fits5}\n')

    print(f'fits_name_of_image =  {fits_name_of_image}\n')
    return 0


#------------------------------------------------------------------------------
# call SEXTRACTOR via shell command
#
#   source-extractor img.fits
#
#------------------------------------------------------------------------------
def apply_sextractor(img_fits_name, stt_data_dir, lis_type="rpi"):
    """
    Apply Source Extractor over an image and generates a catalog.
    """

    # get the camera pixel size (mm)
    if lis_type == "rpi":
        pix2mm = PIX2MM_RPI         # pixel size (mm) of RPi camera
    elif lis_type == "stereo":
        pix2mm = PIX2MM_STEREO      # pixel size (mm) of STEREO camera
    else:
        raise NameError("ERROR: See function apply_sextractor")
    
    # change DIR to /home/.../RPi/stt_data
    os.chdir(stt_data_dir)

    # construct the shell command with argument
    sext_task = "{} {}".format(SEXTRACTOR_STR, img_fits_name)

    print(f'sext_task =  {sext_task}')

    #--------------------------------------------
    # call SEXTRACTOR
    #
    # create a shell with subprocess (sp) and run the external command 
    #   'source-extractor img.fits'
    #
    # configuration files:
    #   stt_data/default.sex            config how SEXTRACTOR works
    #       stt_data/default.param      config which fileds should be included
    #                                   in the output catalog file (stt_data/test.cat)
    #       stt_data/default_conv       config the filter
    #
    #--------------------------------------------

    # execute an external command and capture its stdout
    print('-----------------------------------------------')
    ret = sp.check_output(sext_task, shell=True)
    print('-----------------------------------------------')
    print(f'ret =  {ret}\n')


    #--------------------------------------------
    # read the output catalog file generated by SExtractor
    # sort the catalog from brightest stars
    # extract X, Y, Mag
    # calculate x_mm, y_mm
    # output stt_data/sext
    #   x_mm, y_mm, mag
    #
    # astropy.ascii.read()
    #
    # type = astropy.table.table.Table
    #   NUMBER MAG_ISO MAGERR_ISO  X_IMAGE   Y_IMAGE  FLAGS
    #--------------------------------------------
    sext1 = ascii.read('./test.cat', format='sextractor')

    #print(f'type(sext1) =  {type(sext1)}\n')
    #print(f'sext1 =\n{sext1}\n')

    # sort according to sext1['MAG_ISO'], ascending
    sext1.sort(['MAG_ISO'])
    #print(f'sorted sext1 =\n{sext1}\n')


    # take only the 40 brightest magnitude stars 
    sext2 = sext1[0:SEXTRACTOR_MAX_STARS]
    #print(f'sext2 (the 40 brightest stars)=\n  {sext2}\n')

    # extract X_IMAGE & Y_IMAGE
    # type = astropy.table.column.Column
    sext_x_pix = sext2['X_IMAGE']
    sext_y_pix = sext2['Y_IMAGE']
    #print(f'type(sext_x_pix) =  {type(sext_x_pix)}\n')
    #print(f'sext_x_pix =\n  {sext_x_pix}\n')


    # extract MAG_ISO
    # type = astropy.table.column.Column
    sext_mag = sext2['MAG_ISO']

    # half image width & height to physical size (mm)
    sext_x_mm = (sext_x_pix - X_PIX) * pix2mm
    sext_y_mm = (sext_y_pix - Y_PIX) * pix2mm

    #print(f'sext_x_mm =\n  {sext_x_mm}\n')

    # output to 'stt_data/sext
    #   sext_x_mm   sext_y_mm   sext_mag
    #  
    sext_filename = 'sext'
    ascii.write([sext_x_mm, sext_y_mm, sext_mag], sext_filename, delimiter=' ', format='no_header', overwrite=True,
                formats={'X': '% 15.10f', 'Y': '% 15.10f', 'Z': '% 15.10f'}, names=['X', 'Y', 'Z'])
    

    return sext_x_pix, sext_y_pix, sext_x_mm, sext_y_mm




def get_catalog_center_points(x_center, y_center, distance, lis_type="rpi"):
    """
    Get the center points for different catalogs segments, for a given distance and starting point.
    It considers declination of center site.
    """

    if lis_type == "rpi":
        catalog_center_list = list()

        for jj1 in range(y_center, 90, distance):
            aux1 = (1 / np.cos(np.deg2rad(jj1)))
            #distance_ra1 = int(round(distance * aux1))
            distance_ra1 = distance

            print(f'aux1 =  {aux1:6.3f}    distance_ra1 =  {distance_ra1}')

            for ii1 in range(x_center, 360, distance_ra1):
                 catalog_center_list.append([ii1, jj1])

        print()

        for jj2 in range(y_center - distance, -90, -distance):
            aux2 = (1 / np.cos(np.deg2rad(jj2)))
            #distance_ra2 = int(round(distance * aux2))
            distance_ra2 = distance

            print(f'aux2 =  {aux2:6.3f}    distance_ra2 =  {distance_ra2}')

            for ii2 in range(x_center, 360, distance_ra2):
                catalog_center_list.append([ii2, jj2])


        catalog_center_list = catalog_center_list + [[0, 90], [0, -90]]

    elif lis_type == "stereo":
        catalog_center_list = [(ra, dec) for ra in range(0, 360, distance) for dec in range(-85, 90, distance)] + [(0, 90), (0, -90)]
    else:
        return NameError("ERROR: See function get_catalog_center_points")
    return catalog_center_list


# This is for the first BLIND SOLVE
#   . All of the selected (according to the distance specified) catalog segment files
#     will be tried
#
# Input:
#   ra_dec_list             list of all the selected catalog segment ceter ra, dec
#
# Return:
#   first_match_results     [(status, result), ...]
#
#   status  result
#   -----------------------------------------
#   0       TRANS: a=4.428697608e-02 b=8.645333719e-02 c=-1.024640887e+00 d=2.314751124e-01 e=-1.025812493e+00 f=-6.849823407e-02 sig=5.6740e-04 Nr=29 Nm=33 sx=2.6295e-02 sy=2.8340e-02
#
#   1       atFindTrans: iter_trans unable to create a valid TRANS
#           initial call to atFindTrans fails
#
def map_match_and_radec_list_multiprocess(ra_dec_list, lis_type="rpi"):
    """
    Apply 'call_match' multiprocessing depending on numbers of cores.
    """
    n_cores = multiprocessing.cpu_count()

    #n_cores = 1
    print(f'n_cores =  {n_cores}\n')

    # call call_match_list() to let Match do the work
    #
    if n_cores == 1:
        if lis_type == "rpi":
            first_match_results = map(call_match_list, ra_dec_list)
        else:
            first_match_results = map(lambda p: call_match_list(p, lis_type="stereo"), ra_dec_list)
    else:
        # dispatch the jobs to multiple cores 
        
        # multiprocessing.Pool.map()
        #   . blocking call
        #   . return results are placed in the same order as the input interable,
        #     even if the processes finish out of order
        #   

        '''
        pool = multiprocessing.Pool(n_cores)

        #print(f'pool =  {pool}\n')

        if lis_type == "rpi":
            # each call_match_list() call will have only one catalog segment assigned
            first_match_results = pool.map(call_match_list, ra_dec_list)
        else:
            first_match_results = pool.starmap(call_match_list, zip(ra_dec_list, repeat("stereo")))
        '''

        # with the 'with' context manager, the poo.close() and p.join() are
        # called automatically when 'with' block is exited
        #
        with multiprocessing.Pool(n_cores) as pool:
            #print(f'pool =  {pool}\n')

            if lis_type == "rpi":
                # each call_match_list() call will have only one catalog segment assigned
                first_match_results = pool.map(call_match_list, ra_dec_list)
            else:
                first_match_results = pool.starmap(call_match_list, zip(ra_dec_list, repeat("stereo")))

    return first_match_results



#------------------------------------------------------------------------------
# call Match to do the matching work
#
# matching is done with PROJECTED ( (x, y) in unit of mm) image and catalog segment
#
# souce star list file (projected, mm)
#   stt_data/sext
#
# projected (mm) catalog segments are used for matching
#   Catalog/RPi/Projected/cat_RA_XXX_DEC_XXX
#
#   ra_dec_list  =  the catalog segment ceter (ra, dec) list, 
#                   should have only one element when called by map()
#
# return:   
#   status = returned exit code
#   result = stdout output of Match
#
#------------------------------------------------------------------------------
def call_match_list(ra_dec_list, lis_type="rpi", base='catalog'):
    """
    Call 'Match' with RA/DEC list in the shell.
    """

    #print(f'call_match_list:  type(ra_dec_list) =  {type(ra_dec_list)}')
    print(f'call_match_list:  ra_dec_list =  {ra_dec_list}\n')


    # construct the external command with parameters, which will be executed in linux shell, for Match
    #
    # Example:
    #   match  =  match sext 0 1 2 /home/galileo/Projects/StarTracker/SOST/Star_Tracker/RPi/Catalog/RPi/Projected/cat_RA_210_DEC_-50 0 1 2 trirad=0.002 nobj=15 max_iter=1 matchrad=0.1 scale=1
    #
    ra, dec = ra_dec_list
    if base == 'catalog':
        if lis_type == "rpi":
            # Match 'sext' with Catalog/RPi/Projected
            # Match parameters:  "trirad=0.002 nobj=15 max_iter=1 matchrad=0.1 scale=1"
            match = set_match_str(ra, dec, DIR_STARS, DIR_PROJ_CAT_RPI, PARAM1)
        elif lis_type == "stereo":
            match = set_match_str(ra, dec, DIR_STARS, DIR_PROJ_CAT_STEREO, PARAM1)
        else:
            raise NameError("ERROR: See function call_match_list ")
        #print(f'match  =  {match}')

    elif base == 'picture':
        if lis_type == "rpi":
            match = set_match_str(ra, dec, DIR_STARS, DIR_PROJ_CAT_RPI, PARAM1, base='picture')
        elif lis_type == "stereo":
            match = set_match_str(ra, dec, DIR_STARS, DIR_PROJ_CAT_STEREO, PARAM1, base='picture')
        else:
            raise NameError("ERROR: See function call_match_list ")
    else:
        raise NameError("---> ERROR: Select a valid base for Match!")
    

    # execute the external command (match) in linux shell via python subprocess, 
    # get return code and output
    #   status = returned exit code
    #   result = stdout output
    #
    #   status  result
    #   -----------------------------------------
    #   0       TRANS: a=4.428697608e-02 b=8.645333719e-02 c=-1.024640887e+00 d=2.314751124e-01 e=-1.025812493e+00 f=-6.849823407e-02 sig=5.6740e-04 Nr=29 Nm=33 sx=2.6295e-02 sy=2.8340e-02
    #
    #   1       atFindTrans: iter_trans unable to create a valid TRANS
    #           initial call to atFindTrans fails

    status, result = sp.getstatusoutput(match)

    '''
    #if status != 1:
    print(f'call_match_list:')
    print(f'  status =  {status}')
    print(f'  result =  {result}')
    print()
    '''

    return status, result





def set_match_str(ra, dec, dir_stars, dir_proj_cat, param, base='catalog'):
    """
    Define and set the 'match' string before calling it in the shell.
    """
    ra_dec_str = "cat_RA_{}_DEC_{}".format(ra, dec)
    if base == 'catalog':
        match_str = "match {} 0 1 2 {}/{} 0 1 2 {}".format(dir_stars, dir_proj_cat, ra_dec_str, param)
        # print(match_str)
    elif base == 'picture':
        match_str = "match {}/{} 0 1 2 {} 0 1 2 {}".format(dir_proj_cat, ra_dec_str, dir_stars, param)
    else:
        raise NameError("---> ERROR: Select a valid base for Match!")
    
    return match_str


# Input:
#   first_match_map_results = [(status, result), ...]
#
#   status  result
#   -----------------------------------------
#   0       TRANS: a=4.428697608e-02 b=8.645333719e-02 c=-1.024640887e+00 d=2.314751124e-01 e=-1.025812493e+00 f=-6.849823407e-02 sig=5.6740e-04 Nr=29 Nm=33 sx=2.6295e-02 sy=2.8340e-02
#
#   1       atFindTrans: iter_trans unable to create a valid TRANS
#           initial call to atFindTrans fails
#
# Return:
#   match_table = [ (ra  dec  sig  Nr), ... ], sorted by 'Nr'
#
def get_table_with_matchs(ra_dec_list, first_match_results):
    """
    Select RA/DEC values in which a successful match was obtained, and generate a 'match' table.
    """
 
    # astropy.Table
    #
    # sig       =  residual between matched pairs after transformation
    # Nr        =  number of reference stars uded
    #
    match_table = Table(names=('RA_center', 'DEC_center', 'sig', 'Nr'))

    for i, (status, result) in enumerate(first_match_results):
        if status == 0:
            # match found
            ra, dec = ra_dec_list[i]

            # use REGULAR EXPRESSION to parse results and extract the selected items 
            #a = MATCH_STD.findall(result)
            regexp_result = MATCH_STD.findall(result)[0]

            print(f'segment(ra, dec) =  {ra}, {dec}')
            #print(f'a      =  {a}')
            #print(f'result =  {result}')
            #print(f'regexp_result =  {regexp_result}')
            #print()

            # match_table =
            #   (ra  dec  sig  Nr), ...
            sig = float(regexp_result[0])
            nr = int(regexp_result[1])
            match_table.add_row([str(ra), str(dec), sig, nr])

    if len(match_table) == 0:
        print("--> After search in the whole catalog, I can not find any match between picture and catalog! :(")
        raise ValueError("---> ERROR: There is no match ...")
    else:
        match_table.sort('Nr')
        match_table.reverse()

    #   match_table = [ (ra  dec  sig  Nr), ... ]
    print(f'match_table =\n{match_table}')

    return match_table



# get the first (up to) 3 matched candidate from the match_table
#
# Input:
#   match_table = [ (ra  dec  sig  Nr), ... ]
#
# Return:
#   match_table[0:3]
#
def get_match_candidates(match_table):
    """
    Select three 'match' candidates from 'first match table'.
    """

    len_table = len(match_table)
    if len_table == 1:
        match_candidates = match_table
    elif len_table == 2:
        match_candidates = match_table
    else:
        match_candidates = match_table[0:3]
    return match_candidates


#------------------------------------------------------------------------------
# call Match again with match_candidates to obtain the matching data
#
# candidates:
#   projected objects from image
#
#------------------------------------------------------------------------------
def get_first_match_data(candidates, try_n, lis_type="rpi", base="catalog"):
    """
    With a list of 'match candidates', recall match to obtain the transformation relationship.
    """
    cand = candidates[try_n]

    # call Match to do the matching again
    #
    if base == "catalog":
        if lis_type == "rpi":
            result = call_match_list([int(cand[0]), int(cand[1])])[1]
        elif lis_type == "stereo":
            result = call_match_list([int(cand[0]), int(cand[1])], lis_type="stereo")[1]
        else:
            raise NameError("ERROR: See function get_first_match_data")
        
    elif base == "picture":
        print("Base is picture!")
        if lis_type == "rpi":
            result = call_match_list([int(cand[0]), int(cand[1])], base="picture")[1]
        elif lis_type == "stereo":
            result = call_match_list([int(cand[0]), int(cand[1])], lis_type="stereo", base="picture")[1]
        else:
            raise NameError("ERROR")
        
    else:
        raise ValueError("---> ERROR: Select a valid base for Match!")
    
    print(f'result =  {result}\n')

    # extract a,b,c,d,e,f from the result via REGUALR EXPRESSION 
    #a = MATCH_NUMBERS.findall(result)
    regexp_result_numbers = MATCH_NUMBERS.findall(result)[0]

    #print(f'type(a) =  {type(a)}')
    #print(f'a =  {a}')
    #print(f'type(regexp_result_numbers) = {type(regexp_result_numbers)}')
    print(f'regexp_result_numbers =  {regexp_result_numbers}')

    # extract sig, Nr, Nm, sx, sy from the result via REGULAR EXPRESSION
    #
    # sig   : The typical residual (e.g., in radians squared) between matched
    #         pairs after transformation, indicating fit quality (default halt
    #         condition is 1.0e-12).
    # Nr    : Number of reference stars used
    # Nm    : Number of matched pairs
    # sx, sy: Standard deviations of residuals in x and y directions
    
    regexp_result_std = MATCH_STD.findall(result)[0]


    print(f'regexp_result_std =  {regexp_result_std}')
    print()

    return regexp_result_numbers, regexp_result_std




def apply_match_trans(data):
    """
    Apply the 'match' transformation between picture and projected catalog.
    This is: center of picture (pix) ==> point in the projected catalog (mm).
    """

    # sensor plane coordinates
    #
    #   x' = a + bx + cy
    #   y' = d + ex + fy
    #

    match_a = float(data[0])
    match_b = float(data[1])
    match_c = float(data[2])
    match_d = float(data[3])

    match_ra_mm, match_dec_mm = match_a, match_d
    
    match_roll_rad = np.arctan2(match_c, match_b)
    match_roll_deg = np.rad2deg(match_roll_rad)
    return match_ra_mm, match_dec_mm, match_roll_deg


# Inverse gnomonic project
#
# Mapping pixel coordinates back to celestial coordinates
#
# eta (η),  xi (ξ)
#
def plane2sky(ra_match_mm, dec_match_mm, ra_catalog, dec_catalog, lis_type="rpi"):
    """
    Deproject any arbitrary point in the camera. This is: mm ==> sky coordinates.
    """
    if lis_type == "rpi":
        focal_len_mm = FOCAL_LEN_MM_RPI
    elif lis_type == "stereo":
        focal_len_mm = FOCAL_LEN_MM_STEREO
    else:
        raise NameError("ERROR")
    
    # ??? why
    xi = ra_match_mm / float(focal_len_mm)
    eta = dec_match_mm / float(focal_len_mm)
    
    dec_catalog_rad = np.deg2rad(dec_catalog)
    arg1 = np.cos(dec_catalog_rad) - eta * np.sin(dec_catalog_rad)
    arg2 = np.arctan(xi / arg1)
    arg3 = np.sin(arg2)
    arg4 = eta * np.cos(dec_catalog_rad) + np.sin(dec_catalog_rad)

    #???    
    alpha = ra_catalog + np.rad2deg(arg2)
    delta = np.rad2deg(np.arctan((arg3 * arg4) / xi))
    return alpha, delta



def search_catalog_objects(ra_first_match, dec_first_match, lis_type="rpi"):
    """
    Search in the normal catalog for all sky-objects (to the nearest catalog), and create a table.
    """

    # contstruct the NORMAL catalog segment file name according to the RA & DEC specified
    #
    if lis_type == "rpi":
        ra_catalog = int(round(ra_first_match))
        dec_catalog = int(round(dec_first_match))
        new_cat_name = "{}/cat_RA_{}_DEC_{}".format(DIR_NORMAL_CAT_RPI, ra_catalog, dec_catalog)
    elif lis_type == "stereo":
        ra_catalog = int(5 * round(float(ra_first_match)/5))
        dec_catalog = int(5* round(float(dec_first_match)/5))
        new_cat_name = "{}/cat_RA_{}_DEC_{}".format(DIR_NORMAL_CAT_STEREO, ra_catalog, dec_catalog)
    else:
        raise NameError("ERROR: See function search_catalog_objects")
    
    print()
    print(f'new_cat_name =  {new_cat_name}\n')

    # read the NORMAL catalog segment file
    new_cat = ascii.read(new_cat_name)

    # construct the catalog segment table
    # row-by-row
    #
    noproj_table = Table([[], [], []])
    for ii in range(len(new_cat)):
        noproj_table.add_row([new_cat[ii][0], new_cat[ii][1], new_cat[ii][2]])
    
    print(f'noproj_table =\n{noproj_table}\n')

    return noproj_table

# Gnomonic projection with the specified projection point
#   projected objects are store in the file stt_data_/new_cat
#
# Output:
#   stt_data/new_cat
#
def sky2plane(star_list, ra_project_point, dec_project_point, lis_type="rpi"):
    """
    With a list of 'matched' stars, project all in the tangent plane.
    """
    cat_projected = Table([[], [], []])
    stars_len = len(star_list)

    if lis_type == "rpi":
        focal_len_mm = FOCAL_LEN_MM_RPI
    elif lis_type == "stereo":
        focal_len_mm = FOCAL_LEN_MM_STEREO
    else:
        raise NameError("ERROR: See function sky2plane")
    
    for index in range(stars_len):
        # RA, DEC, MAG from the catalog
        alpha_deg = star_list[index][0]
        delta_deg = star_list[index][1]
        mag = star_list[index][2]

        alpha_rad = np.deg2rad(alpha_deg)
        delta_rad = np.deg2rad(delta_deg)
        
        alpha_0_rad = np.deg2rad(ra_project_point)
        delta_0_rad = np.deg2rad(dec_project_point)
        
        #----------------------------------------
        # Gnomonic projection
        #----------------------------------------

        # xi (ξ)
        #
        # numerator
        xi_up = np.cos(delta_rad) * np.sin(alpha_rad - alpha_0_rad)
        # denominator
        xi_down = np.sin(delta_0_rad) * np.sin(delta_rad)\
            + np.cos(delta_0_rad) * np.cos(delta_rad) * np.cos(alpha_rad - alpha_0_rad)
        xi = xi_up/xi_down

        # eta (η)
        # numerator
        eta_up = np.cos(delta_0_rad) * np.sin(delta_rad)\
            - np.sin(delta_0_rad) * np.cos(delta_rad) * np.cos(alpha_rad - alpha_0_rad)
        # denominator
        eta_down = xi_down
        eta = eta_up / eta_down

        xi_mm = xi * focal_len_mm
        eta_mm = eta * focal_len_mm
        cat_projected.add_row([xi_mm, eta_mm, mag])

    cat_name = 'new_cat'
    ascii.write(cat_projected, cat_name, delimiter=' ', format='no_header', overwrite=True,
                formats={'X': '% 15.5f', 'Y': '% 15.5f', 'Z': '% 15.2f'}, names=['X', 'Y', 'Z'])
    return 0


#------------------------------------------------------------------------------
# Invoke the shell and execute Match command
#
# Return:
#    regexp_result_numbers :   a, b, c, d, e, f
#    regexp_result_std     :   sig, Nr, Nm, sx, sy
#
#------------------------------------------------------------------------------
def call_match_once(base="catalog", outfile=None):
    """
    Call 'Match' one time, in further 'match' iterations.
    """

    # sext, new_cat
    # PARAM2 = "trirad=0.002 nobj=20 max_iter=5 matchrad=0.01 scale=1"
    #
    # the matchrad is tighter than the 1st iteration
    #
    # match_str
    #   the shell command to execute Match
    #
    if base == "catalog":
        match_str = "match {} 0 1 2 {} 0 1 2 {}".format(DIR_STARS, NEW_PROJ_CAT, PARAM2)
    elif base == "picture":
        match_str = "match {} 0 1 2 {} 0 1 2 {}".format(NEW_PROJ_CAT, DIR_STARS, PARAM2)
    else:
        raise ValueError("---> ERROR: Select a valid base for Match!")
    
    
    if outfile is not None:
        match_str = match_str + ' outfile=' + outfile

    print(f'match_str =  {match_str}')

    # invoke the shell to execute the command
    #   Popen.check_output()
    #
    #   status = exit code
    #   result = output
    #
    status, result = sp.getstatusoutput(match_str)
    
    print("Status: ", status)
    print("Result: ", result)

    # regexp_result_numbers :   a, b, c, d, e, f
    # regexp_result_std     :   sig, Nr, Nm, sx, sy
    #
    regexp_result_numbers = MATCH_NUMBERS.findall(result)[0]
    regexp_result_std = MATCH_STD.findall(result)[0]

    print(f'regexp_result_numbers =  {regexp_result_numbers}')
    print(f'regexp_result_std     =  {regexp_result_std}')

    print()

    return regexp_result_numbers, regexp_result_std




def solve_lis(img_full_dir, catalog_division, stt_data_dir, lis_type="rpi"):
    """
    Solve the Lost-In-Space problem.
    """

    #--------------------------------------------
    # *.jpg -> img.fits
    #
    # From:
    #   Sample_images/RPi/img_x.jpg
    # To:
    #   stt_data/img.fits
    #
    #--------------------------------------------
    tm1 = time.time()

    # Apply SExtractor.
    if lis_type == "rpi":
        str_fits = "img.fits"
        img_fits_name = "{}/{}".format(stt_data_dir, str_fits)
        jpg2fits(img_full_dir, img_fits_name)
    elif lis_type == "stereo":
        str_fits = img_full_dir
    else:
        raise NameError("ERROR: Please introduce a valid lis_type parameter <rpi> or <stereo>")

    #print(f'img_full_dir  =  {img_full_dir}')
    #print(f'stt_data_dir  =  {stt_data_dir}')
    #print(f'img_fits_name =  {img_fits_name}')
    #print()

    #--------------------------------------------
    # call SEXTRACTOR
    # generates
    #   stt_data/test.cat
    #       . The selected info of all extracted stars in the image
    #   stt_data/sext
    #       . the x_mm, y_mm, mag of the brightest magnitude 40 stars in the image
    #
    #--------------------------------------------
    print(f'apply_sextractor:')
    print(f'  str_fits     =  {str_fits}')
    print(f'  stt_data_dir =  {stt_data_dir}')
    print(f'  lis_type     =  {lis_type}')
    print()

    # extract stars
    # generates
    #   stt_data/test.cat
    #   stt_data/sext
    #
    apply_sextractor(str_fits, stt_data_dir, lis_type)



    #--------------------------------------------
    # Apply Match - First iteration.
    #
    #--------------------------------------------

    # find all of the segment centers accroding to the distance spceified
    #
    ra_dec_list = get_catalog_center_points(0, 0, catalog_division, lis_type)
    #ra_dec_list = get_catalog_center_points(45, 45, catalog_division, lis_type)

    print(f'ra_dec_list =\n  {ra_dec_list}\n')
 

    # dispatch the Match process to all cores for the selected (according to the segment distance)
    # catalog segments
    #   ra_dec_list     = the segment ceter list
    #
    # return results (in the same order as ra_dec_list): 
    #   first_match_map_results = [(status, result), ...]
    #
    first_match_map_results = map_match_and_radec_list_multiprocess(ra_dec_list, lis_type)

    #print(f'first_match_map_results =  {first_match_map_results}')


    # first_match_table = [ (ra  dec  sig  Nr), ... ], all of the match
    # candidates, sorted by 'Nr'
    #
    first_match_table = get_table_with_matchs(ra_dec_list, first_match_map_results)

    # print(first_match_table)

    # Select (up to) 3 'match' candidate catalog segments from 'first match table'
    #   match_candidates = [ (ra  dec  sig  Nr), ... ]
    #
    match_candidates = get_match_candidates(first_match_table)
    print("\n---> Match candidates:")
    print(match_candidates)


    attempts = 0
    third_alpha, third_delta, third_roll_deg, third_match_std = (0, 0, 0, [0, 0])

    # max 3 attempts  (if error occurs)
    #
    while attempts < LIS_MAX_ITER:
        try:
            # from the matched candidate segments,
            # get the ra_center & dec_center
            first_ra_catalog, first_dec_catalog = match_candidates[attempts][0], match_candidates[attempts][1]

            print("attempts= {}  ra= {}  dec= {}".format(attempts, first_ra_catalog, first_dec_catalog))
            
            # call Match again with match_candidates to obtain the matching data
            #   match_candidates = [ (ra  dec  sig  Nr), ... ]
            #                    =>  the first 3 candidate segments
            #
            #   first_match_data = a, b, c, d, e, f
            #   first_match_std  = sig, Nr, Nm, sx, sy
            #
            # sensor plane coordinates
            #   x' = a + bx + cy
            #   y' = d + ex + fy
            #

            first_match_data, first_match_std = get_first_match_data(match_candidates, attempts, lis_type)
            
            print(f'first_match_data =\n  {first_match_data}')


            #------------------------------------
            # 2nd. iterations
            #
            # . calculate the ra & dec of the IMAGE CENTER
            # . select and read new NORMAL catalog segment according to the IMAGE
            #   CENTER ra & dec
            # . create the projected catalog of the newly selected NORMAL catalog
            #   segment with IMAGE CENTER ra & dec as the project point 
            # . call Match to match the image with the new projected catalog
            # .
            #------------------------------------


            #--------------------------
            # calculate the ra & dec of the IMAGE CENTER
            #--------------------------

            # . (x, y)   is the image pixel locations (mm)
            # . (x', y') is the transformation of IMAGE to the projected catalog
            #   segment on the sensor plane (in unit of mm) 
            #
            #   x' = a + bx + cy
            #   y' = d + ex + fy
            #
            # the location of the image CENTER in the projected catalog segment  
            #   first_ra_mm  = a,  when x = 0 (the center of the IMAGE)
            #   first_dec_mm = d,  when y = 0
            #   first_roll_deg = np.arctan2(c, b)

            first_ra_mm, first_dec_mm, first_roll_deg = apply_match_trans(first_match_data)

            
            # calculate the image center RA & DEC after the image-to-catalog
            # matching and position transformation
            #
            # Inverse gnomonic projection
            #
            # . transform the first_ra_mm & first_dec_mm (the location of the
            #   image center in the catalog segment) back to ra & dec
            #               
            # the ra & dec of the center of the IMAGE is
            #   ra_image_center  = first_alpha 
            #   dec_image_center = first_delta
            #
            first_alpha, first_delta = plane2sky(first_ra_mm, first_dec_mm, first_ra_catalog, first_dec_catalog, lis_type)

            print()
            print(f'IMAGE center ra & dec  1:')
            print(f'    first_ra_mm =  {first_ra_mm}    first_dec_mm =  {first_dec_mm}')
            print(f'    first_alpha =  {first_alpha}    first_delta  =  {first_delta}')


            #--------------------------
            # select and read new NORMAL catalog segment according to the IMAGE
            # CENTER ra & dec
            #--------------------------

            # read the NEAREAST NORMAL catalog segment file according to the
            # calculated IMAGE center ra & dec, and contruct the NORMAL catalog
            # segment table
            #  
            noproj_table = search_catalog_objects(first_alpha, first_delta, lis_type)


            #--------------------------
            # create the projected catalog of the newly selected NORMAL catalog,
            # with the calculated IMAGE CENTER as the projection point
            #--------------------------

            # Build the PROJECTED catalog for the newly selected NORMAL catalog
            # segment with the IMAGE center (ra & dec) as the NEW projection point
            #
            # do the gnomonic projection for the NORMAL catalog segment selected 
            # with calculated (in the 1st iteration) image center
            #
            # Output (the projected catalog of the selected catalog segment):
            #   stt_data/new_cat
            #
            sky2plane(noproj_table, first_alpha, first_delta, lis_type)


            #--------------------------
            # call Match to match the image with the new projected catalog
            #--------------------------

            # invoke shell and execute the Match command to obtain the matching data
            #   second_match_data = a, b, c, d, e, f
            #   second_match_std  = sig, Nr, Nm, sx, sy
            #
            # sensor plane coordinates
            #   x' = a + bx + cy
            #   y' = d + ex + fy
            #
            second_match_data, second_match_std = call_match_once()


            #--------------------------
            # calculate the ra & dec of the IMAGE CENTER
            #--------------------------

            second_ra_mm, second_dec_mm, second_roll_deg = apply_match_trans(second_match_data)

            second_alpha, second_delta = plane2sky(second_ra_mm, second_dec_mm, first_alpha, first_delta, lis_type)

            print()
            print(f'IMAGE center ra & dec  2:')
            print(f'    second_ra_mm =  {second_ra_mm}    second_dec_mm =  {second_dec_mm}')
            print(f'    second_alpha =  {second_alpha}    second_delta  =  {second_delta}')


            #------------------------------------
            # 3rd. iterations
            #------------------------------------

            #--------------------------
            # create the projected catalog of the selected NORMAL catalog in the
            # 2nd iteration, with the newly calculated IMAGE CENTER as the
            # projection point
            #--------------------------
            sky2plane(noproj_table, second_alpha, second_delta, lis_type)


            #--------------------------
            # call Match to match the image with the new projected catalog
            #--------------------------
            third_match_data, third_match_std = call_match_once()


            #--------------------------
            # calculate the ra & dec of the IMAGE CENTER
            #--------------------------
            third_ra_mm, third_dec_mm, third_roll_deg = apply_match_trans(third_match_data)
            third_alpha, third_delta = plane2sky(third_ra_mm, third_dec_mm, second_alpha, second_delta, lis_type)

            print()
            print(f'IMAGE center ra & dec  3:')
            print(f'    third_ra_mm  =  {third_ra_mm}     third_dec_mm  =  {third_dec_mm}')
            print(f'    third_alpha  =  {third_alpha}     third_delta   =  {third_delta}')


            # third_alpha_normalized, third_roll_deg_normalized = normalize_coord(third_alpha, third_roll_deg)
            break

        except Exception as err:
            attempts += 1
            print('---> ERROR: {}'.format(err))

    if attempts == LIS_MAX_ITER:
        raise ValueError("---> ERROR: After {} attempts to find a match, it can not be done :(".format(LIS_MAX_ITER))
    
    # Print final results.
    tm2 = time.time()
    exec_time = tm2 - tm1
    print("\n---> ATTITUDE SOLUTION:\n RA   = {:.4f}°\n DEC  = {:.4f}°\n Roll = {:.4f}°\n Exec time = {:.4f} s".format(
        third_alpha, third_delta, third_roll_deg, exec_time))
    
    return third_alpha, third_delta, third_roll_deg, third_match_std[0], third_match_std[1]
