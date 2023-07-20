import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from reproject import reproject_interp
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import aplpy
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from astropy.table import Table
import matplotlib.gridspec as gridspec
from deproject import deproject
import seaborn as sns
from astropy.coordinates import SkyCoord
import Metallicity 
import alphaCO 

v_M87 = 1284.011278 * u.km/u.s # helio_velocity (from NED) from M87
dv_M87 = 5.096472*u.km/u.s
d_sun_M87 = 16.5 *u.Mpc #(VERTICO Team assumes that)
r_virgo = 1550*u.kpc #McLaughlin 1999; Ferrarese et al. 2012
v_disp_virgo = 593 *u.km/u.s #(Mei et al. 2007)
color_lst = color_list = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39',
    '#5254a3', '#6b6ecf', '#637939', '#8c6d31', '#843c39', '#5254a3', '#6b6ecf', '#8ca252',
    '#bd9e39', '#ad494a', '#756bb1', '#9c9ede', '#756bb1', '#9c9ede', '#a55194', '#ce6dbd',
    '#de9ed6', '#3182bd', '#e6550d', '#636363', '#969696', '#636363', '#969696', '#9e9ac8',
    '#cbc9e2', '#756bb1', '#9c9ede', '#756bb1', '#9c9ede'
]

#['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','lightblue','yellow']


loc = '/home/greisc1/'
incl_cut = 75
position_file =loc+ 'Project/Data/VERTICO_coord.fits'  #('Galaxy', 'S7'), ('RA', '>f4'), ('DEC', '>f4'), ('v_opt', '>f8'), ('inclination', '>f8'), ('pa', '>f8')
VERTICO_table = Table.read(position_file, format='fits')
galaxies = VERTICO_table['Galaxy']
PHANGS_table = Table.read(loc + 'Project/Data/phangs_sample_table_v1p6.fits', format ='fits')
PHANGS_galaxies = PHANGS_table['name']
Zabel_table = pd.read_csv(loc+ 'Project/Data/zabel22_table1.csv', sep=',')



def FOV(hdr):
    '''Takes the header of a fits file and calculates the radius of the image
    returns width, height and r in deg'''
    
    NAXIS1 = hdr['NAXIS1']
    NAXIS2 = hdr['NAXIS2']
    CDELT1 = hdr['CDELT1']
    CDELT2 = hdr['CDELT2']
    width = NAXIS1*CDELT1
    height = NAXIS2*CDELT2
    if width > height:
        r = width/2
        
    else:
        r = height/2

    return width, height, r 


def basic_plot(axis_number, scalebar_length,ra,dec,hdr):
    '''Function adds a colorbar, removes axis and tick_labels and ticks, adds a scalebar, and recenters the image
    
    Add this line after f.show_colorscale() and before f.colorbar.show()
    axis number = f1 or f2..., scalebar_length = flot or integer (number in kpc), ra,dec in deg
   '''
    
    axis_number.add_colorbar()
    width, height,r = FOV(hdr)
    axis_number.axis_labels.hide()
    axis_number.tick_labels.hide()
    axis_number.ticks.hide()
    axis_number.add_scalebar(length=np.rad2deg((u.rad*scalebar_length*u.kpc/(16.5*u.Mpc)).to('deg')))
    axis_number.scalebar.set_color('k')
    #axis_number.scalebar.set_corner('bottom left')
    axis_number.scalebar.set_label(f"{scalebar_length} kpc")
    axis_number.scalebar.set_font_size("small")
    #axis_number.recenter(ra, dec, radius= r)

def PHANGS_data(galaxy):
    '''Returns ra, dec, incl, posang and r_eff, r_25 in deg'''
    
    table = Table.read(loc + 'Project/Data/phangs_sample_table_v1p6.fits', format ='fits')
    idx = np.where(table['name']==f'ngc{galaxy[-4:]}')
    ra = np.array(table['orient_ra'])[idx] 
    dec = np.array(table['orient_dec'])[idx] 
    incl = np.array(table['orient_incl'])[idx]   
    posang = np.array(table['orient_posang'])[idx] 
    r_eff = np.array(table['size_reff'])[idx] 
    r_25 = np.array(table['size_r25'])[idx] 


    return ra,dec,incl,posang,r_eff,r_25

def VERTICO_data(galaxy):
    '''v_M87 (in km/s) and d_sun_M87 (in Mpc) have to be defined before in the file
    Returns ra, dec, incl, posang  in deg, v_gal (radial) and projected ditance d_proj in kpc'''
    
    props = pd.read_csv(loc + 'Project/Data/VERTICO_prop_Nov22.txt', sep=',')
    idx_prop = np.where(props['Galaxy']==f'NGC{galaxy[-4:]}')
    v_gal = (np.array(props['v_opt'])[idx_prop]-v_M87.value)*u.km/u.s  #relative velocity galaxy to M87
    dM87 = np.array(props['dM87'])[idx_prop]   #projected distance in degree
    d_proj = (np.deg2rad(dM87)*d_sun_M87).to('kpc')    #projected distance to M87 in Mpc

  
    table = Table.read(loc+ 'Project/Data/VERTICO_coord.fits', format='fits')
    idx = np.where(table['Galaxy']==f'NGC{galaxy[-4:]}')
    ra = np.array(table['RA'])[idx] 
    dec = np.array(table['DEC'])[idx] 
    incl = np.array(table['inclination'])[idx]   
    posang = np.array(table['pa'])[idx] 
    
    return ra,dec,incl,posang,v_gal,d_proj


def Zabel_table1(galaxy):
    '''Returns a dictonary containing table 1 of Zabel et al 2022 
    contains: HI_class (0,I,II..),HI_def_M,HI_def_R, H2_def  and stellar radius R* '''
    table = pd.read_csv(loc+ 'Project/Data/zabel22_table1.csv', sep=',')

    for position, Zab_galaxy in enumerate(table['Galaxy']):
        if Zab_galaxy == f'NGC{galaxy[-4:]}' or Zab_galaxy == f'NGC{galaxy[-4:]} (a)':
            
            Zabel_dict = {
                'Galaxy': table['Galaxy'][position],
                'HI_class': table['cl'][position],
                'HI_def_M': table['def_HI;M* (dex)'][position],
                'HI_def_R': table['def_HI;R* (dex)'][position],
                'H2_def': table['def_H2 (dex)'][position],
                'R_star': table['R* (kpc)'][position],
                'logM': table['log M* (Msol)'][position]
            }

    return Zabel_dict


def VERTICOIV_table1(galaxy):
    '''Returns a dictonary containing table 1 of Villanueva et al 2023 (VERTICOIV) 
    contains: M_mol, M_star, lmol, l*(exponential scale length), Re,mol, Re,* (half-light radius, effective radius)  '''
    table = pd.read_csv(loc+ 'Project/Data/VERTICOIV.txt', sep=',')

    
    for i, VERTICOIV_galaxy in enumerate(table['Name']):
        if VERTICOIV_galaxy == f'NGC{galaxy[-4:]}' or VERTICOIV_galaxy == f'NGC{galaxy[-4:]} (a)':
    
            VERTICOIV_dict = {
                'Galaxy': table['Name'][i],
                'HI_class': table['HI-Class'][i],
                'logMmol': float(table['logMmol'][i][3:7]),
                'logM*': float(table['logM*'][i][3:8]),
                'l_mol': float(table['l_mol(kpc)'][i][3:7]),
                'l_*': float(table['l_*(kpc)'][i][3:7]),
                'R_e_mol': float(table['R_e_mol(kpc)'][i][3:7]),
                'R_e_*': float(table['R_e_*(kpc)'][i][3:7])
            }

    return VERTICOIV_dict

    
def convert_co_mass(L_co): 
    '''convert mol data from CO linewidth to gas mass
    returns M_mol in units of M_sol/pc^2'''
    
    alpha_co =4.35 * u.Msun / u.pc **2
    R_21=0.8
    M_mol = alpha_co/R_21*L_co
    
    return M_mol

def Convert_CO_Mass_Radial(galaxy, L_co, ra,dec, incl, posang, hdr_mol):
    '''convert mol data from CO linewidth to gas mass
    returns M_mol in units of M_sol/pc^2 using metallicity dependent convertion factor
    Predict 12+log(O/H) with the 'SAMI19' MZR (Sanchez+19), 
    alphaCO10 with a simple power-law metallicity dependence (Sun+ 2020)'''

    R_array,_ = deproject((ra*u.deg,dec*u.deg),incl*u.deg, posang*u.deg, header=hdr_mol)
    R_array= np.deg2rad(np.array(R_array))*d_sun_M87.to('kpc')
    VERTICOIV_dict = VERTICOIV_table1(galaxy)
    M_star = 10**(VERTICOIV_dict['logM*'])
    R_e = VERTICOIV_dict['R_e_*']*u.kpc
    logOH_center = Metallicity.predict_logOH_SAMI19(M_star)
    logOH_galaxy = Metallicity.extrapolate_logOH_radially(logOH_center, gradient='CALIFA14', Rgal=R_array, Re=R_e)
    alphaCO_galaxy = alphaCO.predict_alphaCO10_S20(logOH_galaxy)

    R_21=0.8
    M_mol = alphaCO_galaxy/R_21*L_co
    print(f'Mstar {M_star} R_e {R_e}, logOHcenter {logOH_center}, logOH_galaxy, {logOH_galaxy} alphaCO {alphaCO_galaxy}, L_co{L_co}')

    
    return M_mol

def Mol_Gas_file(galaxy,resolution):
    '''returns mol gas file for specific resolution and galaxy. 
    Here is all the mol gas data one uses in the project'''
    
    if resolution == 'VERTICO_9as':
        Mol_gas_file = f"{loc}VERTICO/Mol_gas/{galaxy}/{galaxy}_7m+tp_co21_pbcorr_9as_round_mom0_Kkms-1.fits"
    elif resolution == 'VERTICO_9as_e':
        Mol_gas_file = f"{loc}VERTICO/Mol_gas/{galaxy}/{galaxy}_7m+tp_co21_pbcorr_9as_round_mom0_unc.fits"
    elif resolution == 'PHANGS_high':
        Mol_gas_file = f"{loc}PHANGS/ngc{galaxy[-4:]}/ngc{galaxy[-4:]}_12m+7m+tp_co21_broad_mom0.fits"
    elif resolution == 'PHANGS_high_e':
        Mol_gas_file = f"{loc}PHANGS/ngc{galaxy[-4:]}/ngc{galaxy[-4:]}_12m+7m+tp_co21_broad_emom0.fits"
    elif resolution == 'PHANGS_high_strict':
        Mol_gas_file = f"{loc}PHANGS/ngc{galaxy[-4:]}/ngc{galaxy[-4:]}_12m+7m+tp_co21_strict_mom0.fits"
    elif resolution == 'PHANGS_high_strict_e':
        Mol_gas_file = f"{loc}PHANGS/ngc{galaxy[-4:]}/ngc{galaxy[-4:]}_12m+7m+tp_co21_strict_emom0.fits"
    elif resolution == 'PHANGS_7.5as':
        Mol_gas_file = f"{loc}PHANGS/ngc{galaxy[-4:]}/ngc{galaxy[-4:]}_12m+7m+tp_co21_7p5as_broad_mom0.fits"
    elif resolution == 'PHANGS_7.5as_e':
        Mol_gas_file = f"{loc}PHANGS/ngc{galaxy[-4:]}/ngc{galaxy[-4:]}_12m+7m+tp_co21_7p5as_broad_emom0.fits"
    
    else:
        Mol_gas_file = f"{loc}VERTICO/Mol_gas/{galaxy}/not_here_file.fits"

    
    return Mol_gas_file


def data_maps(galaxy, resolution, ra, dec, incl, posang):
    '''NO UNITS
    for a specific galaxy with molecular gas of spedific resolution the funcion 
    returns the molecular gas and stellar mass surface density and hdr_mol 
    corrected for the inclination of the galaxy (surface density is smaller than observed one cos(incl))
    !stellar maps: <= 0 to nans -> !'''
    
    mol_map_path = Mol_Gas_file(galaxy, resolution)
        
    # Mol mass maps
    hdul_mol = fits.open(mol_map_path)
    hdr_mol = hdul_mol[0].header
    data_mol = hdul_mol[0].data
    sigma_mol = convert_co_mass(data_mol)  #constant converstion CO line to H2
    #sigma_mol = convert_CO_Mass_Radial(galaxy, data_mol, ra,dec, incl, posang, hdr_mol)
    # Stellar mass maps
    stellar_maps = f"{loc}sfr_mstar_maps/15as_new/{galaxy}_mstar_w1+w3_15as.fits"
    hdul_s = fits.open(stellar_maps)
    hdr_s = hdul_s[0].header
    data_s = hdul_s[0].data
    
    # Replace 0 with NaN
    data_s = np.where(data_s <= 0,np.nan,data_s)
    
    # make same grid as Vertico mol data
    sigma_s, _ = reproject_interp((data_s,hdr_s), hdr_mol)


    #inlination effect
    sigma_s = sigma_s*np.cos(np.deg2rad(incl))
    sigma_mol =sigma_mol*np.cos(np.deg2rad(incl))
    tot_mol = np.nansum(np.nansum(sigma_mol))

    
    return sigma_s, sigma_mol.value, hdr_mol, tot_mol.value


def e_data_maps(galaxy, resolution, ra, dec, incl, posang):
    '''returns e_mol maps for specific galaxy and resolution. Corrected for inclination?'''
    
    mol_map_path = Mol_Gas_file(galaxy, f'{resolution}_e')  
    # Mol mass maps
    hdul_mol = fits.open(mol_map_path)
    hdr_mol = hdul_mol[0].header
    e_data_mol = hdul_mol[0].data

    #e_sigma_mol = convert_co_mass(e_data_mol)
    #e_sigma_mol = convert_CO_Mass_Radial(galaxy, e_data_mol, ra,dec, incl, posang, hdr_mol)

    e_sigma_mol =e_sigma_mol*np.cos(np.deg2rad(incl))
    
    return e_sigma_mol.value


def sigma_mol_limits(sigma_mol,e_sigma_mol,Type):
    if Type == 'strict':
      sigma_mol = np.where((sigma_mol < 3*e_sigma_mol ),0, sigma_mol)
    elif Type == 'Upper limit':
      sigma_mol = np.where((sigma_mol < 3*e_sigma_mol ),e_sigma_mol , sigma_mol)
    elif Type == '3*Upper limit':
      sigma_mol = np.where((sigma_mol < 3*e_sigma_mol ),3*e_sigma_mol , sigma_mol)
    else:
        sigma_mol = 'error'
    return sigma_mol


def Susceptible_gas(div, sigma_mol, e_sigma_mol, P_s, P_RP,Type):
    sigma_max =np.divide(P_RP, np.divide(P_s,sigma_mol))
    if Type == 'Trust broad':
        susc_mol = np.where(div <= 1, sigma_mol,0)
    if Type == 'Lower limit':
        susc_mol = np.where(np.logical_or(div > 1, np.logical_and(sigma_mol < 3 * e_sigma_mol,sigma_mol+ e_sigma_mol > sigma_max)), 0, sigma_mol)
    if Type == 'Upper limit':
        susc_mol = np.where(np.logical_or(div <= 1, np.logical_and(sigma_mol < 3 * e_sigma_mol,sigma_mol- e_sigma_mol < sigma_max)), sigma_mol,0)
    if Type == 'Lower limit2':
        susc_mol = np.where(np.logical_or(div > 1, sigma_mol < 3* e_sigma_mol), 0, sigma_mol)
    if Type == 'Upper limit2':
        susc_mol = np.where(np.logical_or(div <= 1, sigma_mol < 3* e_sigma_mol), sigma_mol,0)
    if Type == 'Lower limit3':
        susc_mol = np.where(np.logical_or(div > 1, sigma_mol+ e_sigma_mol > sigma_max), 0, sigma_mol)
    if Type == 'Upper limit3':
        susc_mol = np.where(np.logical_or(div <= 1, sigma_mol- e_sigma_mol < sigma_max), sigma_mol,0)


    return susc_mol




'''This Part of the Code gives galaxies properties and calculates their RP and Restoring Pressure'''

def Pressure(sigma_s, sigma_mol, tot_mol, hdr, v_ICM,d_center,phi,e_sigma_mol, Type):
    '''Normal case we would v_ICM = sqrt(3) * v_rad * np.cos(np.deg2rad(phi))
    and d_center = d_proj *np.pi/2
    
    Returns Restoring and Ram pressure in N/m^2 and rho_ICM in kg/m^3'''
     
    rho1=3.38*10**(-25)*u.g/u.cm**3
    rho2=1.175*10**(-26)*u.g/u.cm**3
    r1 = 1.7*u.kpc
    r2=21.4*u.kpc
    beta1 =0.42
    beta2=0.47
    
    rho_ICM= (rho1*(1+(d_center/r1)**2)**(-3*beta1/2)+rho2*(1+(d_center/r2)**2)**(-3*beta2/2)).to(u.kg/u.m**3)
    P_RP =(rho_ICM*(v_ICM*np.cos(np.deg2rad(phi)))**2).to(u.N/u.m**2) 

    '''Restoring Pressure'''    
    g_s = (2*const.G*sigma_s*u.solMass/u.pc**2*np.pi).to(u.m/u.s**2)  #restoring force stellar mass
    P_s = (g_s*sigma_mol*u.solMass/u.pc**2).to(u.N/u.m**2)
    
    '''Ratio of P_s and RP'''
    RP_arr = np.full_like(P_s, P_RP)   #array full of the RP value in size of restoring pressure
    div = np.divide(P_s, RP_arr)
    tot_mol = np.nansum(np.nansum(sigma_mol))
    
    susc_mol_gas = Susceptible_gas(div, sigma_mol, e_sigma_mol, P_s, P_RP,Type)
    tot_susc_mol = np.nansum(np.nansum(susc_mol_gas))
    strip_perc = 100 * tot_susc_mol/tot_mol

    return P_s, P_RP, rho_ICM, div, strip_perc


def Edge_On_Pressure(galaxy,resolution, sigma_s, sigma_mol, tot_mol, hdr_mol, v_ICM, d_center, e_sigma_mol, Type):
    ''''''
    if resolution == 'PHANGS_high':
        ra,dec,incl,posang,r_eff,r_25 = PHANGS_data(galaxy)    
    else:
        ra,dec,incl,posang,v_gal,d_proj = VERTICO_data(galaxy)
        
    #load rotation velocity, interfolate and calculate v_array (v for every pixel through deprojected radius)
    table = Table.read(f'/home/greisc1/Project/Data/rotation_curves/rotation_curves/{galaxy}_model_legendre.ecsv', format ='ascii.ecsv')
    v_rot = table['V_circ']
    r_rot = table['r_gal']

    v_rot[0] = 0
    r_rot= (r_rot.to('deg'))

    r_array,_ = deproject((ra*u.deg,dec*u.deg),incl*u.deg, posang*u.deg, header=hdr_mol)
    v_array = np.interp(r_array, r_rot.value,v_rot)

    v_array = (v_array*u.km/u.s).to(u.parsec/u.s)
    r_array= np.deg2rad(np.array(r_array))*d_sun_M87.to('pc')
    P_s = np.divide(np.multiply(sigma_mol*u.solMass/u.pc**2,v_array**2),r_array).to(u.N/u.m**2)    

    _, P_RP, rho_ICM, _, _ = Pressure(sigma_s, sigma_mol,tot_mol, hdr_mol, v_ICM, d_center,0, e_sigma_mol, Type)
    P_RP = np.full_like(P_s, P_RP)   #array full of the RP value in size of restoring pressure

    div = np.divide(P_s, P_RP)
    #tot_mol = np.nansum(np.nansum(sigma_mol)) 
    susc_mol = Susceptible_gas(div, sigma_mol, e_sigma_mol, P_s, P_RP,Type)
    tot_susc_mol = np.nansum(np.nansum(susc_mol))
    strip_perc = 100 * tot_susc_mol/tot_mol

    return P_s, P_RP, rho_ICM, div, strip_perc
    


def Monte_Carlo_sys_error(N_MC, sigma_s, sigma_mol, tot_mol, hdr, e_sigma_mol, Type):
    '''N_MC gives the amount of iterations, drawn from velocity, inclination and distance distribution'''
    
    #velocity, distance and wind-impact angle phi distribution for galaxy
    v_distr =
    d_dist =
    phi_dist =
    i = 0

    while i <= N_MC:
        v = #draw randomlly from np.random. v_distr
        d = #draw randomlly from d_dist
        phi = #draw randomly from phi_dist
        P_s, P_RP, rho_ICM, div, strip_perc = Pressure(sigma_s, sigma_mol, tot_mol, hdr, v, d, phi, e_sigma_mol, Type)
        i += 1

