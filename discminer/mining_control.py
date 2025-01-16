"""
Useful repeating code
"""
import numpy as np
import argparse
import runpy
import sys
import os

path_mining = os.path.dirname(os.path.realpath(__file__)) + '/mining/'

def _check_and_return_parser(parserobj, prog='', description=''):
    if parserobj is None:
        parser = argparse.ArgumentParser(prog=prog, description=description)
    elif isinstance(parserobj, argparse._SubParsersAction):
        parser = parserobj.add_parser(prog, help=description)
    elif isinstance(parserobj, ArgumentParser):
        parser = parserobj
    else:
        print("Please provide a valid parser object from the argparse library...")
        sys.exit()
    return parser

#Customised action (from https://stackoverflow.com/questions/25295487/python-argparse-value-range-help-message-appearance)
class Range(argparse.Action):
    def __init__(self, minimum=None, maximum=None, *args, **kwargs):
        self.min = minimum
        self.max = maximum
        kwargs["metavar"] = "[%d, %d]" % (self.min, self.max)
        super(Range, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        value = np.asarray(value)
        ind = (value >= self.min) & (value <= self.max)
        if not ind.all():
            msg = 'invalid choice: %r (choose value from [%d, %d])' % (value, self.min, self.max)
            raise argparse.ArgumentError(self, msg)
        setattr(namespace, self.dest, value)
        
#**************************************
#DEFINE AND CUSTOMISE MINING ARGUMENTS
#-------------------->
def _mining_channels(parserobj, prog='channels', description='Make model channel maps and compare to data'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-mb', '--make_beam', default=-1, choices=[-1,0,1], type=int,
                        help="Convolve by beam? Defaults to -1 (convolve if downsampling size used for fit < beam size)")
    #Define and parse additional arguments. 'True' means 'enable' the argument, they can default to different values; use -h for help.
    add_parser_args(parser, planck=True) 
    return parser

def _mining_moments1d(parserobj, prog='moments1d', description='Make (gaussian, bell, or quadratic) moment maps and save output into .fits files'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-pk', '--peakkernel', default=1, type=int,
                        help="Return peak from fitted kernel (1) or global peak intensity (0). DEFAULTS to 1 (Return peak from kernel).")
    parser.add_argument('-fc', '--fit_continuum', default=0, type=int,
                        help="Fit continuum and save into .fits file? Defaults to 0")
    add_parser_args(parser, kernel=True, planck=True, sigma=4)
    return parser

def _mining_moments2d(parserobj, prog='moments2d', description='Make (double Gauss or double Bell) moment maps and save output into .fits files'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-ni', '--niter', default=10, type=int,
                        help="Number of iterations to re-do fit on hot pixels. DEFAULS to 10")
    parser.add_argument('-ne', '--neighs', default=5, type=int,
                        help="Number of neighbour pixels on each side of hot pixel considered for the iterative fit. DEFAULS to 5")
    add_parser_args(parser, kernel='doublebell', kind=True, planck=True, sigma=3)
    return parser

def _mining_moment_residuals(parserobj, prog='moment+residuals', description='Show Data vs Model moment map and residuals'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, smooth=True, mask_phi=True, mask_R=True, Router=True)    
    return parser

def _mining_moment_offset(parserobj, prog='moment+offset', description='Show moment map and a zoom-in illustrating offset from the centre'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-scontours', '--show_contours', default=1, type=int, help="Overlay moment map contours? DEFAULTS to 1.")
    parser.add_argument('-zoom', '--zoom_size', default=100, type=float, help="Physical size of the zoom-in region. DEFAULTS to 100 au.")
    add_parser_args(parser, moment=True, kernel=True, kind=True, show_continuum='none', surface=True, smooth=True,
                    radius_planet=True, phi_planet=True, label_planet=True, input_coords=True)
    return parser

def _mining_residuals_deproj(parserobj, prog='residuals+deproj', description='Show residuals from a moment map, deprojected onto the disc reference frame'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    add_parser_args(
        parser,
        moment=True, kernel=True, kind=True, surface=True, projection=True, Rinner=True, Router=0.95, absolute_Rinner=True, absolute_Router=True, smooth=True,
        show_continuum=True, show_filaments=True, spiral_ids=True, spiral_type=True, spiral_moment=True, colorbar=True,
        mask_R=True, mask_phi=True, radius_planet=True, phi_planet=True, label_planet=True, input_coords=True
    )
    return parser

def _mining_residuals_all(parserobj, prog='residuals+all', description='Show ALL moment map residuals, deprojected onto the sky OR disc reference frame'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-c', '--coords', default='sky', type=str, choices=['disc', 'disk', 'sky'], help="Reference frame in which the maps will be displayed. DEFAULTS to 'sky'.")
    add_parser_args(parser, kernel=True, kind=True, surface=True, Rinner=True, Router=True, smooth=True, radius_planet=True, phi_planet=True, label_planet=True, input_coords=True)    
    return parser

def _mining_radial_profiles(parserobj, prog='radprof', description='Extract and show radial profiles from moment maps AND residuals'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-sf', '--savgol_filter', default=0, type=int,
                        help="Apply a Savitzky-Golay filter to smooth out the plotted curves. DEFAULTS to False.")
    parser.add_argument('-vp', '--vphi_discminer', default=0, type=int,
                        help="Remove discminer vphi background? DEFAULTS to False (i.e. remove perfect Keplerian).")
    add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, writetxt=True, mask_minor=True, mask_major=True,
                    Rinner=True, absolute_Rinner=True, Router=True, absolute_Router=True, sigma=True, smooth=True)
    return parser

def _mining_radial_profiles_wedge(parserobj, prog='radprof+wedge', description='Extract and show radial profiles from moment residuals within specific wedges'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-wedges', '--wedges', nargs='*', default=[-60, 0, 0, 60, 120,  180, -180, -120], type=float, minimum=-180, maximum=180, action=Range,
                        help="Azimuthal boundaries of the wedges where radial profiles will be extracted. USAGE: -wedges 30 40 -60 -40 means two wedges, with boundaries (30, 40) and (-60, -40) deg. DEFAULTS to -60, 0, 0, 60, 120,  180, -180, -120.")
    parser.add_argument('-sf', '--savgol_filter', default=0, type=int,
                        help="Apply a Savitzky-Golay filter to smooth out the plotted curves. DEFAULTS to False.")
    add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, Rinner=True, absolute_Rinner=True, Router=True, absolute_Router=True, sigma=np.inf, smooth=True)
    return parser

def _mining_azimuthal_profiles(parserobj, prog='azimprof', description='Extract and show azimuthal profiles from moment maps OR residuals'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-r', '--radius', default=100, type=float,
                        help="Reference annulus shown as a solid black line. DEFAULTS to 100 au")
    parser.add_argument('-t', '--type', default='residuals', type=str, choices=['data', 'model', 'residuals'],
                        help="Compute profiles on data, model or residual moment map. DEFAULTS to 'residuals'")
    add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, Rinner=True, Router=True, smooth=True)
    return parser

def _mining_spectra(parserobj, prog='spectra', description='Extract and show line profiles along a specific annulus, every 30 deg'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-r', '--radius', default=200, type=float, help="Annulus where line profiles will be extracted. DEFAULTS to 200 au")
    parser.add_argument('-f', '--showfit', default=1, type=int, choices=[0, 1], help="Overlay best-fit line profile? DEFAULTS to 1")
    parser.add_argument('-ch', '--channel_id', default=-1, type=int, help="Id of intensity channel map to show. DEFAULTS to -1 (if -1, show moment map passed through -m flag)")
    parser.add_argument('-phi0', '--phi0', default=0, type=float, help="Starting azimuthal location for line profile extraction. DEFAULTS to 0 deg")
    parser.add_argument('-np', '--npix', default=0, type=int, help="Number of pixels around central pixel considered for line profile extraction. DEFAULS to 0")
    parser.add_argument('-t', '--type', default='data', type=str, choices=['data', 'model'], help="Show line profiles from data or model? DEFAULTS to 'data'")
    parser.add_argument('-scontours', '--show_contours', default=1, type=int, help="Overlay contours. DEFAULTS to 1.")    
    add_parser_args(parser, moment='peakintensity', kernel=True, kind=True, smooth=True, planck=True)
    return parser

def _mining_channels_peakint(parserobj, prog='channels+peakint', description='Show Data vs Model channel maps, peak intensities, and residuals'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-nc', '--nchans', default=5, type=int, help="Number of channels to plot")
    parser.add_argument('-st', '--step', default=4, type=int, help="Plot every #step channels")
    add_parser_args(parser, sigma=3, moment='peakintensity', kernel=True, surface=True, smooth=True, Rinner=True, Router=1.1)
    return parser

def _mining_isovelocities(parserobj, prog='isovelocities', description='Show Data vs Model isovelocity contours'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    add_parser_args(
        parser,
        moment=True, kernel=True, kind=True, surface=True, projection=True, Rinner=True, Router=0.95, absolute_Rinner=True, absolute_Router=True, smooth=True,
        show_continuum=True, show_filaments=True, spiral_ids=True, spiral_type=True, spiral_moment=True,
        mask_R=True, mask_phi=True, radius_planet=True, phi_planet=True, label_planet=True, input_coords=True
    )
    return parser

def _mining_pv_diagram(parserobj, prog='pv', description='Show PV diagram extracted along a specific axis'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-pvphi', '--pvphi', default=0.0, type=float, help="Azimuth of the axis along which the PV diagram is to be extracted. DEFAULTS to 0 deg")
    add_parser_args(parser, moment='peakintensity', kernel=True, kind=True, surface=True, smooth=True, Rinner=True, Router=1)
    return parser

def _mining_attributes(parserobj, prog='attributes', description='Show model attributes (z, v, I, Lw) as a function of radius'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    add_parser_args(parser)
    return parser

def _mining_gradient(parserobj, prog='gradient', description='Show peak, radial AND/OR azimuthal gradient from residual maps'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-gt', '--threshold', default=2, type=float, help="Minimum gradient value a peak must have to be taken into account. Defaults to 2 (m/s/au).")
    parser.add_argument('-sleg', '--show_legend', default=0, type=int, help="Show markers legend. DEFAULTS to 0.")
    add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, Rinner=True, absolute_Rinner=True, Router=True, absolute_Router=True,
                    gradient=True, smooth=True, radius_planet=True, phi_planet=True, label_planet=True, input_coords='disc')
    return parser

def _mining_parcube(parserobj, prog='parcube', description='Show cube reconstructed from fit parameters vs data cube'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    add_parser_args(parser, kernel=True, kind=True, surface='both')
    return parser

def _mining_parfile(parserobj, prog='parfile', description='Make JSON parameter file based on input log_pars.txt and prepare_data.py'):
    parser = _check_and_return_parser(parserobj, prog=prog, description=description)
    parser.add_argument('-f', '--log_file', default='', type=str, help="If not specified, try to guess input log_pars.txt with existing files. DEFAULTS to ''.")
    parser.add_argument('-p', '--prepare_data', default='prepare_data.py', type=str, help="Script employed to clip and downsample the cube of interest. DEFAULTS to prepare_data.py.")
    parser.add_argument('-j', '--json_file', default='parfile.json', type=str, help="Name of output JSON file. DEFAULTS to parfile.json.")
    parser.add_argument('-o', '--overwrite', default=0, type=int, help="overwrite if parfile.json exists. DEFAULTS to 0.")
    parser.add_argument('-r', '--reset', default=0, type=int,
                        help="If (1) AND --overwrite, rewrite parfile.json. If (0) AND --overwrite, forward 'custom' dictionary, and rewrite metadata and model parameters only. DEFAULTS to 0.")
    parser.add_argument('-d', '--download_cube', default=0, type=int, help="Download reduced ready-to-use cube from the NRAO server (valid for exoALMA data). DEFAULTS to 0.")
    add_parser_args(parser)


_mining_parser_func = {
    'parfile': _mining_parfile,    
    'channels': _mining_channels,
    'moments1d': _mining_moments1d,
    'moments2d': _mining_moments2d,
    'parcube': _mining_parcube,
    'channels+peakint': _mining_channels_peakint,    
    'attributes': _mining_attributes,
    'radprof': _mining_radial_profiles,
    'radprof+wedge': _mining_radial_profiles_wedge,    
    'azimprof': _mining_azimuthal_profiles,    
    'spectra': _mining_spectra,    
    'moment+residuals': _mining_moment_residuals,
    'moment+offset': _mining_moment_offset,
    'residuals+deproj': _mining_residuals_deproj,
    'residuals+all': _mining_residuals_all,    
    'gradient': _mining_gradient,
    'isovelocities': _mining_isovelocities,
    'pv': _mining_pv_diagram,
}

scripts = {
    'parfile': 'make_parfile.py',
    'channels': 'make_channels.py',
    'moments1d': 'make_single_moments.py',
    'moments2d': 'make_double_moments.py',
    'parcube': 'show_parcube.py',        
    'channels+peakint': 'plot_channels+peakint.py',    
    'attributes': 'plot_attributes_model.py',
    'radprof': 'plot_radial_profiles.py',
    'radprof+wedge': 'plot_radial_profiles+wedge.py',    
    'azimprof': 'plot_azimuthal_profiles.py',    
    'spectra': 'plot_spectra.py',    
    'moment+residuals': 'plot_moment+residuals.py',
    'moment+offset': 'plot_moment+offset.py',
    'residuals+deproj': 'plot_residuals+deproj.py',
    'residuals+all': 'plot_residuals+all.py',
    'gradient': 'plot_gradient.py',
    'isovelocities': 'plot_isovelocities.py',
    'pv': 'plot_pv_diagram.py',
}

#<----------------------

#*********************************
#PARSE ARGUMENTS AND CALL SCRIPTS
#*********************************
def main():
    parser = argparse.ArgumentParser(description="****>Run mining analysis scripts<****")
    subparsers = parser.add_subparsers(dest='make', help="Select a subscript to run.")

    for key in scripts:
        _mining_parser_func[key](subparsers) #Append subparsers

    args = parser.parse_args()
    _adjust_args(args)
    options = argparse.Namespace(**vars(args)) #Organised set of arguments passed as globals to the script
    
    if args.make in scripts:
        script_path = os.path.join(path_mining, scripts[args.make])
        script_args = {k: v for k, v in vars(args).items() if v is not None and k not in ('make',)}
        
        print(f"Running {script_path} with arguments {script_args}")

        #Run the selected script, passing the arguments as globals
        runpy.run_path(script_path, init_globals={'args': options} )
        
    else:
        print("Please specify a valid subcommand. Use -h for help.")

        
def add_parser_args(parser,
                    show_output=True, show_block=True,
                    moment=False, kernel=False, surface=False, smooth=False,
                    Rinner=False, Router=False, absolute_Rinner=False, absolute_Router=False,
                    kind=False, projection=False,
                    planck=False,
                    sigma=False,
                    mask_minor=False, mask_major=False, 
                    mask_phi=False, mask_R=False,
                    fold=False,
                    writetxt=False, writefits=False,
                    gradient=False,
                    colorbar=False,
                    show_continuum=False, show_filaments=False,
                    spiral_ids=False, spiral_type=False, spiral_moment=False,
                    filament_ids=False, filament_moment=False,
                    radius_planet=False, phi_planet=False, label_planet=False, input_coords=False,
):

    def set_default(val, default):
        if isinstance(val, bool):
            return default
        else:
            return val
        
    if moment:
        d0 = set_default(moment, 'velocity')
        parser.add_argument('-m', '--moment', default=d0, type=str,
                            choices=['velocity', 'linewidth', 'lineslope', 'peakint', 'peakintensity', 'v0r', 'v0phi', 'v0z', 'vr_leftover', 'delta_velocity', 'delta_linewidth', 'delta_peakintensity', 'reducedchi2'],
                            help="Type of moment map to be analysed. DEFAULTS to '%s'"%d0)

    if kernel:
        d0 = set_default(kernel, 'gaussian')
        parser.add_argument('-k', '--kernel', default=d0, type=str,
                            choices=['gauss', 'gaussian', 'bell', 'quadratic', 'dgauss', 'doublegaussian', 'dbell', 'doublebell'],
                            help="Kernel used for line profile fit and calculation of moment maps. DEFAULTS to '%s'"%d0)
        
    if kind:
        d0 = set_default(kind, 'mask')
        parser.add_argument('-ki', '--kind', default=d0, type=str, choices=['mask', 'sum'],
                            help="Method for merging upper and lower surface kernel profiles. DEFAULTS to '%s'"%d0)
        
    if surface:
        d0 = set_default(surface, 'upper')
        parser.add_argument('-s', '--surface', default=d0, type=str,
                            choices=['up', 'upper', 'low', 'lower', 'both'],
                            help="Use upper or lower surface moment map. DEFAULTS to '%s'"%d0)
    
    if show_continuum:
        d0 = set_default(show_continuum, 'none')
        parser.add_argument('-sc', '--show_continuum', default=d0, type=str,
                            choices=['none', 'all', 'band7', 'scattered'],
                            help="Which continuum map to show. DEFAULTS to '%s'"%d0)

    if show_filaments:
        d0 = set_default(show_filaments, 0)
        parser.add_argument('-sf', '--show_filaments', default=d0, type=int,
                            choices=[0, 1],
                            help="Make and show filaments? DEFAULTS to '%s'"%d0)        
    if show_output:
        d0 = set_default(show_output, 1)
        parser.add_argument('-so', '--show_output', default=d0, type=int,
                            choices=[0, 1],
                            help="Show output plots? DEFAULTS to '%s'"%d0)        

    if show_block:
        d0 = set_default(show_block, 1)
        parser.add_argument('-sb', '--show_block', default=d0, type=int,
                            choices=[0, 1],
                            help="Block ipython session when a figure is displayed? DEFAULTS to '%s'"%d0)        
        
    if mask_minor:
        d0 = set_default(mask_minor, 30.0)
        parser.add_argument('-b', '--mask_minor', default=d0, type=float,
                            help="+- azimuthal mask around disc minor axis for calculation of vphi and vz velocity components. DEFAULTS to %.1f deg"%d0)

    if mask_major:
        d0 = set_default(mask_major, 30.0)
        parser.add_argument('-a', '--mask_major', default=d0, type=float,
                            help="+- azimuthal mask around disc major axis for calculation of vR velocity component. DEFAULTS to %.1f deg"%d0)

    if mask_phi:
        d0 = set_default(mask_phi, [])
        parser.add_argument('-mask_phi', '--mask_phi', nargs='*', default=d0, type=float, help="Azimuthal boundaries of mask(s) to apply to data and model products. FORMAT: -mask_phi 30 40 -60 -40 means two masks, with boundaries (30, 40) and (-60, -40) deg. DEFAULTS to [].")

    if mask_R:
        d0 = set_default(mask_R, [])
        parser.add_argument('-mask_R', '--mask_R', nargs='*', default=d0, type=float, help="Radial boundaries of mask(s) to apply to data and model products. FORMAT: -mask_R 50 75 30 60 means two masks, with boundaries (50, 75) and (30, 60) au . DEFAULTS to [].")                

    if fold:
        d0 = set_default(fold, 'absolute')        
        parser.add_argument('-f', '--fold', default=d0, type=str,
                            choices=['absolute', 'standard'],
                            help="if moment=velocity, fold absolute or standard velocity residuals. DEFAULTS to '%s'"%d0)

    if projection:
        d0 = set_default(projection, 'cartesian')                
        parser.add_argument('-p', '--projection', default=d0, type=str,
                            choices=['cartesian', 'polar'],
                            help="Project residuals onto a cartesian or a polar map. DEFAULTS to '%s'"%d0)

    if planck:
        d0 = set_default(planck, 0)                
        parser.add_argument('-planck', '--planck', default=d0, type=int,
                            choices=[0, 1],
                            help="Use full Planck's law to convert Jy/bm to K. Defaults to 0 (i.e. assume RJ)")
        
    if writetxt:
        d0 = set_default(writetxt, 1)                
        parser.add_argument('-w', '--writetxt', default=d0, type=int,
                            choices=[0, 1],
                            help="Write output into txt file(s)? DEFAULTS to %d"%d0)

    if writefits:
        d0 = set_default(writefits, 1)                
        parser.add_argument('-wf', '--writefits', default=d0, type=int,
                            choices=[0, 1],
                            help="Write output into fits file(s)? DEFAULTS to %d"%d0)
        
    if Rinner:
        d0 = set_default(Rinner, 1.0)                        
        parser.add_argument('-i', '--Rinner', default=d0, type=float,
                            help="Number of beams to mask out from the disc inner region. DEFAULTS to %.2f"%d0)

    if absolute_Rinner:
        d0 = set_default(absolute_Rinner, -1)
        parser.add_argument('-ai', '--absolute_Rinner', default=d0, type=float,
                            help="If >= 0, assume this absolute value instead of Rinner to set the inner radius of the analysis domain. Defaults to -1.")
        
    if Router:
        d0 = set_default(Router, 0.98)                
        parser.add_argument('-o', '--Router', default=d0, type=float,
                            help="Fraction of Rout to consider as the disc outer radius for the analysis. DEFAULTS to %.2f"%d0)

    if absolute_Router:
        d0 = set_default(absolute_Router, -1)
        parser.add_argument('-ao', '--absolute_Router', default=d0, type=float,
                            help="If >= 0, assume this absolute value instead of Router to set the outer radius of the analysis domain. Defaults to -1.")
                
    if sigma:
        d0 = set_default(sigma, 5)                
        parser.add_argument('-si', '--sigma', default=d0, type=float,
                            help="Mask out pixels with values below sigma threshold. DEFAULTS to %.1f"%d0)

    if gradient:
        d0 = set_default(gradient, 'r')
        parser.add_argument('-g', '--gradient', default=d0, type=str, choices=['peak', 'r', 'phi'],
                            help="Coordinate along which the gradient will be computed. If 'peak', the maximum gradient is computed. DEFAULTS to '%s'"%d0)

    if colorbar:
        d0 = set_default(colorbar, 1)
        parser.add_argument('-cbar', '--colorbar', default=d0, type=int, help="Show colorbar. DEFAULTS to %d"%d0)        
        
    if smooth:
        d0 = set_default(smooth, 0.0)
        parser.add_argument('-sm', '--smooth', default=d0, type=float,
                            help="Smooth up moment_data using scipy.ndimage gaussian_filter? DEFAULTS to %.1f (characteristic size of smoothing kernel in pixels)"%d0)        

    if spiral_ids:
        d0 = set_default(spiral_ids, [])
        parser.add_argument('-spids', '--spiral_ids', nargs='*', default=d0, type=int, help="Which filament should be fitted with a spiral function. Spirals are overlaid on the map and their best-fit parameters are saved into a txt file. FORMAT: -spids 1 3 -2. DEFAULTS to [].")

    if spiral_type:
        d0 = set_default(spiral_type, 'linear')        
        parser.add_argument('-sptype', '--spiral_type', default=d0, choices=['linear', 'log'], help="Type of spiral functional form to fit filaments. DEFAULTS to %s."%d0)

    if spiral_moment:
        d0 = set_default(spiral_moment, 'velocity')        
        parser.add_argument('-spmom', '--spiral_moment', default=d0, choices=['velocity', 'linewidth', 'peakint', 'peakintensity'], help="Moment map from which spirals were extracted. DEFAULTS to %s."%d0)        

    if filament_ids:
        d0 = set_default(filament_ids, [])
        parser.add_argument('-filids', '--filament_ids', nargs='*', default=d0, type=int, help="Filament(s) to be shown. FORMAT: -filids 1 3 -2. DEFAULTS to [].")

    if filament_moment:
        d0 = set_default(filament_moment, 'velocity')        
        parser.add_argument('-filmom', '--filament_moment', default=d0, choices=['velocity', 'linewidth', 'peakint', 'peakintensity'], help="Moment map from which filament(s) were extracted. DEFAULTS to %s."%d0)        

    if radius_planet:
        d0 = set_default(radius_planet, [])                
        parser.add_argument('-rp', '--rp', nargs='*', default=d0, type=float, help="Mark radial location of planet(s) in au. DEFAULTS to [].")

    if phi_planet:
        d0 = set_default(phi_planet, [])                
        parser.add_argument('-phip', '--phip', nargs='*', default=d0, type=float, help="Mark azimuthal location of planet(s) in degrees. DEFAULTS to [].")
        
    if label_planet:
        d0 = set_default(label_planet, [])                
        parser.add_argument('-labelp', '--labelp', nargs='*', default=d0, type=str, help="Add label(s) centred on the planet(s) marker. DEFAULTS to [].")

    if input_coords:
        d0 = set_default(input_coords, 'disc')
        parser.add_argument('-input_coords', '--input_coords', default=d0, type=str,
                            choices=['sky', 'disc', 'disk'],
                            help="Reference frame of the input planet coordinates. If 'sky', -rp must be the projected distance of the planet in arcsecs and -phip the Position Angle of the planet measured from the North. If 'disc' or 'disk', -rp must be the orbital radius of the planet in au and -phip the azimuthal location measured from the disc x axis. DEFAULTS to '%s'"%d0)

        
def _adjust_args(args):
    try:
        if args.moment=='peakint':
            args.moment = 'peakintensity'
    except AttributeError:
        pass

    try:
        if args.spiral_moment=='peakint':
            args.spiral_moment = 'peakintensity'
    except AttributeError:
        pass
    
    try:
        if args.surface=='up':
            args.surface = 'upper'
    except AttributeError:
        args.surface = 'upper'

    try:
        if args.surface=='low':
            args.surface = 'lower'
    except AttributeError:
        args.surface = 'upper'

    try:
        if args.kernel=='gauss':
            args.kernel = 'gaussian'
    except AttributeError:
        args.kernel = 'gaussian'
    
    try:
        if args.kernel=='dgauss':
            args.kernel = 'doublegaussian'
    except AttributeError:
        args.kernel = 'gaussian'

    try:
        if args.kernel=='dbell':
            args.kernel = 'doublebell'
    except AttributeError:
        args.kernel = 'gaussian'
    

if __name__ == '__main__':
    main()

    
