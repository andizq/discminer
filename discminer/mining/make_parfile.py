from discminer.mining_control import _mining_parfile

import os
import json
import warnings
import numpy as np

if __name__ == '__main__':
    parser = _mining_parfile(None)
    args = parser.parse_args()

#'kind' tags available: '', fixincl, I2pwl, I2pwlnosurf

#**************************
#CUSTOMISED PROPS FOR PLOTS
#**************************
#Put here any info you'd like to have access through the parfile.json in the analysis scripts 

custom_dict = {
    'chan_step': 2, #Plot channels every n steps, adjusted to 2 for CS 
    'nchans': 15, #Plot n channels
    'vlim': 0.2, #Velocity max and min plot limit
    'Llim': 0.2, #Line width max and min
    'Ilim': 14, #Peak intensity max and min
    
}

#****************
#DISCS BASIC INFO
#****************
discs = 'aatau cqtau dmtau hd34282 hd135344 j1604 j1615 j1842 j1852 hd143006 lkca15 mwc758 pds66 sycha v4046'
vel_sign = '1 -1 1 1 -1 1 1 -1 1 1 -1 1 -1 -1 1'
vel_sign_dict = dict(zip(discs.split(), np.asarray(vel_sign.split()).astype(int)))

incl_dict = { #Continuum inclination. Inclination sign assumes discminer convention
    'dmtau': 35.5,
    'aatau': -59.10,
    'lkca15': 50.4,
    'hd34282': -59.6,
    'mwc758': 21.00,
    'cqtau': -35.00,
    'sycha': -51.6,
    'pds66': -31.8,
    'hd135344': 15.00,
    'j1604': 5.9518, #12CO value, near 6 deg dust incl
    'j1615': 47.1,
    'v4046': -33.2,
    'j1842': 39.5,
    'j1852': -32.5,
    'hd143006': -16.916, #13CO value
    'mwc480': -37.000, 
    'twhyd': 5.800,
    'imlup': -47.500,
    'hd163296': 46.700,
    'as209': -35.000,
    'gmaur': 53.200,
    'hd169142': 13.0
} #These include Curone+2024 and literature values

gaps_dict = {
    'dmtau': [12.7, 71.8, 102.5],
    'aatau': [11.1, 64, 79.9, 105],
    'lkca15': [15.1, 86.5],
    'hd34282': [21.8, 60.6, 148, 188.2],
    'mwc758': [30, 60],
    'cqtau': [],
    'sycha': [33.3],
    'pds66': [],
    'hd135344': [66.7],
    'j1604': [],
    'j1615': [12.3, 82.6, 125.5],
    'v4046': [7.5, 20.4, 29.6],
    'j1842': [63.2],
    'j1852': [30.7],
    'hd143006': [22, 51],
    'mwc480': [76, 149],
    'twhyd': [82],
    'imlup': [116, 209],
    'hd163296': [49, 85, 145],
    'as209': [61, 100],
    'gmaur': [68, 142],
    'hd169142': [38]
} #exoALMA mm dust gaps from Curone+2024

rings_dict = {
    'dmtau': [24.1, 89.5, 111.4],
    'aatau': [41.5, 71.9, 89.8, 111],
    'lkca15': [68, 100],
    'hd34282': [46.8, 125.8, 159.1, 197.9],
    'mwc758': [47, 82],
    'cqtau': [41.5],
    'sycha': [101.2],
    'pds66': [0],
    'hd135344': [51, 78.6],
    'j1604': [82.7],
    'j1615': [25.9, 105.6, 132.9],
    'v4046': [2.3, 13.1, 27.2, 31.6],
    'j1842': [35.8, 69.7],
    'j1852': [19.2, 50.4],
    'hd143006': [41, 65],
    'mwc480': [98, 165],
    'twhyd': [70, 160], #Elevated optical depths from Teague+2022
    'imlup': [133, 220],
    'hd163296': [67, 101, 159],
    'as209': [74, 121],
    'gmaur': [42, 86, 163],
    'hd169142': [25, 59]
} #exoALMA mm dust rings from Curone+2024

kinks_dict = {
    'dmtau': [],
    'aatau': [],
    'lkca15': [],
    'hd34282': [],
    'mwc758': [35, 90],
    'cqtau': [],
    'sycha': [],
    'pds66': [],
    'hd135344': [100],
    'j1604': [],
    'j1615': [],
    'v4046': [],
    'j1842': [],
    'j1852': [],
    'hd143006': [51],
    'mwc480': [245],
    'twhyd': [],
    'imlup': [],
    'hd163296': [94, 260],
    'as209': [210],
    'gmaur': [],
    'hd169142': [40] 
} #Multiple references

#**************************
#GET TAGS AND PARS FROM LOG 
#**************************
files_dir = os.listdir()
file_data = ''

if args.log_file == '':
    log_file = files_dir[np.argmax(['log_pars' in f for f in files_dir])]
    for f in files_dir:
        if 'log_pars' in f and 'default' in f and 'converted' not in f:
            log_file = f
            break        
else:
    log_file = args.log_file
    
def get_tags_dict(log_file):
    global file_data
    tag_full = log_file.split('log_pars_')[1].split('_cube')[0]
    log_file_split = np.asarray(log_file.split('_'))
    tag_disc = log_file_split[2]
    tag_mol = log_file_split[3]
    tag_dv = log_file_split[4]
    tag_program = log_file_split[5]
    """
    exoalma = 'exoalma' in log_file_split[5]
    if not exoalma:
        raise ValueError('********Please rename your log file according to the naming conventions and try again********')
    """
    icube ,= np.where(log_file_split == 'cube')[0]
    di = icube-6
    tag_kind = log_file_split[6:icube]
    if len(tag_kind)==0: tag_kind=np.asarray(['allfree'])
    tag_walkers = log_file_split[7+di].split('w')[0]
    tag_steps = log_file_split[8+di].split('s')[0]

    #Try to get filename, distance and downsampling factors from prepare_data.py
    #pfile = np.asarray(os.listdir())[np.argmax(['prepare_data' in tmp for tmp in os.listdir()])]
    pfile = args.prepare_data
    prep = open(pfile, "r")
    downsamp = []
    file_read, file_clip = 0, 0
    for line in prep.readlines():
        if 'exit' in line and line[0]!='#':
            break
        if '.downsample(' in line:
            downsamp.append(line.split('(')[1].split(',')[0])
        if 'u.pc' in line:
            dpc = float(line.split('*')[0].split('=')[-1])
        if 'file_data' in line and line[0] !='#' and not file_read:
            file_read = 1
            exec(line, globals()) #file_data = 'cube.fits'; needs global vars info
        if 'datacube.clip' in line and not file_clip and line[0]!='#':
            file_clip = 1
            file_data += '_clipped'
            
    if len(downsamp)==2:
        pro, fit = np.sort(np.asarray(downsamp).astype(int))
    elif len(downsamp)==1:
        fit = int(downsamp[0])
        pro = 1
    elif len(downsamp)==0:
        pro = fit = 1
    else:
        pro = np.min(int(downsamp))
        fit = np.max(int(downsamp))
        message = "More than two downsampling factors were found in prepare_data.py script, taking smaller of all, you can specify the downsampling factor wished for prototype manually in parfile.json changing 'downsamp_pro'"
        warnings.warn(message)

    if pro > 1:
        file_data += '_downsamp_%dpix'%pro
    file_data += '.fits'

    return dict(file_data=file_data, prepare_script=pfile, log_file=log_file, tag=tag_full, disc=tag_disc, mol=tag_mol, dpc=dpc, dv=tag_dv, program=tag_program, kind=tag_kind.tolist(), nwalkers=int(tag_walkers), nsteps=int(tag_steps), downsamp_pro=int(pro), downsamp_fit=int(fit), downsamp_factor=(fit/pro)**2)


def get_base_pars(log_file, tags_dict):

    kind = tags_dict['kind']
    mol = tags_dict['mol']

    def read_logpars(log_file):
        log_pars = np.loadtxt(log_file)[1]
        header = np.loadtxt(log_file, comments=None, dtype=str, delimiter=' ', max_rows=1)
        header[1] = header[1][1:] #Remove first bracket of first entry
        header = [hdr[1:-2] for hdr in header[1:]] #Jump # and remove brackets and commas
        return log_pars, header
    
    try:
        log_file_split = np.asarray(log_file.split('_'))
        tag_disc = log_file_split[2]

        log_pars, header = read_logpars(log_file)

        #RENAME REPEATED TAGS IN HEADER
        hdr_arr = np.array(header)
        argI0, = np.where(hdr_arr == 'I0') #Array with one entry
        argL0, = np.where(hdr_arr == 'L0')
        argLs, = np.where(hdr_arr == 'Ls')
        argz0, = np.where(hdr_arr == 'z0')
        units = []
        
        if len(argz0)==1:
            argupper, arglower = argz0[0], np.inf
            header[argupper] = 'z0_upper'
            log_pars = np.append(log_pars, np.zeros(4))
            header += ['z0_lower', 'p_lower', 'Rb_lower', 'q_lower']
        elif len(argz0)==2:
            argupper, arglower = argz0
            header[argupper] = 'z0_upper'
            header[arglower] = 'z0_lower'
        else:
            argupper, arglower = np.inf, np.inf
            log_pars = np.append(log_pars, np.zeros(8))
            header += ['z0_upper', 'p_upper', 'Rb_upper', 'q_upper', 'z0_lower', 'p_lower', 'Rb_lower', 'q_lower']

            
        if 'fixincl' in kind:
            log_pars = np.append(log_pars, np.radians(incl_dict[tag_disc]))
            header += ['incl']

        if 'fix12co' in kind:
            #path_12co = os.getcwd().replace(mol, '12co').replace('b0p30','').replace('highres','') #last two skip dir from subfolder back to mainfolder
            path_12co = os.getcwd().split(mol)[0]+'12co'
            files_dir_12co = os.listdir(path_12co)

            for f in files_dir_12co:
                if 'log_pars' in f and 'default' in f:
                    log_file_12co = f
                    founddefault = True
                    break
                
            if not founddefault:
                raise FileNotFoundError('No default parameter file was found in the 12CO folder of the disc you wish to analyse. Make sure one of your parameter files is tagged as exoalmav1default.')

            log_pars_12co, header_12co = read_logpars(os.path.join(path_12co, log_file_12co))
            hdr_arr_12co = np.array(header_12co)

            if 'fixincl' in kind:
                hdr_fix = ['vsys', 'PA', 'xc', 'yc']
            else:
                hdr_fix = ['vsys', 'incl', 'PA', 'xc', 'yc']

            log_pars = np.append(
                log_pars,
                [
                    log_pars_12co[hdr_arr_12co == hdr_i] for hdr_i in hdr_fix
                ]
            )
            header += hdr_fix            

        if 'vel_sign' not in header:
            log_pars = np.append(log_pars, vel_sign_dict[tag_disc])
            header += ['vel_sign']

        if 'xc' not in header:
            log_pars = np.append(log_pars, 0)
            header += ['xc']

        if 'yc' not in header:
            log_pars = np.append(log_pars, 0)
            header += ['yc']
            
        for i in range(len(header)):
            hdr = header[i]
            if hdr in ['p', 'q', 'Rb', 'p0', 'p1']:
                diff_arg = np.array([i-argI0[0], i-argL0[0], i-argLs[0], i-argupper, i-arglower])                
                diff_arg = np.where(diff_arg<=0, np.inf, diff_arg)
                diff_min = np.argmin(diff_arg)
                if diff_min==0: header[i]+='_I0'
                if diff_min==1: header[i]+='_L0'
                if diff_min==2: header[i]+='_Ls'
                if diff_min==3: header[i]+='_upper'
                if diff_min==4: header[i]+='_lower'
                    
            #UNITS STUFF
            if 'R' in hdr or 'xc' in hdr or 'yc' in hdr or 'z' in hdr:
                units.append('au')
            elif 'incl' in hdr or 'PA' in hdr:
                units.append('rad')
            elif 'Ls' in hdr or 'p' in hdr or 'q' in hdr or 'vel_sign' in hdr:
                units.append('none')
            elif 'I0' in hdr:
                units.append('Jy/pix')
            elif 'L0' in hdr or 'vsys' in hdr:
                units.append('km/s')
            elif 'Mstar' in hdr:
                units.append('Msun')
            else:
                units.append('unknown')
                        
        #********
        #SIMPLE DICT OF BEST FIT PARS AND UNITS
        par_dict = dict(zip(header, log_pars))
        uni_dict = dict(zip(header, units))        

        #********
        #CLASSIFY PARS IN ATTRIBUTES
        par_dict_att = {
            'velocity': {}, 'orientation': {},
            'intensity': {}, 'linewidth': {}, 'lineslope': {},
            'height_upper': {}, 'height_lower': {}
        }
        uni_dict_att = {
            'velocity': {}, 'orientation': {},
            'intensity': {}, 'linewidth': {}, 'lineslope': {},
            'height_upper': {}, 'height_lower': {}
        }

        def update_dict(att, key):
            if key=='vel_sign':
                key_sp = key
            else:
                key_sp = key.split('_')[0]
            par_dict_att[att].update({key_sp: par_dict[key]})
            uni_dict_att[att].update({key_sp: uni_dict[key]})            
            
        for key in par_dict:
            if key in ['Mstar', 'vsys', 'vel_sign']:
                update_dict('velocity', key)
            if key in ['incl', 'PA', 'xc', 'yc']:
                update_dict('orientation', key)
            if 'I0' in key or 'Rout' in key or 'Rbreak' in key:
                update_dict('intensity', key)
            if 'L0' in key:
                update_dict('linewidth', key)
            if 'Ls' in key:
                update_dict('lineslope', key)
            if 'upper' in key:
                update_dict('height_upper', key)
            if 'lower' in key:
                update_dict('height_lower', key)

        return par_dict_att, uni_dict_att
    
    except FileNotFoundError:
        message = 'Log par file not found: %s'%log_file
        warnings.warn(message)
        return None, None

#*************
#MAKE PAR FILE
#*************    
def make_json(dicts_list=[], keys_list=[], filename=args.json_file):
    master_dict = {key: dicts_list[i] for i,key in enumerate(keys_list)}

    def make():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(master_dict, f, ensure_ascii=False, indent=4)

    if args.overwrite:
        if args.json_file in os.listdir():
            message = 'Overwriting JSON parfile...'
            #warnings.warn(message)
            print ('*'*10 + message)
            if args.reset:
                message = "'custom' dictionary was reset to default values..."
                warnings.warn(message)                

            if not args.reset:
                with open(args.json_file) as json_file:
                    pars = json.load(json_file)
                message = "Forwarding 'custom' dictionary to the new %s..."%args.json_file
                print ('*'*10 + message)
                    
                master_dict['custom'].update(pars['custom'])
        make()
        
    else:
        if args.json_file in os.listdir():
            message = '\n********JSON parfile exists and overwrite mode is off, therefore NO output was produced********'
            warnings.warn(message)
        else:
            make()

def make_all():
    tags_dict = get_tags_dict(log_file)
    pars_dict, units_dict = get_base_pars(log_file, tags_dict)
    for tk in tags_dict['kind']:
        if 'nosurf' in tk:
            pars_dict['intensity'].update({'q': 0.0})
            units_dict['intensity'].update({'q': "none"})
            pars_dict['linewidth'].update({'q': 0.0})
            units_dict['linewidth'].update({'q': "none"})
            pars_dict['height_upper'].update({'z0': 0.0})
            units_dict['height_upper'].update({'z0': "au"})
            pars_dict['height_lower'].update({'z0': 0.0})
            units_dict['height_lower'].update({'z0': "au"})
            
    try:
        custom_dict.update(gaps=gaps_dict[tags_dict['disc']], rings=rings_dict[tags_dict['disc']], kinks=kinks_dict[tags_dict['disc']])
    except KeyError:
        custom_dict.update(gaps=[], rings=[], kinks=[])
            
    make_json(dicts_list = [custom_dict, tags_dict, pars_dict, units_dict], keys_list = ['custom', 'metadata', 'best_fit', 'units'])

if args.download_cube:
    from urllib.request import urlretrieve

    url_nrao = 'https://bulk.cv.nrao.edu/exoalma/ALMA_self_cal_data/images_v1/ds_cubes/'

    urlretrieve(url=url_nrao+file_data,filename=file_data)

make_all()
