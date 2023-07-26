import os
import sys
import json
import warnings
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(prog='make json parfile', description='Make master json file with model information to use as input for analysis scripts')
parser.add_argument('-f', '--log_file', default='', type=str, help="if none tries to guess input log file")
parser.add_argument('-p', '--prepare_file', default='prepare_data.py', type=str, help="script employed to clip and downsample the cube of interest. DEFAULTS to prepare_data.py")
parser.add_argument('-o', '--overwrite', default=0, type=int, help="overwrite if parfile.json exists. DEFAULTS to 0")
parser.add_argument('-r', '--reset', default=0, type=int, help="If (1) and overwrite, reset 'custom' dictionary in new parfile. If not (0) and overwrite, forward 'custom' dictionary and rewrite metadata and model parameters only. DEFAULTS to 0")
args = parser.parse_args()

#'kind' tags available: '', fixincl, I2pwl, I2pwlnosurf

#**************************
#CUSTOMISED PROPS FOR PLOTS
#**************************
#Put here any info you'd like to have access through the parfile.json in the analysis scripts 

custom_dict = {
    'chan_step': 4, #Plot channels every n steps
    'nchans': 15, #Plot n channels
    'vlim': 0.2, #Velocity max and min plot limit
    'Llim': 0.2, #Line width max and min
    'Ilim': 15, #Peak intensity max and min

    
}

#****************
#DISCS BASIC INFO
#****************
discs = 'mwc480 twhyd imlup hd163296'
vel_sign = '-1 1 1 1'
vel_sign_dict = dict(zip(discs.split(), np.asarray(vel_sign.split()).astype(int)))

incl_dict = {
    'mwc480': -37.0,
    'twhyd': 5.8,
    'imlup': -47.5,
    'hd163296': 46.7
}

gaps_dict = {
    'mwc480': [76, 149],
    'twhyd': [82],
    'imlup': [116, 209],
    'hd163296': [49, 85, 145]
} #mm dust gaps

rings_dict = {
    'mwc480': [98, 165],
    'twhyd': [70, 160], #rings of elevated optical depth (Teague+2022)
    'imlup': [133, 220],
    'hd163296': [67, 101, 159]
} #mm dust rings

#**************************
#GET TAGS AND PARS FROM LOG 
#**************************
files_dir = os.listdir()
file_data = ''

if args.log_file == '':
    log_file = files_dir[np.argmax(['log_pars' in f for f in files_dir])]
    for f in files_dir:
        if 'log_pars' in f and 'default' in f:
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
    icube ,= np.where(log_file_split == 'cube')[0]
    di = icube-6
    tag_kind = log_file_split[6:icube]
    if len(tag_kind)==0: tag_kind=np.asarray(['allfree'])
    tag_walkers = log_file_split[7+di].split('w')[0]
    tag_steps = log_file_split[8+di].split('s')[0]

    #Try to get filename, distance and downsampling factors from prepare_data.py
    pfile = args.prepare_file
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

    return dict(file_data=file_data, prepare_script=pfile, log_pars=log_file, tag=tag_full, disc=tag_disc, mol=tag_mol, dpc=dpc, dv=tag_dv, program=tag_program, kind=tag_kind.tolist(), nwalkers=int(tag_walkers), nsteps=int(tag_steps), downsamp_pro=int(pro), downsamp_fit=int(fit), downsamp_factor=(fit/pro)**2)


def get_base_pars(log_file, kind=[]):
    try:
        log_pars = np.loadtxt(log_file)[1]
        log_file_split = np.asarray(log_file.split('_'))
        tag_disc = log_file_split[2]
        
        header = np.loadtxt(log_file, comments=None, dtype=str, delimiter=' ', max_rows=1)
        header[1] = header[1][1:] #Remove first bracket of first entry
        header = [hdr[1:-2] for hdr in header[1:]] #Jump # and remove brackets and commas

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
def make_json(dicts_list=[], keys_list=[], filename='parfile.json'):
    master_dict = {key: dicts_list[i] for i,key in enumerate(keys_list)}

    def make():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(master_dict, f, ensure_ascii=False, indent=4)

    if args.overwrite:
        if 'parfile.json' in os.listdir():
            message = 'Overwriting JSON parfile...'
            #warnings.warn(message)
            print ('*'*10 + message)
            if args.reset:
                message = "'custom' dictionary was reset to default values..."
                warnings.warn(message)                

            if not args.reset:
                with open('parfile.json') as json_file:
                    pars = json.load(json_file)
                message = "Forwarding 'custom' dictionary to the new parfile.json..."
                print ('*'*10 + message)
                    
                master_dict['custom'].update(pars['custom'])
        make()
        
    else:
        if 'parfile.json' in os.listdir():
            message = '\n********JSON parfile exists and overwrite mode is off, therefore NO output was produced********'
            warnings.warn(message)
        else:
            make()

if __name__ == '__main__':
    tags_dict = get_tags_dict(log_file)
    pars_dict, units_dict = get_base_pars(log_file, kind=tags_dict['kind'])
    for tk in tags_dict['kind']:
        if 'nosurf' in tk:
            pars_dict['intensity'].update({'q': 0.0})
            units_dict['intensity'].update({'q': "none"})            
            pars_dict['linewidth'].update({'q': 0.0})
            units_dict['linewidth'].update({'q': "none"})

    try:
        custom_dict.update(gaps=gaps_dict[tags_dict['disc']], rings=rings_dict[tags_dict['disc']])
    except KeyError:
        custom_dict.update(gaps=[], rings=[])
    
    make_json(dicts_list = [custom_dict, tags_dict, pars_dict, units_dict], keys_list = ['custom', 'metadata', 'best_fit', 'units'])
