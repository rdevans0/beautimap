
import os
from matplotlib.colors import is_color_like


def find_and_load_colorschemes(folder):
    colorschemes = {}
    
    files = os.listdir(folder)
    for cs_file in files:
        # Skip files that don't look like colorschemes
        if not cs_file.endswith('.py'):
            continue
        if cs_file.startswith('__'):
            continue
        
        
        # Read the plugin text
        cs_path = os.path.join(folder, cs_file)
        
        err_ctx = cs_path + ' is malformed: '
        cs = load_colorscheme(cs_path)
        
        cs_name = cs_file.replace('.py','')
        if cs_name in colorschemes:
            raise Exception(err_ctx + 'Colorscheme {} already exists!')
        colorschemes[cs_name] = cs
    
    return colorschemes

def load_colorscheme(cs_path):
    
    err_ctx = cs_path + ' is malformed: '
    with open(cs_path,'r') as fid:
        cs_text = fid.read()
    
    try:
        cs = eval(cs_text)
    except Exception as err:
        raise Exception(err_ctx + 'Could not parse file') from err
    
    # Some basic checks
    if not isinstance(cs, dict):
        raise Exception(err_ctx + 'Not a dict!')
        
    if not len(cs):
        raise Exception(err_ctx + 'Size is 0!')
        
    if 'land' not in cs and 'water' not in cs:
        raise Exception(err_ctx + 'Land and water keys missing!')
    
    for c in cs.values():
        if c is None:
            continue
        elif not isinstance(c, str):
            raise Exception(err_ctx + 'Values are not all strings!')
        elif not is_color_like(c):
            raise Exception(err_ctx + c + ' is not a valid matplotlib color!')
            
    
    return cs
        
    

_MY_FOLDER = os.path.dirname(os.path.realpath(__file__))
COLORSCHEMES = find_and_load_colorschemes(_MY_FOLDER)

