#!/usr/bin/env python3

import sys
import os
import time

import argparse

from collections import defaultdict

from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib as mpl
from matplotlib.colors import is_color_like
import matplotlib.pyplot as plt

    
import numpy as np

import geopandas as gpd

import osmnx as ox
from osmnx import utils_graph

import PIL


from colorschemes import COLORSCHEMES, load_colorscheme


_MY_FOLDER = os.path.dirname(os.path.realpath(__file__))

_DEFAULT_ARGS = {
    'point': (0.0, 0.0),  # Lat, Lng
    'dist': 1000,         # meters of longest edge
    'water_shift': (0.0, 0.0),  # +Lat, +Lng
    'size': (1920, 1080), # pixels
    'dpi': 500,           # Figure export DPI
    'color': 'neon',      # Colorscheme to use
    'replace_color': [],  # Override colors in pallette
    'road_width': 0.5,    # Base road width
    'show': False,        # Whether to show image
    'adjust': None,       # Adjustment for figure size
    'overscan': 1.1,      # Amount of bbox overscan to ensure all geometries caught
    'verbose': 0,         # Print everything
    'out': 'map.png',     # Output file and format
    
    # For downloading water polygons
    'water_shape_url': 'https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip',
    'water_shape_folder': '/home/rdevans/Documents/map_generator_cache', #_MY_FOLDER,
    'water_shape_file': 'water-polygons-split-4326/water_polygons.shp',
}
D = _DEFAULT_ARGS
    

def parse_args(argv=None, defaults=_DEFAULT_ARGS):
    parser = argparse.ArgumentParser(description='Map generator using osmnx')
    
    ds = lambda k: ' (default: {})'.format(defaults[k])  # Default string
    
    parser.add_argument('--point','-p', nargs=2, default=None, type=float,
                        help='Lattitude/Longitude for the center of the plot' + ds('point'))
    
    parser.add_argument('--dist', '-d', default=None, type=float, 
                        help='Length of the longest dimension of the image (in meters)' + ds('dist'))
    
    parser.add_argument('--water_shift', nargs=2, default=None, type=float,
                        help='Lat/Lng shift for locating water geometries. Use if plotting a coastline and water geometry does not show' + ds('water_shift'))
    
    parser.add_argument('--size','-s', nargs=2, default=None, type=int,
                        help='Resulting image size in pixels' + ds('size'))
    
    parser.add_argument('--dpi', default=None, type=int,
                        help='Image export DPI. Controls figure canvas size' + ds('dpi'))
    
    avail_colors = ', '.join(COLORSCHEMES.keys())
    parser.add_argument('--color', default=None, type=str,
                        help='Colorscheme to use. Available under ./colorshemes: ' + avail_colors + ds('color'))
    
    parser.add_argument('--replace_color', '-r', default=None, type=str, metavar=('key','color'), nargs=2, action='append',
                        help='Replace color of key with the specified color. Accepts Matplotlib colors, or None')
    
    
    parser.add_argument('--road_width', default=None, type=float,
                        help='Plotted road width.' + ds('road_width'))
    
    parser.add_argument('--adjust', default=None, type=float,
                        help='Override automatic figure size adjustment to this amount')
    
    parser.add_argument('--overscan', default=None, type=float,
                        help='Amount to overscan figure to ensure all geometries plotted.' + ds('overscan'))
    
    parser.add_argument('--show', action='store_true', 
                        help='Show the figure after writing to file')
    
    parser.add_argument('--out', '-o', type=str, default=None,
                        help='Output image' + ds('out'))
    
    
    parser.add_argument('--water_shape_url', default=None, type=str,
                        help='[ADVANCED] URL to download water polygons' + ds('water_shape_url'))
    
    parser.add_argument('--water_shape_folder', default=None, type=str,
                        help='[ADVANCED] folder to store downloaded water polygons' + ds('water_shape_folder'))
    
    parser.add_argument('--water_shape_file', default=None, type=str,
                        help='[ADVANCED] Path to water shape file (relative to water_shape_folder)' + ds('water_shape_file'))
                        
    
    parser.add_argument('-v', '--verbose', action='store_const', const=1, default=None, 
                        help='Print more output')
    
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='.py dict holding any of the above arguments. Overridden by command line arguments')
    
    
    args = parser.parse_args(argv) 
    kwargs = vars(args)
    source = {k:'COMMAND LINE' for k in kwargs.keys()}
    
    def update(d1, d1_names, d2, d2_name, empty=None):
        for k,v in d2.items():
            if d1[k] == empty:
                d1[k] = v
                d1_names[k] = d2_name
    
    # Apply config
    if kwargs['config']:
        with open(kwargs['config'], 'r') as fid:
            text = fid.read()
        try:
            config = eval(text)
        except Exception as err:
            # Add error context and re-raise
            print('Problem parsing file. Text is :')
            for i, s in enumerate(text.split('\n')):
                print('{: 3d}: {:s}'.format(i+1,s))
                
            raise Exception('Error parsing ' + kwargs['config']) from err
        update(kwargs, source, config, 'CONFIG ' + kwargs['config'])
    
    # Apply defaults
    update(kwargs, source, defaults, 'DEFAULT')
    
    del kwargs['config']
    
    # Properly parse and check the replacement colors
    if kwargs['replace_color']:
    
        # Load colorscheme
        color_scheme_name = kwargs['color']
        if color_scheme_name in COLORSCHEMES:
            pallette = COLORSCHEMES[color_scheme_name]  # in ./colorschemes
        else:
            pallette = load_colorscheme(color_scheme_name)  # load from file
        avail_keys = list(pallette.keys())
        
        replacements = {}
        for k,v in kwargs['replace_color']:
            if k not in pallette:
                parser.error('{} is not a valid key. Options are {}'.format(k, avail_keys))
            if v.lower() == 'none':
                replacements[k] = None
            elif not is_color_like(v):
                parser.error('{} does not define a valid color'.format(v))
            else:
                replacements[k] = v
        
        kwargs['replace_color'] = replacements
                
    
    ext = kwargs['out'].split('.')[-1]
    if ext not in 'eps,jpeg,jpg,pdf,pgf,png,ps,raw,rgba,svg,svgz,tif,tiff'.split(','):
        parser.error('{} does not have an image extension'.format(kwargs['out']))
    
    
    if args.verbose >= 1:
        print('ARGUMENTS AND SOURCES:')
        for k,v in sorted(kwargs.items()):
            print('{!r:32}: {!r:32}, # {}'.format(k, v, source[k]))
        print('')
    
        
    return kwargs
    


def bbox2d_from_point(point, dist=1000):
    """ bbox_from_point that allows dx and dy, modified from osmnx.utils_geo.bbox_from_point """
    
    earth_radius = 6_371_009  # meters
    lat, lng = point
    
    if hasattr(dist, '__len__'):
        if len(dist) != 2:
            raise ValueError('dist must be (x,y) or xy, not len == {}'.format(len(dist)))
        dx, dy = dist
    else:
        dx = dy = float(dist)

    delta_lat = (dy / earth_radius) * (180 / np.pi)
    delta_lng = (dx / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi / 180)
    north = lat + delta_lat
    south = lat - delta_lat
    east = lng + delta_lng
    west = lng - delta_lng    
    
    return north, south, east, west

    
def main(point=(0.,0.), dist=1000, water_shift=(0., 0.), size=(1920,1080), 
         dpi=500, color='neon', replace_color={}, road_width=3.5, show=False, adjust=None, 
         overscan=1.1, out='map.png', verbose=0, 
         water_shape_url=D['water_shape_url'],
         water_shape_folder=D['water_shape_folder'],
         water_shape_file=D['water_shape_file'],
         max_fig_dim=13):
    
    size = np.asarray(size)
    pixels = size.astype('i')
    if (size != pixels).all():
        raise ValueError('Size {} is not an integer number of pixels'.format(size))
    
    
    # Load colorscheme
    if color in COLORSCHEMES:
        pallette = COLORSCHEMES[color]  # in ./colorschemes
    else:
        pallette = load_colorscheme(color)  # load from file
    
    if replace_color:
        pallette = pallette.copy()
        for k,v in replace_color.items():
            pallette[k] = v
    
    street_widths = {
        'footway' : 0.5,
        'steps' : 0.5,
        'pedestrian' : 0.5,
        'path' : 0.5,
        'track' : 0.5,
        'service' : 1,
        'residential' : 1,
        'tertiary': 1, 
        'tertiary_link': 1, 
        'secondary' : 1.8,
        'secondary_link' : 1.8,
        'primary' : 1.8,
        'primary_link' : 1.8,
        'motorway' : 1.8,
        'motorway_link' : 1.8,
        }
    
    street_mult = road_width
    for k in street_widths.keys():
        street_widths[k] *= street_mult
    
    street_colors = {
        'footway' : pallette['path'],
        'steps' : pallette['path'],
        'pedestrian' : pallette['path'],
        'path' : pallette['path'],
        'track' : pallette['path'],
        'service' : pallette['road1'],
        'residential' : pallette['road1'],
        'tertiary' : pallette['road2'],
        'tertiary_link' : pallette['road2'],
        'secondary' : pallette['road2'],
        'secondary_link' : pallette['road2'],
        'primary' : pallette['road2'],
        'primary_link' : pallette['road2'],
        'motorway' : pallette['road2'],
        'motorway_link' : pallette['road2'],
        }
    
    color_remap = {
        'strait':'water',
        'bay':'water',
        'river':'water',
        'harbour':'water',
        'basin':'water',
        # 'spring':'water', # Tends to be a marker, not geometry
        'beach':'sand',
        'farmland':'grass',
        'farmyard':'grass',
        'meadow':'grass',
        }
    
    
    ################################################################################################
    # Setup
    
    # Correct for aspect ratio
    ar = pixels[0] / pixels[1]
    if ar < 1: 
        raise ValueError('Assumes widescreen format')
    large_dim = 0
    small_dim = 1
    
    dist_2d = np.empty((2,), dtype='f')
    dist_2d[large_dim] = dist
    dist_2d[small_dim] = dist/ar
    
    bbox = bbox2d_from_point(point, dist_2d)
    big_bbox = bbox2d_from_point(point, dist_2d*overscan)
    water_point = np.array(water_shift) + np.array(point)
    water_bbox = bbox2d_from_point(water_point, dist_2d*2)
    
    
    # This snippet gets the available screen space in inches
    # window = plt.get_current_fig_manager().window
    # screen_size = np.array([window.wm_maxsize()])
    # fig = plt.figure()
    # fig_dpi = fig.dpi
    # screen_size = screen_size / fig.dpi
    
    # Create plot
    kwargs = dict(retain_all=True, simplify=True, network_type='all')
    init_figsize = pixels/dpi
    
    if (init_figsize[large_dim] > max_fig_dim).any():
        # Adjust the DPI
        # Pyplot cannot scale a figure larger than the window size, which 
        # means that we need to keep the figure on a single monitor.
        # The pyplot default is 100 dpi, so rendering at a multiple of 100
        # makes it more likely to find the a figure size that renders to our 
        # target resolution. 
        new_dpi = int(pixels[large_dim] / max_fig_dim) # Min DPI that fits
        new_dpi = int(np.ceil(new_dpi / 100) * 100)    # Round up to nearest 100
        
        print('WARNING: Increasing DPI from {} to {} to keep image on one screen'.format(dpi, new_dpi))
        dpi = new_dpi
        init_figsize = pixels/dpi
    
    fig, ax = plt.subplots(1,1, figsize=init_figsize)
    
        
    fig.tight_layout(pad=0)
    
    if not show:
        print('WARNING: Turning on "show", disabling does not function correctly')
        show = True
    
    if show:
        plt.ion()
        plt.show()
    else:
        pass
        #plt.show(block=False)
        
    def draw():
        plt.show(block=False)
        plt.draw()
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        pass
        
        
    land_color = pallette['land']
    water_color = pallette['water']
    ax.set_facecolor(land_color)
    
    draw()
    
    ################################################################################################
    # Custom method to track changes to the axis
    
    artist_map = {}
    def register_new_artists(key, ax=ax, artist_map=artist_map):
        old_artists = sum(artist_map.values(), [])
        old_artists = set(old_artists)
        
        new_artists = []
        print('Locating artists for "{}"'.format(key))
        for artist in ax.get_children():
            if artist not in old_artists and artist not in new_artists:
                new_artists.append(artist)
                print('    add {}'.format(artist))
        
        artist_map[key] = new_artists
    
    def change_artist_color(key, color, ax=ax, artist_map=artist_map):
        artists = artist_map[key]
        old_color = []
        for artist in artists:
            if isinstance(artist, mpl.collections.LineCollection):
                old_color.append(artist.get_ec())
                artist.set_ec(color)
            elif hasattr(artist, 'set_fc'):
                old_color.append(artist.get_fc())
                artist.set_fc(color)
    
    register_new_artists('axis')
    
    
    ################################################################################################
    # Get water
    
    
    water_poly = read_water_polygons(bbox=water_bbox, 
        url=water_shape_url, folder=water_shape_folder, file=water_shape_file,
        verbose=verbose)
    
    if verbose >= 1:
        print('Downloading water geometries')
    water_geom = ox.geometries_from_bbox(*bbox, {'natural':'water'})
    
    if verbose >= 1:
        print('Plotting water')
        
    ox.plot_footprints(water_poly, ax=ax, bbox=bbox, color=water_color)
    register_new_artists('water/poly')
    
    
    sx,ex,sy,ey = ax.get_xlim() + ax.get_ylim()
    W = (ex - sx) / 2
    H = (ey - sy) / 2
    cx = (ex + sx) / 2
    cy = (ey + sy) / 2
    ax_lims = [(cx - W*0.9, cx + W*0.9), (cy - H*0.9, cy + H*0.9)] # 10% zoom 
    
    ax = water_geom.plot(ax=ax, fc=water_color, markersize=0)
    register_new_artists('water/geom')
    
    draw()
        
    
    ################################################################################################
    # Natural Geometries
    # Reused for landuse geometries
    def print_value_counts(category, geom):
        print('Available {}:'.format(category))
        for k,v in geom[category].value_counts().items():
            print('    {:<20s} {:>4d}'.format(k,v))
            
    
    
    def plot_geom(category, key, geom):
        artist_key = '{}/{}'.format(category, key)
        
        if key in pallette:
            color = pallette[key]
        elif key in color_remap:
            remap_key = color_remap[key]
            color = pallette[remap_key]
        else:
            raise Exception('Could not find color for {}/{}'.format(category, key))
        
        if color is None:
            print('Skipping %s (color is None)...'%artist_key)
            return
    
        poly_keys = geom[category].isin([key])
        gdf_polys = geom[poly_keys]
        
        if gdf_polys.size:
            if verbose:
                print('Plotting %s...'%artist_key)
                
            gdf_polys.plot(ax=ax, color=color)
            register_new_artists(artist_key)
            
        else:
            if verbose:
                print('Skipping %s (empty)...'%artist_key)
            
        
        
    print('Plotting natural geometries...')
    natural_geom = ox.geometries_from_bbox(*big_bbox, {'natural':True})   
    
    
    water_keys = ('water','strait','bay','river')
    natural_keys = (
        'wood','beach','sand',
        #'tree','coastline', 'spring' #Tend to be markers, not geometry
        )
    
                  
    
    print_value_counts('natural', natural_geom)
    
    for k in water_keys:
        plot_geom('natural', k, natural_geom)
    draw()
    
    
    # Uncomment this to help with targeting a specific location
    # if show:
    #     print('Showing water and land')
    #     _ = input('Press ENTER to continue...')
    
    for k in natural_keys:
        plot_geom('natural', k, natural_geom)
    
    
    
    ################################################################################################
    # Road graph
    print('Retrieving ROAD graph...')
    road_graph = ox.graph_from_bbox(*big_bbox, **kwargs)
    
    #road_graph = simplification.simplify_graph(road_graph, strict=False)
    road_graph = utils_graph.get_undirected(road_graph)
    
    # Set sizes by street type
    edge_linewidths = []
    edge_colors = []
    
    street_type_counts = defaultdict(int)
    
    for _, _, d in road_graph.edges(keys=False, data=True):
        street_type = d["highway"][0] if isinstance(d["highway"], list) else d["highway"]
        street_type_counts[street_type] += 1
        if street_type in street_widths:
            edge_linewidths.append(street_widths[street_type])
            edge_colors.append(street_colors[street_type])
        else:
            edge_linewidths.append(road_width)
            edge_colors.append(pallette['road1'])
    
    print('Available street types: ')
    street_type_counts_swap = [(v,k) for k,v in street_type_counts.items()]
    for v,k in sorted(street_type_counts_swap, reverse=True):
            print('    {:<20s} {:>4d}'.format(k,v))
        
    
    # do NOT map road length
    #road_colors = pallette['road1']
    road_colors = edge_colors
    #road_widths = road_width
    road_widths = edge_linewidths
    

    print('Plotting...')
    fig, ax = ox.plot_graph(road_graph, ax, node_size=0, figsize=(27, 40), 
                            dpi=300, 
                            save=False, edge_color=road_colors,
                            edge_linewidth=road_widths, edge_alpha=1)
    
    register_new_artists('road')
    
    ################################################################################################
    # Manmade Geometries
    print('Plotting landuse geometries...')
    landuse_geom = ox.geometries_from_bbox(*big_bbox, {'landuse':True})
    
    print_value_counts('landuse', landuse_geom)
    landuse_keys = ('grass','farmland','meadow',
                    'harbour','basin')
        
    for k in landuse_keys:
        plot_geom('landuse', k, landuse_geom)
    
    draw()
    
    
    # Buildings
    if pallette['building'] is not None:
        building_geom = ox.geometries_from_bbox(*big_bbox, {'building':True})
        ox.plot_footprints(building_geom, ax, color=pallette['building'])
    
    register_new_artists('building')
    
    draw()
    
    
    ################################################################################################
    # Finalize
    ax.set_xlim(ax_lims[0])
    ax.set_ylim(ax_lims[1])
    
    figsize = pixels/dpi
    fig.set_size_inches(figsize)
    
    
    kwargs = dict(bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), transparent=False)
    #fig.savefig(out, dpi=dpi, **kwargs)
    savefig_pixels(fig, out, pixels, dpi=dpi, verbose=verbose, **kwargs)
    
    print('Done')

def savefig_pixels(fig, out, target_pixels, dpi=200, verbose=False, **kwargs):
    
    if verbose:
        print('Attempting to save {} as {} x {} image @ {} dpi'.format(out, *target_pixels, dpi))
    
    # A hacky way to get pyplot to export exact figure inches. Actually exports 4 figures
    def savefig(size_inches):
        size_inches = tuple(size_inches)
        
        if verbose:
            print('Try figsize {:.2f} x {:.2f}in @ {} dpi'.format(*size_inches, dpi))
            
        fig.set_size_inches(size_inches)
        fig.savefig(out, dpi=dpi, **kwargs)
        
        
        # Return actual pixel size of image
        with PIL.Image.open(out) as img:
            size_pixels = img.size
        
        if verbose:
            print('    -> {} x {} pixels'.format(*size_pixels))
        return np.array(size_pixels)
    
    # Export a figure we think is the right size:
    wh_in = np.zeros((2,3)).astype('f')
    wh_pix = wh_in.copy().astype('i')
    
    
    # Try 1: Guess correct size
    wh_in[:,0] = target_pixels / dpi  # First try, why this doesn't work... idk...
    wh_pix[:,0] = savefig(wh_in[:,0])
    
    if (wh_pix[:,0] == target_pixels).all():
        return
    
    # Try 2: Interpolate to find size, but overestimate by 20%
    ppi = wh_pix[:,0] / wh_in[:,0]
    wh_in[:,1] = target_pixels * 1.2 / ppi
    wh_pix[:,1] = savefig(wh_in[:,1])
    
    if (wh_pix[:,1] == target_pixels).all():
        return
    
    
    # Try 3: Interpolate to find size, but underestimate by 20%
    ppi = wh_pix[:,1] / wh_in[:,1]
    wh_in[:,2] = target_pixels * 0.8 / ppi
    wh_pix[:,2] = savefig(wh_in[:,2])
    
    if (wh_pix[:,2] == target_pixels).all():
        return
    
    # Ensure we have bracketed the dimensions
    has_smaller_in_dim = (wh_pix < target_pixels[:,None]).any(axis=1) # (2,)
    has_larger_in_dim = (wh_pix > target_pixels[:,None]).any(axis=1) # (2,)
    
    if not has_smaller_in_dim.all():
        raise Exception('Failed to bracket image size: could not save a smaller image')
    if not has_larger_in_dim.all():
        raise Exception('Failed to bracket image size: could not save a larger image')
    
    
    
    # Try 4: Hopefully we have bracketed the true size, so now linearly fit
    pw = np.polynomial.Polynomial.fit(wh_pix[0,:], wh_in[0,:], deg=1)
    ph = np.polynomial.Polynomial.fit(wh_pix[1,:], wh_in[1,:], deg=1)
    
    w = pw(target_pixels[0])
    h = ph(target_pixels[1])
    
    cur_in = np.array([w,h], dtype='f')
    cur_pix = savefig(cur_in)
        
    
    if (cur_pix != target_pixels).any():
        
            print('Failed to achieve target image size {} x {} pixels'.format(*target_pixels))
            print('Best attempt was {} x {}'.format(*cur_pix))
        
    return
    
    
    

def map_by_length(data, lengths, choices, default=None):
    
    def mapfn(item):
        if default is not None and 'length' not in item:
            return default
        length = item.get('length', -1)
        ind = (length >= lengths).nonzero()[0][-1]
        return choices[ind]
    
    return [mapfn(x) for x in data]

class IntervalReporter:
    def __init__(self, interval_sec=3):
        self.interval_sec = interval_sec
        self.last = -1
    
    def __call__(self, chunk_num, chunk_size, total_size):
        now = time.time()
        done_size = chunk_num * chunk_size
        percent = 100 * done_size / total_size # Might be negative is total is unknown
        percent = max(percent, 0.)
        
        if now > (self.last + self.interval_sec):
            print('{:4.1f}%  ({:6.2f}MB / {:6.2f}MB)'.format(percent, done_size/1e6, total_size/1e6))
            self.last = now
            
        

def read_water_polygons(folder=D['water_shape_folder'], 
                        file=D['water_shape_file'], 
                        url=D['water_shape_url'], 
                        zipfile='water-polygons.zip',
                        verbose=0,
                        bbox=None):
    
    if not os.path.exists(folder):
        raise Exception('Water shape folder {} does not exist!'.format(folder))
    
    full_path = os.path.join(folder, file)
    full_zip = os.path.join(folder, zipfile)
    
    if not os.path.exists(full_path):
        # Download zip
        if not os.path.exists(full_zip):
            print('Downloading from {} to {}'.format(url, full_zip))
            report = IntervalReporter(3)
            
            try:
                urlretrieve(url, filename=full_zip, reporthook=report)
            except Exception as err:
                os.remove(full_zip)
                raise err
            
            if not os.path.exist(full_zip):
                raise Exception('Download {} failed: {} does not exist'.format(url, full_zip))
                
        elif verbose >= 1:
            print('Found {}'.format(full_zip))
            
        
        # Extract zip
        with ZipFile(full_zip, 'r') as zfid:
            print('Extracting {} to {}'.format(full_zip, folder))
            zfid.extractall(path=folder)
            
            if not os.path.exists(full_path):
                raise Exception('Extract failed: {} does not exist'.format(full_path))
    
    elif verbose >= 1:
        print('Found {}'.format(full_path))
    
    if verbose >= 1:
        print('Reading {}'.format(full_path))
    polys = gpd.read_file(full_path, bbox=bbox)
    return polys
    

###############################################################################
###############################################################################

def test_savefig():
    out = 'map_test.png'
    dpi = 200
    target_pixels = np.array([1200, 800])

    ar = target_pixels[0] / target_pixels[1]
    
    W = int(100*ar)
    H = 100
    
    init_figsize = target_pixels/dpi
    fig, ax = plt.subplots(1,1, figsize=init_figsize)
    fig.tight_layout(pad=0)
    
    ax.set_facecolor('#FF0000')
    plt.axis('off')
    
    Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
    plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest',
               extent=(0, W, 0, H))
    
    plt.ion()
    plt.show()
    
    
    
    kwargs = dict(bbox_inches='tight', 
                  pad_inches=0, 
                  facecolor=fig.get_facecolor(), 
                  transparent=False)
    
    savefig_pixels(fig, out, target_pixels, dpi=dpi, verbose=1, **kwargs)
    
        

if __name__ == '__main__':
    kwargs = parse_args()
    
    main(**kwargs)
