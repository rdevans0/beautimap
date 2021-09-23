
# Beautimap
A beautiful and configurable map generator for python using osmnx and matplotlib. 

![](/example_images/rome_italy.png)
*Rome, Italy*


## About The Project

Beautimap is a project I created to generate desktop wallpapers. 
Maps can be customized with configs (which specify the map location and dimensions) and colorschemes.
The project is designed to be easy to understand and configure. 



## Installation

Beautimap is a single python file and a series of colorschemes. The script requires no installation, however to install dependencies, do the following:

1. (Optional) Create a virtual environment
   ```sh
   python3 -m venv ~/beautimap-venv
   source ~/beautimap-venv/bin/activate
   ```
2. Clone the repo
   ```sh
   git clone https://github.com/rdevans0/beautimap.git
   ```
2. Install the dependencies
   ```sh
   cd beautimap
   pip3 install -r requirements.txt
   ```

   
## Usage

The simplest way to use beautimap is to load one of the example configurations:
```sh
python3 beautimap.py --config example_configs/rome_italy.py --out rome_italy.png
```

The commands used in the example configs correspond to command line arguments, which can be viewed with
```sh
python3 beautimap.py --help
```

Commands on the command line can be used to override the config settings, for instance:
```sh
# Change colorscheme to light
python3 beautimap.py --config example_configs/rome_italy.py --color light

# Change distance to 2km for the image
python3 beautimap.py --config example_configs/rome_italy.py --dist 2000

# Replace the pallette color for buildings
python3 beautimap.py --config example_configs/rome_italy.py --replace_color building yellow

# Make image 1024x1024
python3 beautimap.py --config example_configs/rome_italy.py --size 1024 1024
```

New color schemes are easy to create by copying an existing color scheme in the ()[colorschemes] folder. 
The format is a python dict of `'object_type': 'color'`. 

More advanced configuration, such as changing color mappings of some objects, the ratios of road widths, and the groupings of roads must be changed by modifying [beautimap.py](beautimap.py).


## Examples
The [example_configs](example_configs) folder has a series of example configs. To regenerate any of these maps, run
```sh
python3 beautimap.py --config example_configs/<config>
```

The color scheme is listed with the images. 

![](/example_images/arc_de_triomphe_1024x1024.png)
*Arc de Triomphe, France (neon, with yellow buildings)*

![](/example_images/new_york_usa.png)
*New York, USA (simplified)*

![](/example_images/copenhagen_denmark.png)
*Copenhagen, Denmark (oblivion, with no buildings)*

![](/example_images/pearson_airport_canada.png)
*Pearson International Airport, Canada (light, with lighter buildings)*

![](/example_images/vancouver_canada_widescreen.png)
*Vancouver, Canada (neon)*

![](/example_images/rome_italy.png)
*Rome, Italy (oblivion)*


## Known Issues
The following are known issues that could be fixed with some small effort:

* The script attempts to generate exact resolutions, but this is difficult to do with matplotlib. If the script always fails to obtain the correct resolution, then try changing the dpi, or use a larger resolution and crop the image to size. 
* The script can currently only handle maps with a wide, or even aspect ratio.
* Some geometries are ignored, for instance some areas have water bodies that will not show
* Some roads are not detected correctly. This has to do with how osmnx queries data from a bounding box, and can usually be resolved by increasing overscan.


## License

Distributed under the GNU GPLv3 License. See `LICENSE` for more information.

