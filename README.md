# WhatCat
This package provides the WhatCat software which aims to aid physics based investigation into proteins and enzymes. The codebase is availible under the MIT license. A detailed description of the software is published on Chemrxiv (URL).

## Installation
To install whatcat
```
git clone
cd to dir

conda create -n whatcat -f enviroment.yml
or on a HPC cluster where compilers are not included in the active env
conda create -n whatcat -f enviroment.yml cgal libgcc

conda activate whatcat
pip install -e .


To further install the HIP platform for AMD GPU:s 
conda remove openmm --force
pip install openmm[hip6]
```

If you get this error when running md:
openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)

you might have to force the cuda version by adding the following to the conda create
```
cuda-version=12.6
```

Check the highest supported cuda version with nvidia-smi command

# Usage
Docs will be written and hosted after the codebase has stabilized and more of the projected features have been implemented. In the meanwhile here follows a short description of the package.

WhatCat can be used either as a python package or as via a CLI. If you are intrested in the API, 

### Python API
Sadlly you will have to wait for the proper docs to become availible for detailed guidance. However, the source code is extensivelly commented and each function has a docstring with input and output along with expected type and format. I would recommend reading through the main() of the CLI implemented modules as this contains a reference implementation which is intended to cover most usecases. 

### CLI
To aid use by non-expert biochemists and similair, the functionallity of WhatCat is also packaged as commandline functions which run a one-size-fits-all version of the functionallity included herein. Extensive commandline parameters allows for quite a lot of customization of what you want WhatCat to do for you.

The entrypoints for WhatCat are currentlly "whatcat-md" and "whatcat-efield".

whatcat-md does, as the name suggests, run MD and MTD simulations as well as analysis of the resulting trajectory. Input files are sanitized using PDBfixer allowing uncurated files from the PDB to be used in most cases. Parametrization of any small-molecule ligands is also done automatically. Metal-ligand interaction parametrization will be included in a future version. 

The key input for whatcat-md is a pdb file containing the protein (and NO small molecules), as well as sdf files of any small molecules with coordinates such that both files together make a valid protein-ligand complex. Such a SDF file can be attained either by selecting and saving the ligand from a protein-ligand complex in ChimeraX or Pymol or by docking a molecule into a protein using autodock vina which gives compatible sdf files as output.

Example commandline input migt look like

```
whatcat-md 5tbm.pdb lig.sdf -t 100 -rt 100
```

Which would run MD using default settings for 100 ns dumping frames to a dcd file every 100 ps. Then analysis would follow on the ligand since nothing else was specified. The analysis data would then be stored as csv files and plotted using matplotlib. Efield analysis would have been run if a vector selection was provided. For more information run:

```
whatcat-md --help
```

whatcat-efield only runs efield analysis on already generated trajectories. For more information see:
```
whatcat-efield --help
```