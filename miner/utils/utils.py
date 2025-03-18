import numpy as np
import pandas as pd
import MDAnalysis as mda
import re
import os
import glob
import warnings

warnings.filterwarnings("ignore", module="MDAnalysis")

def numpy_to_blankspace_sep_str(np_array):
    # Convert each element to a string and join them with a space
    np_string = ' '.join(map(str, np_array))
    return np_string

def load_universe(structure_file, trajectory_file=None):
    if trajectory_file:
        universe = mda.Universe(structure_file, trajectory_file)
    else:
        universe = mda.Universe(structure_file)
    return universe

def get_frame_num(dir):
    """
    Reads a directory for files printed by run_caver_tempdir
    returns max, min, len of frame number
    """
    #find run length
    # Regular expression to match "frame_{i}.pdb"
    pattern = re.compile(r"frame_(\d+)\.pdb")

    # Extract frame numbers from filenames
    frame_numbers = [
        int(match.group(1))
        for filename in os.listdir(dir)
        if (match := pattern.match(filename))
    ]

    # Get the maximum frame number
    max_i = max(frame_numbers, default=None)  # Use None if no files are found
    min_i = min(frame_numbers, default=None)
    len_i = len(frame_numbers)

    return (max_i, min_i, len_i)

def read_caver_bottleneck_csv(csv_file):
    """
    This function parses the bottlenecks csv from caver manually and returns a well ordered df.
    """
    # Open the file and process manually
    with open(csv_file, "r") as f:
        lines = f.readlines()

    # Extract the actual header (skipping the first row containing information)
    header = lines[1].strip().split(",")[:9]  # First 9 fixed columns
    header.append("Bottleneck residues")  # Last column is variable-length

    # Process the rows
    data = []
    for line in lines[2:]:  # Skip first two rows
        parts = line.strip().split(",")  # Split by commas
        fixed_cols = parts[:9]  # First 9 columns are fixed
        residues = parts[9:]  # The remaining columns are residues
        fixed_cols.append(residues)  # Append residues as a list
        data.append(fixed_cols)

    # Create DataFrame
    bottleneck_df = pd.DataFrame(data, columns=header)

    #drop leading spaces in column names
    bottleneck_df.columns = bottleneck_df.columns.str.strip()

    #Drop useless tunnel identifier (numbers tunnels in each snapshot from 1 no matter cluster)
    bottleneck_df = bottleneck_df.drop(columns=["Tunnel"])

    # Convert numeric columns
    numeric_cols = ["Throughput", "Cost", "Bottleneck X", "Bottleneck Y", "Bottleneck Z", "Bottleneck R"]
    bottleneck_df[numeric_cols] = bottleneck_df[numeric_cols].astype(float)

    return bottleneck_df

def read_tunnel_coords(caver_out_dir):
    """
    This function reads tunnel coordinates from data/clusters_timeless in a caver_out_dir
    Returns a df ready to concat with bottlenecks.csv derived df
    """

    # Directory containing PDB files
    pdb_dir = f"{caver_out_dir}/data/clusters_timeless/*.pdb"

    # List to store data
    data = []

    # Loop over PDB files
    for pdb_file in glob.glob(pdb_dir):
        # Load the PDB file
        u = mda.Universe(pdb_file)
        
        # Extract coordinates
        coords = u.atoms.positions  # Shape: (N, 3)

        #optional extraction of point radii
        point_radii = u.atoms.tempfactors  # Returns a NumPy array
        
        # Store data, we have to add more than just coords for rows to be generated properlly
        data.append({
            "filename": pdb_file, 
            "num_atoms": coords.shape[0], 
            "coordinates": coords,
            "tunnel_point_radii": point_radii.round(decimals=2)
        })

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)

    #drop redundant columns
    df = df.drop(columns = ["filename", "num_atoms"])
    return df



