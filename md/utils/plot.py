import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from utils import utils
from scipy.ndimage import minimum_filter
import openmm.unit as unit
import matplotlib.patheffects as patheffects

def line_plotter_2d(x, y, x_var, y_var, basename, plot_type, annotate_minima= False):
    """
    Plots a 2D dataset using matplotlib.
    
    Parameters:
    x: x-axis dataset (list, numpy array, or pandas Series)
    y: y-axis dataset (list, numpy array, or pandas Series/DataFrame)
    xvar: x-axis label (string)
    yvar: y-axis label (string)
    basename: Prefix for the output filename
    plot_type = ending of saved file name, corresponding to the type of plot, eg RMSD
    
    Handles multiple y columns by adding a legend.
    if x_var == "Time (ps)" we do a check to see if we can convert to ns
    """
    plt.figure(figsize=(6, 4))  # Optional: Define figure size

    if max(x) > 1000 and x_var == "Time (ps)":
        x = x /1000
        x_var = "Time (ns)"

    if isinstance(y, pd.DataFrame):  
        # If multiple y-values are passed as a pd dataframe
        for col in y.columns:
            plt.plot(x, y[col], label=col.replace("resname ", ""))  # Plot each column separately
        plt.legend()  # Add a legend with column names
        
    else:
        #if y is list or 1d array
        plt.plot(x, y, label=y_var)

    plt.xlim(min(x), max(x))  # Set x limits to avoid overhang
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f"{y_var} vs {x_var}") 
    plt.savefig(f"{basename}_{plot_type}.png")
    plt.close()

def time_heatmap(matrix, x_var, y_var, heat_var, titel, plot_type,  basename, reporting_time, sparsity = 1, start_frame = 0):
    """
    Plots a heatmap of a 2D matrix infering x and y axis ticks from simulation time
    matrix = 2d symmetric matrix
    x_var = name of the x variabel
    y_var = name of the y variabel
    heat_var = name of the heat variabel
    titel = string of titel
    basename: Prefix for the output filename
    plot_type = ending of saved file name, corresponding to the type of plot, eg RMSD

    if x_var and y_var == "Time (ps)" we do a check to see if we can convert to ns
    """
    if start_frame == None:
        start_frame = 0

    # Calculate min and max for axis labels
    n = matrix.shape[0]
    axis_min = (start_frame) * reporting_time
    axis_max = ((n - 1) * sparsity + start_frame) * reporting_time

    if axis_max > 1000 and x_var == y_var == "Time (ps)":
        axis_min /= 1000
        axis_max /= 1000
        x_var = "Time (ns)"
        y_var = "Time (ns)"

    heatmap(matrix, x_axis_min_max=(axis_min, axis_max), y_axis_min_max=(axis_min, axis_max), x_var=x_var,
        y_var=y_var, heat_var=heat_var, titel=titel, basename=basename, plot_type=plot_type)

def nice_ticks(min_max_tuple, axis_size):
    """
    Returns tick positions and labels using matplotlib's MaxNLocator for nice ticks.
    
    min_max_tuple= tuple of (start, end) for the axis range
    axis_size= number of data points on the axis to be ticked (used for tick positions)

    Returns: tuple of (indices, ticks)
        indices: np array of tick positions
        ticks: np array of tick labels
    """
    
    locator = ticker.MaxNLocator(nbins=12, steps=[1, 2, 2.5, 3, 5, 10], prune=None, min_n_ticks=5)
    start, end = min_max_tuple

    # Get nice tick locations within the data range
    ticks = locator.tick_values(start, end)

    # If all tick labels are integer-valued floats, convert to int
    if np.all(np.isclose(ticks, ticks.astype(int))):
        ticks = ticks.astype(int)

    # Generate tick positions based on the axis size
    indices = np.clip(np.round((ticks - start) / (end - start) * (axis_size - 1)).astype(int), 0, axis_size - 1)

    return indices, ticks

def heatmap(matrix, x_axis_min_max, y_axis_min_max, x_var, y_var, heat_var, titel, basename, plot_type, annotate_minima=False):
    """
    Plots a heatmap of a 2D matrix with user given axis ticks
    matrix = 2d symmetric matrix
    x_axis_min_max = tuple of (start, end) for the x axis range
    y_axis_min_max = tuple of (start, end) for the y axis range

    x_var = name of the x variabel
    y_var = name of the y variabel
    heat_var = name of the heat variabel

    titel = string of titel
    basename: Prefix for the output filename
    plot_type = ending of saved file name, corresponding to the type of plot, eg RMSD
    """

    #make the colorbar a separate subplot to avoid overlapping with the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(cax, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label=heat_var)

    # Set ticks and scaled labels using nice_ticks
    x_ticks, x_tick_labels = nice_ticks(x_axis_min_max, matrix.shape[1])
    y_ticks, y_tick_labels = nice_ticks(y_axis_min_max, matrix.shape[0])
    plt.xticks(x_ticks, x_tick_labels)
    plt.yticks(y_ticks, y_tick_labels)

    #Add annotations for minima
    if annotate_minima:
        # Detect local minima (including plateaus)
        local_min = np.isclose(minimum_filter(matrix, size=3, mode="constant", cval=np.inf), matrix)

        # Create a 3x3 footprint with the center excluded to detect plateus
        footprint = np.ones((3, 3), dtype=bool)
        footprint[1, 1] = False
        strict_min = (matrix < minimum_filter(matrix, footprint=footprint, mode='constant', cval=np.inf))

        # Exclude plateaus: only keep points strictly less than all neighbors
        minima_mask = local_min & strict_min
        minima_coords = np.argwhere(minima_mask)

        # Add annotations
        for y, x in minima_coords:
            plt.plot(x, y, 'ro')  # Red circle at each minimum
            
            #change from the absolute index to the x and y axis values
            x2 = np.linspace(x_axis_min_max[0], x_axis_min_max[1], matrix.shape[1])[x]
            y2 = np.linspace(y_axis_min_max[0], y_axis_min_max[1], matrix.shape[0])[y]

            plt.text(x, y, f"{x2:.2f},{y2:.2f} {matrix[y, x]:.2f} kj/mol", color='white', fontsize=8, ha='center', va='bottom')

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(titel)
    plt.savefig(f"{basename}_{plot_type}.png")
    plt.close()

def plot_3d_scatter(x, y, z, heat, x_var, y_var, z_var, heat_var, basename, plot_type, annotate_minima=False):
    """
    Plots a 3D scatter plot of x, y, z data colored by heat using matplotlib.
    
    Parameters:
    x: x-axis dataset (list or numpy array)
    y: y-axis dataset (list or numpy array)
    z: z-axis dataset (list or numpy array)
    heat: heat dataset (list or numpy array) for coloring the points

    x_var: x-axis label (string)
    y_var: y-axis label (string)
    z_var: z-axis label (string)
    heat_var: label for the heat variable (string)

    basename: Prefix for the output filename
    plot_type = ending of saved file name, corresponding to the type of plot, eg RMSD
    """
    
    # Flatten the grid for scatter plot
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    fig = plt.figure(figsize=(8, 7))

    ax = fig.add_axes([0, 0.05, 0.80, 0.80], projection='3d')  # [left, bottom, width, height]
    p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=heat.flatten(), cmap='viridis', s=0.01)

    cax = fig.add_axes([0.89, 0.05, 0.03, 0.80])  # Colorbar: [left, bottom, width, height]
    fig.colorbar(p, cax=cax, label='Energy')

    # Set ticks and labels for x, y, z axes
    x_ticks, x_tick_labels = nice_ticks((x[0], x[-1]), len(x))
    y_ticks, y_tick_labels = nice_ticks((y[0], y[-1]), len(y))
    z_ticks, z_tick_labels = nice_ticks((z[0], z[-1]), len(z))
    ax.set_xticks(x_tick_labels)
    ax.set_xticklabels(x_tick_labels, rotation=0, ha='center', va = "bottom")
    ax.set_yticks(y_tick_labels)
    ax.set_yticklabels(y_tick_labels, rotation=-45, ha='center', va = "center")
    ax.set_zticks(z_tick_labels, z_tick_labels, rotation=0)

    #set the axis limits to avoid overhang
    ax.set(xlim = (x[0], x[-1]), ylim = (y[0], y[-1]), zlim = (z[0], z[-1]))

    # Add annotations for minima in 3D scatterplot
    if annotate_minima:
        # Detect local minima in 3D
        local_min = np.isclose(minimum_filter(heat, size=3, mode="constant", cval=np.inf), heat)

        # Create a 3x3x3 footprint with the center excluded to detect plateaus
        footprint = np.ones((3, 3, 3), dtype=bool)
        footprint[1, 1, 1] = False
        strict_min = (heat < minimum_filter(heat, footprint=footprint, mode='constant', cval=np.inf))

        # Exclude plateaus: only keep points strictly less than all neighbors
        minima_mask = local_min & strict_min
        minima_coords = np.argwhere(minima_mask)

        # Add annotations
        for z_idx, y_idx, x_idx in minima_coords:
            x_val = x[z_idx]
            y_val = y[y_idx]
            z_val = z[x_idx]
            ax.scatter(x_val, y_val, z_val, color='red', s=20)
            # Plot the text after all scatter points, with a higher zorder so it's in front
            ax.text(
                x_val, y_val, z_val,
                f"{x_val:.2f},{y_val:.2f},{z_val:.2f}\n{heat[z_idx, y_idx, x_idx]:.2f} kj/mol",
                color='black', fontsize=8, ha='center', va='bottom', zorder=50,
            )

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)
    plt.savefig(f"{basename}_{plot_type}.png")
    plt.close()


def metadynamics_plotter(pes, atom_indices, colvar_parameters, basename):
    """
    Plots the PES from the metadyanmics simulation in 1 or 2 D
    """
    energy_label = "kJ/mol"

    atom_indices_css = [utils.list_to_css(lst) for lst in atom_indices]  # Convert atom indices to CSS format

    if len(atom_indices) == 1:
        # If 1 colvar, pes should be a 1D array
        if pes.ndim != 1:
            raise ValueError("For 1 colvars, pes must be a 1D numpy array.")

        #generate x data
        x_values = np.linspace(colvar_parameters[0][0], colvar_parameters[0][1], len(pes))
        x_var = f"{atom_indices_css[0]} ({utils.metadynamics_unit_finder(atom_indices[0])})"

        line_plotter_2d(x_values, pes, x_var = x_var, y_var = energy_label, basename = basename, plot_type= "metadynamics_pes")

    elif len(atom_indices) == 2:
        # If 2 colvars, pes should be a 2D array
        if pes.ndim != 2:
            raise ValueError("For 2 colvars, pes must be a 2D numpy array.")
        
        #generate units
        x_var = f"{atom_indices_css[0]} ({utils.metadynamics_unit_finder(atom_indices[0])})"
        y_var = f"{atom_indices_css[1]} ({utils.metadynamics_unit_finder(atom_indices[1])})"

        titel = "temp"  # TODO
        heatmap(pes/ unit.kilojoule_per_mole, x_axis_min_max= colvar_parameters[0][0:2], y_axis_min_max= colvar_parameters[1][0:2], 
                x_var=x_var, y_var=y_var, heat_var=energy_label, titel=titel, basename=basename, 
                plot_type="metadynamics_pes", annotate_minima=True)

    elif len(atom_indices) == 3:
        # If 3 colvars, pes should be a 3D array
        if pes.ndim != 3:
            raise ValueError("For 3 colvars, pes must be a 3D numpy array.")
        
        #prepare for 3D plot
        x_ticks = np.linspace(colvar_parameters[0][0], colvar_parameters[0][1], pes.shape[0])
        x_var = f"{atom_indices[0]} ({utils.metadynamics_unit_finder(atom_indices[0])})"

        y_ticks = np.linspace(colvar_parameters[1][0], colvar_parameters[1][1], pes.shape[1])
        y_var = f"{atom_indices[1]} ({utils.metadynamics_unit_finder(atom_indices[1])})"

        z_ticks = np.linspace(colvar_parameters[2][0], colvar_parameters[2][1], pes.shape[2])
        z_var = f"{atom_indices[2]} ({utils.metadynamics_unit_finder(atom_indices[2])})"
        pes = pes / unit.kilojoule_per_mole

        plot_3d_scatter(x_ticks, y_ticks, z_ticks, pes, x_var, y_var, z_var, energy_label, basename, plot_type="metadynamics_pes", annotate_minima=True)     
