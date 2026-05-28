import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
from whatcat.md.utils import utils
from scipy.ndimage import minimum_filter

def line_plotter_2d(x, y, x_var, y_var, basename=None, plot_type=None,  plot_format="png", 
                    titel = None, annotate_minima= False, force_zero_start_x = False, 
                    force_zero_start_y = False, save_fig=True, histogram = False,):
    """
    Plots a 2D dataset using matplotlib. Optionally adds a histogram of Y values to the right.
    
    Parameters:
        x: x-axis dataset (list, numpy array, or pandas Series)
        y: y-axis dataset (list, numpy array, or pandas Series/DataFrame)
        xvar: str
            x-axis label
        yvar: str
            y-axis label

        basename: str
            Prefix for the output filename, only needed when save_fig= True
        plot_type: str
            ending of saved file name, corresponding to the type of plot, eg RMSD
        plot_format: str
            What matplotlib image format do you want plots to be saved as
        titel: str
            If set to a string, it will be used as the plot title. Otherwise, the title will be set to "{y_var} vs {x_var}".
        
        annotate_minima: bool
            If True, annotate local minima in the plot
        force_zero_start_x: bool
            If True, set x-axis to start at 0, otherwise use min(x) as start.
        force_zero_start_y: bool
            If True, set y-axis to start at 0, otherwise use min(y) as start.
        save_fig: bool
            If True, save the plot as a PNG file with the basename and plot_type. Otherwise return plt
        histogram: bool
            If True, plot a histogram of Y values to the right of the line plot.
        
        Handles multiple y columns by adding a legend.
        if x_var == "Time (ps)" we do a check to see if we can convert to ns
    """

    if histogram:
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.25)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    else:
        plt.figure(figsize=(6, 4))
        ax_main = plt.gca()

    if max(x) > 1000 and x_var == "Time (ps)":
        x = np.array(x) #Turn to array to make sure broadcasting division works
        x = x /1000
        x_var = "Time (ns)"

    if isinstance(y, pd.DataFrame):  
        for col in y.columns:
            ax_main.plot(x, y[col], 
                         label=utils.strip_mda_selection(selection_string=col, spaces_to_underscores=False))
        #Make sure legend is at the bottom right
        ax_main.legend(loc="lower right")
    else:
        ax_main.plot(x, y, label=y_var)

    # Handle x-axis limits
    if force_zero_start_x:
        ax_main.set_xlim(0, np.max(x))
    else:
        ax_main.set_xlim(np.min(x), np.max(x))

    # Handle y-axis limits
    if isinstance(y, pd.DataFrame):
        y_min = y.min().min()
        y_max = y.max().max()
    else:
        y_min = np.min(y)
        y_max = np.max(y)

    if force_zero_start_y:
        ax_main.set_ylim(0, y_max)
    else:
        ax_main.set_ylim(y_min, y_max)

    # Annotate local minima if requested
    if annotate_minima:
        if type(y) == pd.DataFrame:
            for col_name in y.columns:
                y_col = y[col_name].to_numpy()
                local_min = np.isclose(minimum_filter(y_col, size=3, mode="constant", cval=np.inf), y_col)
                footprint = np.ones((3,), dtype=bool)
                footprint[1] = False
                strict_min = (y_col < minimum_filter(y_col, footprint=footprint, mode='constant', cval=np.inf))
                minima_mask = local_min & strict_min
                minima_indices = np.where(minima_mask)[0]
                for idx in minima_indices:
                    if force_zero_start_y == False:
                        ax_main.plot(x[idx], y_col[idx], 'ro')
                        ax_main.text(x[idx], y_col[idx], f"{x[idx]:.2f}:{y_col[idx]:.2f} {y_var}", color='black', fontsize=8, ha='center', va='bottom')
                    elif force_zero_start_y == True and y_col[idx] > 0:
                        ax_main.plot(x[idx], y_col[idx], 'ro')
                        ax_main.text(x[idx], y_col[idx], f"{x[idx]:.2f}:{y_col[idx]:.2f} {y_var}", color='black', fontsize=8, ha='center', va='bottom')
        else:
            local_min = np.isclose(minimum_filter(y, size=3, mode="constant", cval=np.inf), y)
            footprint = np.ones((3,), dtype=bool)
            footprint[1] = False
            strict_min = (y < minimum_filter(y, footprint=footprint, mode='constant', cval=np.inf))
            minima_mask = local_min & strict_min
            minima_indices = np.where(minima_mask)[0]
            for idx in minima_indices:
                if force_zero_start_y == False:
                    ax_main.plot(x[idx], y[idx], 'ro')
                    ax_main.text(x[idx], y[idx], f"{x[idx]:.2f}:{y[idx]:.2f} {y_var}", color='black', fontsize=8, ha='center', va='bottom')
                elif force_zero_start_y == True and y[idx] > 0:
                    ax_main.plot(x[idx], y[idx], 'ro')
                    ax_main.text(x[idx], y[idx], f"{x[idx]:.2f}:{y[idx]:.2f} {y_var}", color='black', fontsize=8, ha='center', va='bottom')

    ax_main.set_xlabel(x_var)
    ax_main.set_ylabel(y_var)

    if titel is None:
        ax_main.set_title(f"{y_var} vs {x_var}")
    else:
        ax_main.set_title(titel)

    # Add histogram if requested
    if histogram:
        ax_hist.hist(y, bins=30, orientation='horizontal', alpha=0.7)
        ax_hist.set_title("Histogram", fontsize=10)
        ax_hist.grid(False)

    if save_fig:
        plt.savefig(f"{basename}_{plot_type}.{plot_format}")
        plt.close()
    else:
        return plt  # Return the plot object for further manipulation or display
    
def scatter_plot_2d(x, y, x_var, y_var, basename=None, plot_type=None, plot_format="png", 
                    titel=None, labels=None, save_fig=True, histogram=False):
    """
    Plots a 2D dataset using matplotlib as a scatter plot. Optionally adds a histogram of Y values to the right.

    Parameters:
        x: x-axis dataset (list, numpy array, or pandas Series)
        y: y-axis dataset (list, numpy array, or pandas Series/DataFrame)
        xvar: str
            x-axis label
        yvar: str
            y-axis label

        basename: str
            Prefix for the output filename, only needed when save_fig= True
        plot_type: str
            ending of saved file name, corresponding to the type of plot, eg RMSD
        plot_format: str
            What matplotlib image format do you want plots to be saved as
        titel: str
            If set to a string, it will be used as the plot title. Otherwise, the title will be set to "{y_var} vs {x_var}".
        
        labels: None or dataset (list, numpy array, pd.Dataframe)
            dataset with which to label each point on the scatterplot with.
            Must be a dataframe if y is a dataframe
        save_fig: bool
            If True, save the plot as a PNG file with the basename and plot_type. Otherwise return plt
        histogram: bool
            If True, plot a histogram of Y values to the right of the scatter plot.
        
        Handles multiple y columns by adding a legend.
        if x_var == "Time (ps)" we do a check to see if we can convert to ns
    """

    if histogram:
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.25)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    else:
        plt.figure(figsize=(6, 4))
        ax_main = plt.gca()

    if max(x) > 1000 and x_var == "Time (ps)":
        x = np.array(x)  # Turn to array to make sure broadcasting division works
        x = x / 1000
        x_var = "Time (ns)"

    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            ax_main.scatter(x, y[col], 
                            label=utils.strip_mda_selection(selection_string=col, spaces_to_underscores=False))
        #Make sure legend is at the bottom right
        ax_main.legend(loc="lower right")
    else:
        ax_main.scatter(x, y, label=y_var)

    # Handle x-axis limits
    ax_main.set_xlim(np.min(x), np.max(x))

    ax_main.set_xlabel(x_var)
    ax_main.set_ylabel(y_var)

    if titel is None:
        ax_main.set_title(f"{y_var} vs {x_var}")
    else:
        ax_main.set_title(titel)

    # Add histogram if requested
    if histogram:
        ax_hist.hist(y, bins=30, orientation='horizontal', alpha=0.7)
        ax_hist.set_title("Histogram", fontsize=10)
        ax_hist.grid(False)

    if labels is not None:
        if isinstance(y, pd.DataFrame):
            for col in y.columns:
                for i,label in enumerate(labels[col]):
                    ax_main.text(x[i], y[col].iloc[i], str(label), fontsize=8, ha='center', va='bottom')
        else:
            for i, label in enumerate(labels):
                ax_main.text(x[i], y[i], str(label), fontsize=8, ha='center', va='bottom')

    if save_fig:
        plt.savefig(f"{basename}_{plot_type}.{plot_format}")
        plt.close()
    else:
        return plt  # Return the plot object for further manipulation or display

def time_heatmap(matrix, x_var, y_var, heat_var, titel, plot_type,  basename, 
                 reporting_time, sparsity = 1, start_frame = 0, plot_format = "png"):
    """
    Plots a heatmap of a 2D matrix infering x and y axis ticks from simulation time

    Parameters:
        matrix: array
            2d symmetric matrix
        x_var: str
            name of the x variabel
        y_var: str
            name of the y variabel
        heat_var: str
            name of the heat variabel
        titel: str
            titel
        basename: str
            Prefix for the output filename
        plot_type: str
            ending of saved file name, corresponding to the type of plot, eg RMSD
        plot_format: str
            What matplotlib image format do you want plots to be saved as

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
        y_var=y_var, heat_var=heat_var, titel=titel, basename=basename, plot_type=plot_type, plot_format=plot_format)

def nice_ticks(min_max_tuple, axis_size):
    """
    Returns tick positions and labels using matplotlib's MaxNLocator for nice ticks.

    Parameters:
        min_max_tuple: tuple
            (start, end) for the axis range
        axis_size: int
            number of data points on the axis to be ticked (used for tick positions)

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
    else:
        ticks = np.round(ticks, 2)  # Round to 2 decimal places for better readability

    # Filter ticks strictly within [start, end]
    ticks_in = ticks[(ticks >= start) & (ticks <= end)]
    
    # Generate tick positions based on the axis size
    indices = np.clip(np.round((ticks_in - start) / (end - start) * (axis_size - 1)).astype(int), 0, axis_size - 1)
    ticks = ticks_in

    return indices, ticks

def heatmap(matrix, x_axis_min_max, y_axis_min_max, x_var, y_var, heat_var, titel, 
            basename=None, plot_type=None, plot_format = "png", annotate_minima=False, 
            save_fig=True, aspect = "equal", contours=False):
    """
    Plots a heatmap of a 2D matrix with user given axis ticks

    Parameters:
        matrix: array
            2d symmetric matrix
        x_axis_min_max: tuple 
            (start, end) for the x axis range
        y_axis_min_max: tuple
            (start, end) for the y axis range

        x_var: str 
            name of the x variabel
        y_var: str 
            name of the y variabel
        heat_var: str 
            name of the heat variabel

        titel: str
            titel
        basename: str
            Prefix for the output filename
        plot_type: str 
            ending of saved file name, corresponding to the type of plot, eg RMSD
        plot_format: str
            What matplotlib image format do you want plots to be saved as
        titel: str 
            If set to a string, it will be used as the plot title. Otherwise, the title will be set to "{y_var} vs {x_var}".

        annotate_minima: bool
            If True, annotate local minima in the heatmap
        save_fig: bool
            If True, save the plot as a PNG file with the basename and plot_type. Otherwise return plt
        aspect: bool
            If set to "auto", the heatmap will always be rectangular to fill up the plot, default equal each pixel will have a 1:1 aspect.
        Contours: bool
            If one wants height curves on the plot.
    """

    #make the colorbar a separate subplot to avoid overlapping with the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap="viridis", aspect = aspect) #aspect=auto means we always make the plot max size by employing rectangular pixels
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

    #Add in height lines if desired
    if contours:
        _, levels = nice_ticks((np.min(matrix), np.max(matrix)), matrix.shape[0])
        contours = plt.contour(np.linspace(-0.5, matrix.shape[1]-0.5, matrix.shape[1]), 
                               np.linspace(-0.5, matrix.shape[0]-0.5, matrix.shape[0]), 
                                matrix, levels=levels, colors = "Red", linestyles = "--")
        plt.clabel(contours, inline=True, fontsize=8)

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(titel)

    if save_fig:
        plt.savefig(f"{basename}_{plot_type}.{plot_format}")
        plt.close()
    else:
        return plt  # Return the plot object for further manipulation or display

def plot_3d_scatter(x, y, z, heat, x_var, y_var, z_var, heat_var, basename = None, 
                    plot_type = None, plot_format = "png", titel=None, annotate_minima=False, 
                    save_fig = True):
    """
    Plots a 3D scatter plot of x, y, z data colored by heat using matplotlib.
    
    Parameters:
        x: list or np.ndarray
            x-axis dataset
        y: list or np.ndarray
            y-axis dataset
        z: list or np.ndarray
            z-axis dataset
        heat: np.ndarray zyx indexed
            3D heat dataset for coloring the points

        x_var: str
            x-axis label 
        y_var: str
            y-axis label 
        z_var: str
            z-axis label 
        heat_var: str
            label for the heat variable 

        basename: str
            Prefix for the output filename
        plot_type: str
            ending of saved file name, corresponding to the type of plot, eg RMSD
        plot_format: str
            What matplotlib image format do you want plots to be saved as
        annotate_minima: bool 
            If True, annotate local minima in the plot
        save_fig: bool
            If True, save the plot as a PNG file with the basename and plot_type. Otherwise return plt
    """
    
    # Flatten the grid for scatter plot
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    fig = plt.figure(figsize=(8, 7))

    ax = fig.add_axes([0, 0.05, 0.80, 0.80], projection='3d')  # [left, bottom, width, height]
    p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=heat.flatten(), cmap='viridis', s=0.01)

    cax = fig.add_axes([0.89, 0.05, 0.03, 0.80])  # Colorbar: [left, bottom, width, height]
    fig.colorbar(p, cax=cax, label=f"{heat_var}")

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
    ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=(z[0], z[-1]))

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
            x_val = x[x_idx]
            y_val = y[y_idx]
            z_val = z[z_idx]

            #print(f"Minimum at ({x_val}, {y_val}, {z_val}) with heat value {heat[x_idx, y_idx, x_idx]:.2f}")

            ax.scatter(x_val, y_val, z_val, color='red', s=20)
            # Plot the text after all scatter points, with a higher zorder so it's in front
            ax.text(x_val, y_val, z_val,
                f"{x_val:.2f},{y_val:.2f},{z_val:.2f}\n{heat[z_idx, y_idx, x_idx]:.2f} {heat_var}",
                color='black', fontsize=8, ha='center', va='bottom', zorder=50,
            )

    ax.set_xlabel(f"x: {x_var}")
    ax.set_ylabel(f"y: {y_var}")
    ax.set_zlabel(f"z: {z_var}")
    
    title_fontsize = plt.rcParams.get("axes.titlesize", 24)
    if titel is None:
        plt.suptitle(plot_type, va="top", ha='center', fontsize="large")
    else:
        plt.suptitle(titel, va="top", ha='center', fontsize="large")
    
    if save_fig:
        plt.savefig(f"{basename}_{plot_type}.{plot_format}")
        plt.close()
    else:
        return plt  # Return the plot object for further manipulation or display


def metadynamics_plotter(pes, atom_indices, colvar_parameters, basename, plot_format="png"):
    """
    Plots the PES from the metadyanmics simulation in 1, 2 or 3D

    Parameters:
        pes: array
            Potential Energy Surface (PES) data
        atom_indices: list
            atom indices for the colvars, e.g. [[0], [1, 2], [3, 4, 5]]
        colvar_parameters: list
            List of tuples with min and max values for each colvar, e.g. [(0, 10), (0, 10), (0, 10)]
        basename: str
            Prefix for the output filename
        plot_format: str
            What matplotlib image format do you want plots to be saved as.
            3D plots are always saved as png since vector formated files get so large and slow with many points.
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

        line_plotter_2d(x_values, pes, x_var = x_var, y_var = energy_label, basename = basename, plot_type= "metadynamics_pes", annotate_minima=True)

    elif len(atom_indices) == 2:
        # If 2 colvars, pes should be a 2D array
        if pes.ndim != 2:
            raise ValueError("For 2 colvars, pes must be a 2D numpy array.")
        
        #generate units
        x_var = f"colvar 1: {atom_indices_css[0]} ({utils.metadynamics_unit_finder(atom_indices[0])})"
        y_var = f"colvar 2: {atom_indices_css[1]} ({utils.metadynamics_unit_finder(atom_indices[1])})"

        titel = "Metadynamics PES (kJ/mol)"
        heatmap(pes, x_axis_min_max= colvar_parameters[0][0:2], y_axis_min_max= colvar_parameters[1][0:2], 
                x_var=x_var, y_var=y_var, heat_var=energy_label, titel=titel, basename=basename, 
                plot_type="metadynamics_pes", annotate_minima=True, aspect="auto", contours=True)

    elif len(atom_indices) == 3:
        # If 3 colvars, pes should be a 3D array
        if pes.ndim != 3:
            raise ValueError("For 3 colvars, pes must be a 3D numpy array.")

        #prepare for 3D plot zyx indexed
        x_ticks = np.linspace(colvar_parameters[0][0], colvar_parameters[0][1], pes.shape[2])
        x_var = f"colvar 1: {atom_indices[0]} ({utils.metadynamics_unit_finder(atom_indices[0])})"

        y_ticks = np.linspace(colvar_parameters[1][0], colvar_parameters[1][1], pes.shape[1])
        y_var = f"colvar 2: {atom_indices[1]} ({utils.metadynamics_unit_finder(atom_indices[1])})"

        z_ticks = np.linspace(colvar_parameters[2][0], colvar_parameters[2][1], pes.shape[0])
        z_var = f"colvar 3: {atom_indices[2]} ({utils.metadynamics_unit_finder(atom_indices[2])})"

        plot_3d_scatter(x_ticks, y_ticks, z_ticks, pes, x_var, y_var, z_var, 
                        energy_label, basename, plot_type="metadynamics_pes", 
                        plot_format="png", annotate_minima=True)
