import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def line_plotter_2d(x, y, xvar, yvar, pdb_name, file_ending):
    """
    Plots a 2D dataset using matplotlib.
    
    Parameters:
    x: x-axis dataset (list, numpy array, or pandas Series)
    y: y-axis dataset (list, numpy array, or pandas Series/DataFrame)
    xvar: x-axis label (string)
    yvar: y-axis label (string)
    pdb_name: Prefix for the output filename
    file_ending: Suffix for the output filename
    
    Handles multiple y columns by adding a legend.
    """
    plt.figure(figsize=(6, 4))  # Optional: Define figure size

    if isinstance(y, pd.DataFrame):  
        # If multiple y-values are passed as a pd dataframe
        for col in y.columns:
            plt.plot(x, y[col], label=col)  # Plot each column separately
        plt.legend()  # Add a legend with column names
        
    else:
        #if y is list or 1d array
        plt.plot(x, y, label=yvar)

    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.title(f"{yvar} vs {xvar}")  # Optional: Add title
    plt.savefig(f"{pdb_name}_{file_ending}.png")
    plt.close()

def heatmap(matrix, x_var, y_var, heat_var, titel, file_suffix,  pdb_name, reporting_time, sparsity = 1, start_frame = 0):
    """
    Plots a heatmap of a 2D matrix
    matrix = 2d symmetric matrix
    x_var = name of the x variabel
    y_var = name of the y variabel
    heat_var = name of the heat variabel
    titel = string of titel
    file_suffix = ending of saved file name

    if x_var and y_var == "time (ps)" we do a check to see if we can convert to ns
    """
    #TODO get even numbers on the axis
    if start_frame == None:
        start_frame = 0

    # Generate exactly 10 tick positions
    n = matrix.shape[0]
    num_ticks = 10
    ticks = np.linspace(0, n - 1, num_ticks, dtype=int)  # Ensure valid indices
    tick_labels = (ticks * sparsity + start_frame) * reporting_time  # Scale labels by sparsity

    if max(tick_labels) > 1000 and x_var == y_var == "time (ps)":
        tick_labels = tick_labels /1000
        x_var = "time (ns)"
        y_var = "time (ns)"

    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(orientation="vertical", fraction=0.1, label= heat_var)

    # Set ticks and scaled labels
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(titel)
    plt.savefig(f"{pdb_name}_{file_suffix}.png")
    plt.close()
