import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def line_plotter_2d(x, y, x_var, y_var, basename, plot_type):
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

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f"{y_var} vs {x_var}")  # Optional: Add title
    plt.savefig(f"{basename}_{plot_type}.png")
    plt.close()

def heatmap(matrix, x_var, y_var, heat_var, titel, plot_type,  basename, reporting_time, sparsity = 1, start_frame = 0):
    """
    Plots a heatmap of a 2D matrix
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

    # Generate exactly 10 tick positions
    n = matrix.shape[0]
    num_ticks = 10
    ticks = np.linspace(0, n - 1, num_ticks, dtype=int)  # Ensure valid indices
    tick_labels = (ticks * sparsity + start_frame) * reporting_time  # Scale labels by sparsity

    if max(tick_labels) > 1000 and x_var == y_var == "Time (ps)":
        tick_labels = tick_labels /1000
        x_var = "Time (ns)"
        y_var = "Time (ns)"

    #make the colorbar a separate subplot to avoid overlapping with the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(cax, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label=heat_var)

    # Set ticks and scaled labels
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(titel)
    plt.savefig(f"{basename}_{plot_type}.png")
    plt.close()
