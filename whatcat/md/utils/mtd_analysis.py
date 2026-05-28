import numpy as np
import pandas as pd
import os
from whatcat.md.utils import plot

class Whatcat_mtd_analysis:
    def __init__(self, filepath, colvar_parameters, name):
        self.filepath = filepath
        self.colvar_parameters = colvar_parameters[::-1]  # Reverse to matc zyx axes order since user input is xyz
        self.pes = self.load_mtd_data(filepath)
        self.ndim = self.pes.ndim
        self.axes = self.generate_axes_from_colvar(self.colvar_parameters, self.pes.shape) #zyx indexed
        self.xyz_axes = self.axes[::-1]  # Store a copy of the axes in xyz order for user reference
        self.pdf = None  # Initialize pdf as None
        self.name = name  # Store the name of the analysis for better identification

    def load_mtd_data(self, filepath):
        "loads the metadynamics data from a CSV (1 or 2D) or NPY (3D) file"
        ext = os.path.splitext(filepath)[1]
        if ext == ".csv":
            pes = pd.read_csv(filepath, header=0, index_col=0).values
        elif ext == ".npy":
            pes = np.load(filepath)
        else:
            raise ValueError("Unsupported file type. Use .csv or .npy")
        return pes

    def check_axes_in_bounds(self, axis_vals):
        "Utility function to check if the provided axis values are within the bounds defined in colvar_parameters."
        for i, val in enumerate(axis_vals):
            if val is not None:
                minv, maxv = self.colvar_parameters[i]
                if not (minv <= val <= maxv):
                    raise ValueError(f"Axis {i}: value {val} is out of bounds: {minv} < val < {maxv}")

    def generate_axes_from_colvar(self, colvar_parameters, data_shape):
        "Generate axes based on the colvar parameters and data shape."
        "uses zyx indexing"
        return [np.linspace(colvar_parameters[i][0], colvar_parameters[i][1], data_shape[i]) for i in range(len(data_shape))]

    def _slice_nd(self, axis_vals, xyz = False):
        """ 
        Utility function for slice_out_mtd_graph

        Parameters
            axis_vals: list
                The values for each axis to slice at, use None for the axis to plot along that axis
                [30, None, None] would slice at 30 on the x axis and plot a heatmap for y and z
            xyz: bool
                Wheteher axis_vals is xyz indexed or not
        
        Returns
            Nonecount: int
            sliced_pes: float or np.array
            colvar_parametes_sliced: list of tuples
                The minmax values of each axis in xyz order with only the unsliced axes remaining
        """
        ndim = self.pes.ndim
        none_count = axis_vals.count(None)

        if xyz:
            axis_vals = axis_vals[::-1]

        if none_count == ndim:
            print("No axis specified for slicing, no slicing will be done")
            return

        # Check if all axis values are provided, in that case print the energy at that point
        if all(v is not None for v in axis_vals):
            idx = tuple((np.abs(self.axes[i] - axis_vals[i])).argmin() for i in range(ndim))
            axis_values = [self.axes[i][idx[i]] for i in range(ndim)]

            #Repackage into colvar format despite min and max being the same
            colvar_parameters_sliced = []
            for i in range(len(axis_values)):
                axis_val = axis_values[i]
                colvar_parameters_sliced.append((axis_val, axis_val))

            return none_count, self.pes[idx], colvar_parameters_sliced

        elif none_count == 1:
            plot_axis = axis_vals.index(None)
            fixed_axis = [i for i, v in enumerate(axis_vals) if v is not None]

            #find index for the fixed axis
            idx = [slice(None) if i == plot_axis else (np.abs(self.axes[i] - axis_vals[i])).argmin() for i in range(ndim)]
            x_vals = self.axes[plot_axis]
            colvar_parameters_sliced = (np.min(x_vals), np.max(x_vals))
            sliced_pes = self.pes[tuple(idx)].astype(float)
            
            return none_count, sliced_pes, colvar_parameters_sliced

        elif none_count == 2:
            plot_axes = [i for i, v in enumerate(axis_vals) if v is None]
            fixed_axis = [i for i, v in enumerate(axis_vals) if v is not None][0] #we only get here with 3D data anyway

            idx = [slice(None) if i in plot_axes else (np.abs(self.axes[i] - axis_vals[i])).argmin() for i in range(ndim)]
            sliced_pes = self.pes[tuple(idx)].astype(float)

            x_axis_min_max = (self.axes[plot_axes[1]][0], self.axes[plot_axes[1]][-1])
            y_axis_min_max = (self.axes[plot_axes[0]][0], self.axes[plot_axes[0]][-1])

            colvar_parameters = [x_axis_min_max, y_axis_min_max]

            return none_count, sliced_pes, colvar_parameters_sliced
        
        else:
            print("Please specify at least one axis value for slicing or plotting.")

    def slice_out_mtd_graph(self, axis_vals, return_plt = False):
        """ 
        Visuallizes high dimensional data by slicing based on provided axis values and making a lower dimensional plot

        Parameters
            axis_vals: list
                The values for each axis to slice at, use None for the axis to plot along that axis
                [30, None, None] would slice at 30 on the x axis and plot a heatmap for y and z
            return_plt: bool
                If True, returns the matplotlib plt object instead of showing the plot directly.
        
        Returns
            plt (optional): matplotlib.pyplot object
        """
        axis_vals = axis_vals[::-1]  # Reverse to match zyx axes order when user inputs xyz

        if len(self.colvar_parameters) != self.ndim or len(axis_vals) != self.ndim:
            raise ValueError("colvar_parameters and axis_vals must match the number of dimensions in the data.")
        self.check_axes_in_bounds(axis_vals)
        none_count, sliced_pes, colvar_parameters_sliced = self._slice_nd(axis_vals)

        if none_count == self.ndim:
            #Reverse back for printing to user
            axis_str = ', '.join([f'{colvar_parameters_sliced[i][0]}' for i in range(len(colvar_parameters_sliced))][::-1])
            print(f"Energy at ({axis_str}): {sliced_pes} kJ/mol")

        elif none_count == 1:
            plot_axis = axis_vals.index(None)
            fixed_axis = [i for i, v in enumerate(axis_vals) if v is not None]
            ndim = self.ndim 

            #manufacture titel
            titel = f"Energy vs axis {self.ndim-1 - plot_axis} for {self.name} at"
            for i in range(len(fixed_axis))[::-1]:
                titel += f" {ndim-1 - fixed_axis[i]}={axis_vals[fixed_axis[i]]}"

            x_vals = self.axes[plot_axis]

            plt = plot.line_plotter_2d(
            x_vals, sliced_pes, x_var=f"axis {plot_axis}", y_var="kJ/mol",
            titel=titel, annotate_minima=True, save_fig=False, force_zero_start_y=True)

            if return_plt == True:
                return_plt
            else:
                plt.show() 
        
        elif none_count == 2:
            plot_axes = [i for i, v in enumerate(axis_vals) if v is None]
            fixed_axis = [i for i, v in enumerate(axis_vals) if v is not None][0] #we only get here with 3D data anyway

            plt = plot.heatmap(sliced_pes, x_axis_min_max=colvar_parameters_sliced[0], y_axis_min_max=colvar_parameters_sliced[1], 
                               x_var=f"axis {ndim-1 -plot_axes[1]}", y_var=f"axis {ndim-1 -plot_axes[0]}", heat_var="kJ/mol", 
                               titel=f"Energy heatmap for {self.name} at axis {ndim-1 - fixed_axis}={axis_vals[fixed_axis]}", 
                               annotate_minima=True, save_fig=False, aspect="auto")
            if return_plt == True:
                return_plt
            else:
                plt.show()     

    def collapse_energy_min(self, axis_lst):
        """ 
        Collapse axes away using np.min and return the zyx indexed PES after collapse.
        Parameters:
            axis_lst (list of int): List of axes to collapse, in xyz indexing.
        
        Returns:
            np.ndarray: The collapsed PES.
        """
        pes = self.pes
        axis_lst = sorted(self.ndim - 1 - np.array(axis_lst))  # Sort axes to avoid collapsing in the wrong order

        for axis in axis_lst[::-1]:  # Collapse axes one by one, starting from the highest index
            pes = np.min(pes, axis=axis)
        return pes

    def pes_to_pdf(self, t=298):
        """ 
        Calculate the probability density function (PDF) from potential energy surface (PES) data.

        Parameters:
            t (float): Temperature in Kelvin for the PDF calculation.

        Returns:
            None: The PDF is stored in the instance variable self.pdf.
        """
        gas_const = 8.314 #J/mol*K
        pes = self.pes * 1000 #kJ/mol to J/mol conversion
        pdf = np.exp(pes/(-gas_const*t))
        pdf = pdf / np.sum(pdf)
        self.pdf = pdf

    def plot_pdf(self, plot_log_pdf = False):
        """ 
        Plots the probability density function (PDF).
        Parameters:
            plot_log_pdf (bool): If True, plots the logarithm of the PDF.
        """
        if self.pdf is None:
            self.pes_to_pdf(t=298)  # Calculate PDF if not already done

        pdf = self.pdf

        if plot_log_pdf:
            pdf = np.log10(pdf + 1e-10)

        # Generate axes for the PDF plot
        if self.ndim == 1:
            x_axis = self.axes[0]
        elif self.ndim == 2:
            x_axis, y_axis = self.axes[1], self.axes[0]
        elif self.ndim == 3:
            x_axis, y_axis, z_axis = self.axes[2], self.axes[1], self.axes[0]

        if self.ndim == 1:
            plt = plot.line_plotter_2d(x_axis, pdf, x_var="x", y_var="Probability Density", 
                                       titel=f"probability distribution for {self.name}", annotate_minima=False, 
                                       save_fig=False, force_zero_start_y=True)
            plt.show()
        elif self.ndim == 2:
            # Generate a heatmap for the PDF
            plt = plot.heatmap((pdf), x_axis_min_max=(x_axis[0], x_axis[-1]), y_axis_min_max=(y_axis[0], y_axis[-1]) if y_axis is not None else (None, None),
                        x_var="x", y_var="y", heat_var="Probability Density", titel=f"Probability Density Function for {self.name}",
                        basename="", plot_type="pdf_heatmap", annotate_minima=False, save_fig=False, aspect="auto")
            plt.show()
        elif self.ndim == 3:
            plt = plot.plot_3d_scatter(x_axis, y_axis, z_axis, pdf, x_var="x", y_var="y", 
                                       z_var="z", heat_var="Probability Density", 
                                       titel=f"Probability Density Function for {self.name}", 
                                       save_fig=False)
            plt.show()

        else: 
            raise ValueError("Invalid dimensions on loaded data")

    def cumulative_dist(self, plot_axis, cutoff_lst=None, t=298, cumulative = True):
        """ 
        Calculate the cumulative distribution function (CDF) for a given axis.

        Parameters:
            plot_axis (int): The axis along which to calculate the CDF.
            cutoff_lst (list of tuples): List of tuples specifying the min and max values for each axis. 
                                        If None, no cutoff is applied for that axis.
            t (float): Temperature in Kelvin for the PDF calculation.
            cumulative(bool): Wheter to return a cumulative distribution or the sliced PDF using the cutoffs

        Returns:
            x_vals (np.ndarray): The values along the specified axis.
            cumulative (np.ndarray): The cumulative distribution function values.
        """
        cutoff_lst = cutoff_lst[::-1] if cutoff_lst is not None else None  # Reverse to match zyx axes order when user inputs xyz
        plot_axis = self.ndim - 1 - plot_axis  # Reverse the plot_axis to match zyx axes order

        if self.pdf is None:
            self.pes_to_pdf(t)  # Calculate PDF if not already done

        pdf = self.pdf
        ndim = pdf.ndim

        axes = [np.linspace(self.colvar_parameters[i][0], self.colvar_parameters[i][1], pdf.shape[i]) for i in range(ndim)]
        mask = np.ones_like(pdf, dtype=bool)

        if cutoff_lst is not None:
            for axis, minmax in enumerate(cutoff_lst):
                if minmax is None:
                    continue
                minval, maxval = minmax
                axis_vals = axes[axis]
                axis_mask = np.ones_like(axis_vals, dtype=bool)
                if minval is not None:
                    axis_mask = axis_mask & (axis_vals >= minval)
                if maxval is not None:
                    axis_mask = axis_mask & (axis_vals <= maxval)
                shape = [1]*pdf.ndim
                shape[axis] = -1
                mask = mask & np.reshape(axis_mask, shape)

        pdf_masked = np.where(mask, pdf, 0)

        sum_axes = tuple(i for i in range(ndim) if i != plot_axis)
        summed_pdf = np.sum(pdf_masked, axis=sum_axes)

        x_vals = axes[plot_axis]

        if cumulative:
            return x_vals, np.cumsum(summed_pdf)

        else:
            return x_vals, summed_pdf

    def collapse_axis_slice(self, plot_axis, cutoff_lst):
        """ 
        Calculate the PES by first applying a cutoff and then collapsing the axes using np.min.

        Parameters:
            plot_axis (int): The axis along which to calculate the CDF.
            cutoff_lst (list of tuples): List of tuples specifying the min and max values for each axis. 
                                         eg [(3, 9, (0, 180), (-180, 180)]

        Returns:
            x_vals (np.ndarray): The values along the specified axis.
            pes (np.ndarray): The sliced pes values.
        """
        cutoff_lst = cutoff_lst[::-1] if cutoff_lst is not None else None # Reverse to match zyx axes order when user inputs xyz
        plot_axis = self.ndim - 1 - plot_axis  # Reverse the plot_axis to match zyx axes order

        pes = self.pes
        ndim = self.ndim

        axes = self.axes
        mask = np.ones_like(pes, dtype=bool)

        if cutoff_lst is not None:
            for axis, minmax in enumerate(cutoff_lst):
                if minmax is None:
                    continue
                minval, maxval = minmax
                axis_vals = axes[axis]
                axis_mask = np.ones_like(axis_vals, dtype=bool)
                if minval is not None:
                    axis_mask = axis_mask & (axis_vals >= minval)
                if maxval is not None:
                    axis_mask = axis_mask & (axis_vals <= maxval)
                shape = [1]*pes.ndim
                shape[axis] = -1
                mask = mask & np.reshape(axis_mask, shape)

        pes_masked = np.where(mask, pes, np.inf)

        min_axes = tuple(i for i in range(ndim) if i != plot_axis)
        summed_pes = np.min(pes_masked, axis=min_axes)

        x_vals = axes[plot_axis]

        return x_vals, summed_pes


    def zoom_on_pes(self, cutoff_lst):
        """
        Allows one to zoom in on the PES and get back axes values and the relevant part of the PES.

        Parameters:
        cutoff_lst : list of (min, max) tuples or None
            Cutoff ranges for each axis in xyz order. Use None to leave an axis unchanged.

        Returns:
            zoomed_colvars (list of tuples): The minmax values of each axis
            pes_zoomed (np.ndarray): The zoomed-in PES subarray.
        """
        pes = self.pes
        cutoff_lst = cutoff_lst[::-1] if cutoff_lst is not None else None  # reverse to match z,y,x indexing
        axes = self.axes
        ndim = self.ndim

        slices = []
        zoomed_axes = []

        for axis, minmax in enumerate(cutoff_lst):
            axis_vals = axes[axis]

            if minmax is None:
                # Keep full axis
                slices.append(slice(None))
                zoomed_axes.append(axis_vals)
                continue

            minval, maxval = minmax

            # Boolean mask for this axis
            axis_mask = np.ones_like(axis_vals, dtype=bool)
            if minval is not None:
                axis_mask &= (axis_vals >= minval)
            if maxval is not None:
                axis_mask &= (axis_vals <= maxval)

            # Find the slice indices
            valid_idx = np.where(axis_mask)[0]
            if len(valid_idx) == 0:
                raise ValueError(f"No values found within cutoff {minmax} for axis {axis}")

            start, stop = valid_idx[0], valid_idx[-1] + 1
            slices.append(slice(start, stop))
            zoomed_axes.append(axis_vals[start:stop])

        # Apply slices to PES
        pes_zoomed = pes[tuple(slices)]

        zoomed_colvars = [(np.min(axis), np.max(axis)) for axis in zoomed_axes][::-1]

        return zoomed_colvars, pes_zoomed  # flip back to xyz order

        