import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import re

from lmfit import Model, Parameters
import matplotlib as mpl

def mpt_to_df(filename, eis=False):
    """
    Function for reading .mpt file (from Biologic output) and converting to pandas dataframe
    
    Input
    ----------
    filename (str):
        The filename, including the path and file extension
        
    eis (bool) optional, default=False:
        If the data is for a sequence of EIS measurements, the data for each spectra are returned in separate columns of the dataframe
        
    Returns
    ----------
    A pandas dataframe with columns corresponding to the variables measured in the Biologic file (column labels identical to Biologic column labels)
    
    """
    with open(filename) as f:
        content = f.readlines()

    n_header_lines = int(re.findall("\d+", content[1])[0])
    column_headers = content[n_header_lines-1].strip("\n").split("\t")[:-1]
    data = np.array([line.strip("\n").split("\t") for line in content[n_header_lines:]], dtype=float)
    if eis == False:
        return pd.DataFrame(data, columns=column_headers)
    else:
        frequencies = np.unique(data[:, 0])
        cycle_id = np.array([item for sublist in [[n]*frequencies.shape[0] for n in range(int(data.shape[0]/frequencies.shape[0]))] 
                             for item in sublist]).reshape(data.shape[0], 1)

        EIS_data_labelled = np.hstack((cycle_id, data))

        column_headers.insert(0, "Cycle id")

        return pd.DataFrame(EIS_data_labelled, columns=column_headers)
      
class CycVolt(object):
    def __init__(self, filename, tolerance=0.1):
        """
        Read, store, display, and process the cyclic voltammetry data from Biologic files
        
        Inputs
        ----------
        filename (str):
            The filename, including the path and the file extension
            
        tolerance (float) optional, default=0.1:
            Maximum difference between the last voltage measurement in each cycle and the voltage limits.
            Used for identifying the beginnings and ends of scans. 
            Eg: the voltage limit may be 2.8 V, but the final measured voltage in a scan may be listed as 2.79 V.
            
        Attributes:
        ----------
        self
        ┠ n_cycles    (int)
        ┠ scan_rate   (float)
        ┠ charge      (_State class)
        ┃ ┠ current   (list of arrays)
        ┃ ┠ voltage   (list of arrays)
        ┃ ┕ time      (list of arrays)
        ┠ discharge   (_State class)
        ┃ ┠ current   (list of arrays)
        ┃ ┠ voltage   (list of arrays)
        ┃ ┕ time      (list of arrays)
            
        Methods
        ----------
        plot_cv(figsize, cycles_to_plot, cycles_to_skip, cmap, category_cmap, reverse_scan_order):
            returns figure with cyclic voltammograms plotted overlaid for specified cycle numbers
        
        calculate_peak_capacity_ratio:
            returns self.peak_ratio_values, with capacity attained at upper and lower cathodic current peaks
            (designed for use with Li-S cells with two cathodic current peaks)
            
        """
        with open(filename) as f:
            for line in f.readlines():
                if "dE/dt" in line and "unit" not in line:
                    self.scan_rate = float(re.findall("\d+.\d+", line)[0])
                    
        data = mpt_to_df(filename)
        voltage = np.array(data["Ewe/V"])
        dV = voltage[1:]-voltage[:-1]
        current = np.array(data["<I>/mA"])
        charge_idx = np.argwhere(np.sign(dV)==1).flatten()
        discharge_idx = np.argwhere(np.sign(dV)==-1).flatten()
        time = np.array(data["time/s"])
        
        discharge_ends_WITH_super = [num for n, num in enumerate(discharge_idx[:-1]) if discharge_idx[n+1]-discharge_idx[n]!=1]
        discharge_ends = [num for num in discharge_ends_WITH_super if any([abs(voltage[num+1]-limit)<tolerance for limit in [np.min(voltage), np.max(voltage)]])]
        
        charge_ends_WITH_super = [num for n, num in enumerate(charge_idx[:-1]) if charge_idx[n+1]-charge_idx[n]!=1]
        charge_ends = [0]+[num for num in charge_ends_WITH_super if any([abs(voltage[num+1]-limit)<tolerance for limit in [np.min(voltage), np.max(voltage)]])]
        
        if discharge_ends[-1]>charge_ends[-1]:
            n_cycles = len(discharge_ends)
            charge_ends += [charge_idx[-1]]
        else:
            n_cycles = len(charge_ends)
            discharge_ends += [discharge_idx[-1]]
            
        self.n_cycles = n_cycles
            
        class _State(object):
            def __init__(state_self, state):              
                if state == "discharge":
                    state_self.voltage = [voltage[charge_ends[n]:discharge_ends[n]] for n in range(len(discharge_ends))]
                    state_self.current = [current[charge_ends[n]:discharge_ends[n]] for n in range(len(discharge_ends))]
                    state_self.time = [time[charge_ends[n]:discharge_ends[n]] for n in range(len(discharge_ends))]
                elif state == "charge":              
                    state_self.voltage = [voltage[discharge_ends[n]:charge_ends[n+1]] for n in range(len(charge_ends)-1)]
                    state_self.current = [current[discharge_ends[n]:charge_ends[n+1]] for n in range(len(charge_ends)-1)]
                    state_self.time = [time[discharge_ends[n]:charge_ends[n+1]] for n in range(len(charge_ends)-1)]
        setattr(self, "discharge", _State("discharge"))
        setattr(self, "charge", _State("charge"))
        
    def plot_cv(self, figsize=(8, 8), cycles_to_plot="all", cycles_to_skip="none", cmap="tab20", 
                category_cmap=True, reverse_scan_order=True):
        """
        Plots the cyclic voltammograms overlaid for a specified range of cycles
        
        Inputs
        ----------
        figsize (tuple, optional - default=(8,8)):
            Output size of the figure (inches)
            
        cycles_to_plot (list or str, optional - default="all"):
            List of cycles to plot (starting at 0 for first cycle. NB legend will label first cycle as 1)
            
        cycles_to_skip (list or str, optional - default="none"):
            List of cycles to skip (NB legend will show true index of plotted cycles)
            
        cmap (str, optional - default = "tab20"):
            The colormap to use for plotting each cycle. NB colormap is not cyclical (to ensure unique colors for sequential maps) - ensure a colormap with enough values is selected
            For colormap options see https://colorbrewer2.org [accessed 17/11/2022]
            
        category_cmap (bool, optional - default=True):
            Use True for categorical cmaps (eg tab10, tab20, Dark2)
            Use False for sequential cmaps (eg viridis, inferno, Grays)
            
        reverse_scan_order (bool, optional - default=True):
            Plots the first scan first, overlaid with later scans.
            Later scans usually have lower capacity, so plotting later scans 'on top' avoids overlap hiding values
            
        Returns
        ----------
        Figure, Axis
        Tip! using:
            f, ax = self.plot_cv()
        means that properties can be set retrospectively.
        Eg: ax.set_xlim([1.5, 2.8])
        """
        
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=figsize)
        
        if cycles_to_plot == "all":
            n_cycles = list(np.arange(0, self.n_cycles))
        else:
            n_cycles = cycles_to_plot
            
        if cycles_to_skip == "none":
            n_cycles = n_cycles
        else:
            n_cycles = [cyc for cyc in n_cycles if cyc not in cycles_to_skip]
        
        if category_cmap==True:
            plot_colors = [vars(plt.cm)[cmap](i) for i in range(len(n_cycles))]
            for ncycle, cycle in enumerate(n_cycles):
                ax.plot([], [], color=plot_colors[ncycle], label=cycle+1)
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            plot_colors = [vars(plt.cm)[cmap](i) for i in range(np.linspace(0, 1, len(n_cycles)))]
            ax.set_position((0.15, 0.15, 0.75, 0.75))
            cbar_ax = f.add_axes((0.8, 0.15, 0.05, 0.6))
            cbar_ticks = np.array([0, 1], dtype=int)
            cmap = plt.cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ticks=cbar_ticks, cax=cbar_ax)
            cbar.set_label("Cycle number", rotation=90)
            cbar.set_ticks([1, max(n_cycles)])
            cbar.set_ticklabels([1, max(n_cycles)])
            
            
        ax.tick_params(which="both", tickdir="in", right=True, top=True)
        ax.set_xlabel("Potential vs. Li/ Li$^{+}$ (V)")
        ax.set_ylabel("Current (mA)")
        ax.axhline(0, color="k", lw=0.5)
        
        for ncycle, cycle in enumerate(n_cycles):
            if reverse_scan_order == True:
                z_order = self.n_cycles-ncycle
            else:
                z_order = ncycle
            try:
                ax.plot(self.discharge.voltage[cycle], self.discharge.current[cycle], color=plot_colors[ncycle], zorder=z_order)
                ax.plot(self.charge.voltage[cycle], self.charge.current[cycle], color=plot_colors[ncycle], zorder=z_order)
            except:
                continue
                            
        plt.tight_layout()
        
        return f, ax
    
    def calculate_peak_capacity_ratio(self, t_step=10, intercept_range=5):
        """
        Calculate the capacity estimate for cathodic current peaks (designed for Li-S cells).
        Capacities are 'negative' because of 'negative' current during reduction. 
        
        How to use:
            self.calculate_peak_capacity_ratio()
            
        Access results using:
            self.peak_ratio_values
            ┠ EL0_capacity (list of capacities in mAh for peak at ~2.1 V)
            ┠ EH0_capacity (list of capacities in mAh for peak at ~2.4 V)
            ┆ Hidden values for checking...
            ┆   ┠ _EH0_capacity_pos - capacity at 2.4 V for current < 0 mA
            ┆   ┠ _EH0_capacity_neg - capacity due to polarization at start of cathodic
            ┆   ┠ _EL0_capacity_pos - capacity at 2.1 V for current < 0 mA
            ┆   ┠ _EL0_capacity_pos - capacity due to polarization at start of anodic
            
        Inputs
        ----------
        t_step (int, optional-default value = 10):
            Peaks are identified using the first derivative of the current/voltage. 
            t_step sets the gradient step range: (y[t_step:]-y[:t_step])/(x[t_step:]-y[:t_step])
            Larger t_step => greater 'smoothing'
        
        intercept_range (int, optional-default value =5):
            Peaks are identified using 0 values of first derivative. 
            Intercept range (dx) checks that x-dx values are <0 and x+dx values are >0 to find 'crossover'
            Larger intercept_range => lower sensitivity (to avoid falsely identifying inflection points as peaks)
            
        """
        setattr(self, "peak_ratio_values", _cvPeaks(cv=self, t_step=t_step, intercept_range=intercept_range))

class _cvPeaks(object):
    def __init__(peaks_self, cv, t_step=10, intercept_range=5):
        """
        See CycVolt.calculate_peak_capacity_ratio for details
        """
        for key in ["_EH0_capacity_pos", "_EH0_capacity_neg", "EH0_capacity", "_EL0_capacity_pos", "_EL0_capacity_neg", "EL0_capacity"]:
            setattr(peaks_self, key, [])
            
        for hour in range(cv.n_cycles):
            try:
                time_d = cv.discharge.time[hour]
                current_d = cv.discharge.current[hour]
                voltage_d = cv.discharge.voltage[hour]
                
### Some of my data have EIS spectra collected during CV measurements. This pauses the measurement (causing a time step difference) and can lead to 
### cell relaxation (current spike when CV resumes). These lines of code remove this gap and spike. 
                dtdt = time_d[1:]-time_d[:-1]
                eis_idx = np.argwhere(dtdt>2*np.nanmedian(dtdt)).flatten()

                if eis_idx.shape[0] > 0:
                    current_d[eis_idx[0]:eis_idx[0]+1] = 0

                dr_idx = np.argwhere(np.sign(current_d)==-1)
                dr_time = time_d[dr_idx]
                dr_current = current_d[dr_idx]
                dr_voltage = voltage_d[dr_idx]

                didv = (current_d[t_step:]-current_d[:-t_step])/(voltage_d[t_step:]-voltage_d[:-t_step])
                v_dv = voltage_d[int(t_step):int(t_step)+didv.shape[0]]

                current_maxima = [[x] for x in range(intercept_range, didv.shape[0]-intercept_range) if 
                     all(np.sign(didv[x-intercept_range:x])==-1) and all(np.sign(didv[x:x+intercept_range])==1)]

                current_maxima_in_range = []
                for m in current_maxima:
                    if m > np.nanmin(dr_idx) and m < np.nanmax(dr_idx):
                        if eis_idx.shape[0] > 0 and m != eis_idx and m != eis_idx+1:
                            current_maxima_in_range.append(m[0])
                        else:
                            current_maxima_in_range.append(m[0])

                EH0_start = np.nanmin(dr_idx.flatten())
                EH0_end = current_maxima_in_range[np.argmin(abs(current_maxima_in_range-EH0_start))]

                EL0_end = np.nanmax(dr_idx.flatten())
                EL0_start = current_maxima_in_range[np.argmin(abs(current_maxima_in_range-EL0_end))]
                EH0_capacity_pos = np.trapz(y=current_d[EH0_start:EH0_end], x=time_d[EH0_start:EH0_end]/60/60)
                EL0_capacity_pos = np.trapz(y=current_d[EL0_start:EL0_end], x=time_d[EL0_start:EL0_end]/60/60)

                ## charge
                current_c = cv.charge.current[hour]
                voltage_c = cv.charge.voltage[hour]
                time_c = cv.charge.time[hour]

                c_EL0_idx = np.arange(np.argmin(abs(voltage_c-voltage_d[EL0_end])), np.argmin(abs(voltage_c-voltage_d[EL0_start])))
                voltage_c_EL0 = voltage_c[c_EL0_idx]
                current_c_EL0 = current_c[c_EL0_idx]
                time_c_EL0 = time_c[c_EL0_idx]

                c_EH0_idx = np.arange(np.argmin(abs(voltage_c-voltage_d[EH0_end])), np.argmin(abs(voltage_c-voltage_d[EH0_start])))
                voltage_c_EH0 = voltage_c[c_EH0_idx]
                current_c_EH0 = current_c[c_EH0_idx]
                time_c_EH0 = time_c[c_EH0_idx]

                c_EL0_neg = np.argwhere(np.sign(current_c_EL0)==-1).flatten()
                c_EH0_neg = np.argwhere(np.sign(current_c_EH0)==-1).flatten()

                EH0_capacity_neg = np.trapz(y=current_c_EH0[c_EH0_neg], x=time_c_EH0[c_EH0_neg]/60/60)
                EL0_capacity_neg = np.trapz(y=current_c_EL0[c_EL0_neg], x=time_c_EL0[c_EL0_neg]/60/60)
        
                peaks_self._EH0_capacity_pos.append(EH0_capacity_pos)
                peaks_self._EH0_capacity_neg.append(EH0_capacity_neg)
                peaks_self.EH0_capacity.append(EH0_capacity_pos-EH0_capacity_neg)

                peaks_self._EL0_capacity_pos.append(EL0_capacity_pos)
                peaks_self._EL0_capacity_neg.append(EL0_capacity_neg)
                peaks_self.EL0_capacity.append(EL0_capacity_pos-EL0_capacity_neg)
            except:
                for key in ["_EH0_capacity_pos", "_EH0_capacity_neg", "EH0_capacity", "_EL0_capacity_pos", "_EL0_capacity_neg", "EL0_capacity"]:
                    vars(peaks_self)[key]
  
  class MultiCV(object):
    """
    Read, store, display, and process the cyclic voltammetry (CV) data from Biologic files for series of CV scans (eg at different rates)

    Inputs
    ----------
    folder (str):
        path to folder containing all the CV files in .mpt format
        Biologic labels CV files with "_CV_" in name, so the filenames do not need to be specified individually

    Attributes
    ----------
    self
    ┕ data    (dict: keys=scan rates, values=CycVolt objects)
              see CycVolt for details

    Methods
    ----------
    plot_cv(reverse_scan_order, cmap, category_cmap)

    calculate_diffusion(cycle_type, center_voltage, voltage_tolerance, step)
    """
    def __init__(self, folder):

        allfiles = glob.glob(os.path.join(folder, "*.mpt"))
        cv_files = [file for file in allfiles if "_CV_" in os.path.split(file)[-1]]
        self.data = {}
        for file in cv_files:
            cv_load = CycVolt(file)
            self.data.update([(cv_load.scan_rate,
                               cv_load)])
            
    def plot_cv(self, reverse_scan_order=True, cmap="tab10", category_cmap=True):
        """
        Plots the cyclic voltammograms for all data (i.e. all scan rates), with one colour per scan rate. 
        
        Inputs
        ----------
        reverse_scan_order (bool, optional - default=True)
            Plots the lower scan rate values 'on top': lower scan rates have lower current.
            Using reverse_scan_order = True prevents obscuring of lower currents by higher currents
            
        cmap (str, optional - default="tab10"):
            The colormap to use for plotting each cycle. NB colormap is not cyclical (to ensure unique colors for sequential maps) - ensure a colormap with enough values is selected
            For colormap options see https://colorbrewer2.org [accessed 17/11/2022]
            
        category_cmap (bool, optional - default=True):
            Use True for categorical cmaps (eg tab10, tab20, Dark2)
            Use False for sequential cmaps (eg viridis, inferno, Grays)
        
        """
        import matplotlib.pyplot as plt
        if category_cmap==True:
            plot_colors = [vars(plt.cm)[cmap](i) for i in range(len(self.data))]
        else:
            plot_colors = [vars(plt.cm)[cmap](i) for i in range(np.linspace(0, 1, len(self.data)))]
        f, ax = plt.subplots()
        ax.tick_params(which="both", tickdir="in", right=True, top=True)
        ax.set_xlabel("Potential vs. Li/ Li$^{+}$ (V)")
        ax.set_ylabel("Current (mA)")
        ax.axhline(0, color="k", lw=0.5)
        
        for ndata, (scan_rate, trace) in enumerate(self.data.items()):
            if reverse_scan_order == True:
                z_order = len(self.data)-ndata
            else:
                z_order = ndata
            ax.plot([], [], color=plot_colors[ndata], label=scan_rate)
            for cycle in range(trace.n_cycles):
                try:
                    ax.plot(trace.discharge.voltage[cycle], trace.discharge.current[cycle], color=plot_colors[ndata], zorder=z_order)
                    ax.plot(trace.charge.voltage[cycle], trace.charge.current[cycle], color=plot_colors[ndata], zorder=z_order)
                except:
                    continue
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        
        return f, ax
    
    def calculate_diffusion(self, cycle_type, center_voltage, voltage_tolerance=0.2, step=10):
        """
        Calculates the peak currents (and corresponding voltages) and relationship between current peak and scan rate for use in the Randles-Sevcik equation (see Notes)
        The Randles-Sevcik equation should be used with reversible processes (i.e. current peaks remaining at same voltage for all scan rates)
        
        How to use - example:
        diff = self.calculate_diffusion(cycle_type="cathodic", center_voltage=2.1)
        ┠ ip   (dict, with keys=scan rates, values=lists of current peaks)
        ┠ slope (float, the slope of ip against the square root of the scan rates)
        ┆ Hidden values for checking and plotting error bars:
            ┠ _all_current_peaks
            ┠ _all_voltage_peaks
            ┠ _ip_voltages
            ┠ _combined_x
            ┠ _combined_y
            ┠ _intercept
            ┠ _total_fit
            
        
        Inputs
        ----------
        cycle_type (str):
            allowed values: "cathodic" or "anodic" (or "discharge" or "charge")
            
        center_voltage (float):
            estimated centre of current peak (e.g. 2.1)
            deviations from estimate can be set using voltage_tolerance (see below)
            
        voltage_tolerance (float, optional - default=0.2):
            maximum allowed deviation of the identified peak from the specified centre (+/-)
            eg center_voltage=2.1, voltage_tolerance=0.2 would return all current peaks between 1.9V and 2.3V
            
        step (int, optional - default=10):
            peaks are identified using the first derivative: (current[step:]-current[:-step])/(voltage[step:]-voltage[:-step])
            larger step => greater 'smoothing' (less sensitivity to current fluctuation)
            
            
        Notes
        ----------
        i_p = 0.4463*n*F*A*C*(n*F*v*D/R/T)**(1/2)
        
            i_p = peak current (A)                                      *** Note A not mA!
            n = number of electrons transferred (2 for Li-S reactions)
            F = Faraday constant (96485 C mol-1)
            A = Geometric area of electrode (cm2)
            D = Diffusion coefficient (cm2/s)
            C = Concentration of conducting ion (mol/cm3)               *** Note cm3 not L!
            v = Scan rate (V/s)                                         *** Note V not mV!
            R = Gas constant (8.31 J mol-1 K-1)
            T = Temperature (K)
            
        Therefore, plotting i_p on y axis and sqrt(v) on x axis:
        D = R*T/n/F*(slope/0.4463/n/F/A/C)**2
        """
        peak_currents = dict([(rate, {}) for rate in self.data.keys()])
        peak_voltages = dict([(rate, {}) for rate in self.data.keys()])

        for scan_rate in self.data.keys():
            for cycle in range(self.data[scan_rate].n_cycles):
                try:
                    peak_currents[scan_rate].update([(cycle, None)])
                    peak_voltages[scan_rate].update([(cycle, None)])
                    voltage = vars(self.data[scan_rate])[cycle_type].voltage[cycle]
                    current = vars(self.data[scan_rate])[cycle_type].current[cycle]/1000

                    didV = (current[step:]-current[:-step])/(voltage[step:]-voltage[:-step])
                    x_di = voltage[int(step/2):int(step/2)+didV.shape[0]]

                    d2idV2 = (didV[step:]-didV[:-step])/(x_di[step:]-x_di[:-step])
                    x_di2 = x_di[step: step+d2idV2.shape[0]]

                    zero_gradient = [nx for nx, x in enumerate(didV[1:]) if np.sign(didV[nx-1]) != np.sign(didV[nx]) and nx<d2idV2.shape[0]]
                    if cycle_type =="discharge":
                        peaks = [x for x in zero_gradient if np.sign(d2idV2[x])==1 and x>step]
                    elif cycle_type=="charge":
                        peaks = [x for x in zero_gradient if np.sign(d2idV2[x])==-1 and x>step]
                        
                    peak_voltages[scan_rate][cycle]=voltage[peaks]
                    peak_currents[scan_rate][cycle]=current[peaks]
                except:
                    pass

        selected_voltages = dict([(rate, []) for rate in self.data.keys()])
        selected_currents = dict([(rate, []) for rate in self.data.keys()])

        for scan_rate in self.data.keys():
            for cycle, peaks in peak_voltages[scan_rate].items():
                try:
                    nearest_voltage = np.array(peaks)[np.argmin(abs(np.array(peaks)-center_voltage))]
                    if abs(nearest_voltage-center_voltage)<voltage_tolerance:
                        selected_voltages[scan_rate].append(nearest_voltage)
                        selected_currents[scan_rate].append(np.array(peak_currents[scan_rate][cycle])[np.argmin(abs(np.array(peaks)-center_voltage))])
                except:
                    selected_voltages[scan_rate].append(np.nan)
                    selected_currents[scan_rate].append(np.nan)
        combined_xs = [item for sublist in [[np.sqrt(keys/1000)]*len(values) for keys, values in selected_currents.items()] for item in sublist]
        combined_ys = [item for sublist in [*selected_currents.values()] for item in sublist]            

        class _RandlesSevcik(object):
            def __init__(rs_self):
                rs_self._all_current_peaks = peak_currents
                rs_self._all_voltage_peaks = peak_voltages
                rs_self.ip = selected_currents
                rs_self._ip_voltages = selected_voltages
                rs_self._combined_x = np.array(combined_xs)[np.isfinite(combined_ys)]
                rs_self._combined_y = np.array(combined_ys)[np.isfinite(combined_ys)]
                try:
                    from lmfit.models import LinearModel

                    model = LinearModel()
                    fit_result = model.fit(x=rs_self._combined_x,
                                           data=rs_self._combined_y)

                    rs_self.slope = fit_result.best_values["slope"]
                    rs_self._intercept = fit_result.best_values["intercept"]
                    rs_self._total_fit = fit_result.best_fit
                except:
                    print("No fit")
                
        return _RandlesSevcik() 
