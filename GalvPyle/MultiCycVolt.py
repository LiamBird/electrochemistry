import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from CycVolt import CycVolt

from lmfit import Model, Parameters
import matplotlib as mpl

class MultiCV(object):
    def __init__(self, folder):
        allfiles = glob.glob(os.path.join(folder, "*.mpt"))
        cv_files = [file for file in allfiles if "_CV_" in os.path.split(file)[-1]]
        self.data = {}
        for file in cv_files:
            cv_load = CycVolt(file)
            self.data.update([(cv_load.scan_rate,
                               cv_load)])
            
    def plot_cv(self, reverse_scan_order=True, cmap="tab20", category_cmap=True):
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
                    if cycle_type == "cathodic":
                        peaks = [x for x in zero_gradient if np.sign(d2idV2[x])==1 and x>step]
                    elif cycle_type == "anodic":
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
