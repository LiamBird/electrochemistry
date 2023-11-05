import numpy as np
from mpt_to_df import mpt_to_df

class cvPeaks(object):
    def __init__(peaks_self, cv, t_step=10, intercept_range=5):
        for key in ["_EH0_capacity_pos", "_EH0_capacity_neg", "EH0_capacity", "_EL0_capacity_pos", "_EL0_capacity_neg", "EL0_capacity", "current_maxima"]:
            setattr(peaks_self, key, [])
            
        for hour in range(cv.n_cycles):
            try:
                time_d = cv.cathodic.time[hour]
                current_d = cv.cathodic.current[hour]
                voltage_d = cv.cathodic.voltage[hour]

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
                            
                peaks_self.current_maxima.append(current_maxima_in_range)

                EH0_start = np.nanmin(dr_idx.flatten())
                EH0_end = current_maxima_in_range[0]
#                 EH0_end = current_maxima_in_range[np.argmin(abs(current_maxima_in_range-EH0_start))]

                EL0_end = np.nanmax(dr_idx.flatten())
                EL0_start = current_maxima_in_range[np.argmin(abs(current_maxima_in_range-EL0_end))]
                EH0_capacity_pos = np.trapz(y=current_d[EH0_start:EH0_end], x=time_d[EH0_start:EH0_end]/60/60)
                EL0_capacity_pos = np.trapz(y=current_d[EL0_start:EL0_end], x=time_d[EL0_start:EL0_end]/60/60)

                ## anodic
                current_c = cv.anodic.current[hour]
                voltage_c = cv.anodic.voltage[hour]
                time_c = cv.anodic.time[hour]

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


class CycVolt(object):
    def __init__(self, filename, tolerance=0.1):
#         print("Using correct function")
        import re
        with open(filename) as f:
            for line in f.readlines():
                if "dE/dt" in line and "unit" not in line:
                    self.scan_rate = float(re.findall("\d+.\d+", line)[0])
                    
        self.version = "13/10/2023"
        self._change_log = ["27/09/2023: modified calculation of peak capacity ratio to use target voltage to deliniate peaks rather than seeking maximum current as in previous versions",
                           "10/10/2023: Repaired peak capacity ratio calculation to output _PeakRatio as attribute of CV object, to enable reference by plot function",
                            "13/10/2023: Added '_turning_idx' to peak_ratio attributes to enable finding of current at 'overlap'"]
                    
        data = mpt_to_df(filename)
        voltage = np.array(data["Ewe/V"])
        dV = voltage[1:]-voltage[:-1]
        current = np.array(data["<I>/mA"])
        anodic_idx = np.argwhere(np.sign(dV)==1).flatten()
        cathodic_idx = np.argwhere(np.sign(dV)==-1).flatten()
        time = np.array(data["time/s"])
        
        cathodic_ends_WITH_super = [num for n, num in enumerate(cathodic_idx[:-1]) if cathodic_idx[n+1]-cathodic_idx[n]!=1]
        cathodic_ends = [num for num in cathodic_ends_WITH_super if any([abs(voltage[num+1]-limit)<tolerance for limit in [np.min(voltage), np.max(voltage)]])]
        
        anodic_ends_WITH_super = [num for n, num in enumerate(anodic_idx[:-1]) if anodic_idx[n+1]-anodic_idx[n]!=1]
        anodic_ends = [0]+[num for num in anodic_ends_WITH_super if any([abs(voltage[num+1]-limit)<tolerance for limit in [np.min(voltage), np.max(voltage)]])]
        
        if cathodic_ends[-1]>anodic_ends[-1]:
            n_cycles = len(cathodic_ends)
            anodic_ends += [anodic_idx[-1]]
        else:
            n_cycles = len(anodic_ends)
            cathodic_ends += [cathodic_idx[-1]]
            
        self.n_cycles = n_cycles
            
        class _State(object):
            def __init__(state_self, state):
                
                ## Patch
                state_self.voltage = []
                state_self.current = []
                state_self.time = []
                ## Patch
                
                if state == "cathodic":
                    ## Patch
                    for n in range(len(cathodic_ends)):
                        try:
                            state_self.voltage.append(voltage[anodic_ends[n]:cathodic_ends[n]])
                            state_self.current.append(current[anodic_ends[n]:cathodic_ends[n]])
                            state_self.time.append(time[anodic_ends[n]:cathodic_ends[n]])
                        except:
                            print("Failed cathodic cycle {}".format(n))
                    ## Patch

#                     state_self.voltage = [voltage[anodic_ends[n]:cathodic_ends[n]] for n in range(len(cathodic_ends))]
#                     state_self.current = [current[anodic_ends[n]:cathodic_ends[n]] for n in range(len(cathodic_ends))]
#                     state_self.time = [time[anodic_ends[n]:cathodic_ends[n]] for n in range(len(cathodic_ends))]
                elif state == "anodic":
                    ## Patch
                    for n in range(len(anodic_ends)-1):
                        try:
                            state_self.voltage.append(voltage[cathodic_ends[n]:anodic_ends[n+1]])
                            state_self.current.append(current[cathodic_ends[n]:anodic_ends[n+1]])
                            state_self.time.append(time[cathodic_ends[n]:anodic_ends[n+1]])
                        except:
                            print("Failed anodic cycle {}".format(n))
                    ## Patch
                
#                     state_self.voltage = [voltage[cathodic_ends[n]:anodic_ends[n+1]] for n in range(len(anodic_ends)-1)]
#                     state_self.current = [current[cathodic_ends[n]:anodic_ends[n+1]] for n in range(len(anodic_ends)-1)]
#                     state_self.time = [time[cathodic_ends[n]:anodic_ends[n+1]] for n in range(len(anodic_ends)-1)]
        setattr(self, "cathodic", _State("cathodic"))
        setattr(self, "anodic", _State("anodic"))
        
    def plot_cv(self, figsize=(8, 8), cycles_to_plot="all", cycles_to_skip="none", cmap="tab20", category_cmap=True, reverse_scan_order=True):
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
                ax.plot(self.cathodic.voltage[cycle], self.cathodic.current[cycle], color=plot_colors[ncycle], zorder=z_order)
                ax.plot(self.anodic.voltage[cycle], self.anodic.current[cycle], color=plot_colors[ncycle], zorder=z_order)
            except:
                continue
                    
#         ax.set_xlim([1.5, 2.8])
        
        plt.tight_layout()
        
        return f, ax
    
    def calculate_peak_capacity(self, dt_step=10, target_voltage=2.2):

        class _PeakRatio(object):
            def __init__(peak_self):
                peak_self.EH_capacity = []
                peak_self.EL_capacity = []
                peak_self._anodic_contribution = []
                peak_self._turning_voltage = []
                peak_self._turning_idx = []
                peak_self.EH_capacity_fraction = []
                peak_self.EL_capacity_fraction = []
                peak_self.total_capacity = []

        peak_ratio = _PeakRatio()

        for cycle_number in range(self.n_cycles):
            gradient = (self.cathodic.current[cycle_number][dt_step:]-self.cathodic.current[cycle_number][:-dt_step])/(self.cathodic.time[cycle_number][dt_step:]-self.cathodic.time[cycle_number][:-dt_step])
            gradient_dt = self.cathodic.time[cycle_number][int(dt_step/2): int(dt_step/2)+gradient.shape[0]]
            maxima = np.argwhere([all(np.sign(gradient[x-dt_step:x])==1) and all(np.sign(gradient[x:x+dt_step])==-1) for x in range(dt_step, gradient.shape[0]-dt_step)]).flatten()
            d2ydx2 = (gradient[dt_step:]-gradient[:-dt_step])/(gradient_dt[dt_step:]-gradient_dt[:-dt_step])
            d2y_dt = gradient_dt[int(dt_step/2): d2ydx2.shape[0]+int(dt_step/2)]
            voltage_inflections = self.cathodic.voltage[cycle_number][maxima]

            try:
                turning_point = maxima[np.argmin(abs(target_voltage-voltage_inflections))]+dt_step
                turning_voltage = self.cathodic.voltage[cycle_number][turning_point]
                reduction_current_idx = np.argwhere(np.sign(self.cathodic.current[cycle_number][:turning_point])==-1).flatten()
                EH_capacity = abs(np.trapz(self.cathodic.current[cycle_number][reduction_current_idx], self.cathodic.time[cycle_number][reduction_current_idx]/60/60))
                EL_capacity = abs(np.trapz(self.cathodic.current[cycle_number][turning_point:], self.cathodic.time[cycle_number][turning_point:]/60/60))
                anodic_turning_voltage = np.argmin(abs(self.anodic.voltage[cycle_number]-turning_voltage))
                anodic_capacity = abs(np.trapz(self.anodic.current[cycle_number][:anodic_turning_voltage],
                                           self.anodic.time[cycle_number][:anodic_turning_voltage]/60/60))

            except:
                print("Skipping cycle "+str(cycle_number))
                EH_capacity = np.nan
                EL_capacity = np.nan
                anodic_capacity = np.nan
                turning_voltage = np.nan
                turning_point = np.nan

            peak_ratio.EH_capacity.append(EH_capacity)
            peak_ratio.EL_capacity.append(EL_capacity)
            peak_ratio._anodic_contribution.append(anodic_capacity)
            peak_ratio._turning_voltage.append(turning_voltage)
            peak_ratio._turning_idx.append(turning_point)
            peak_ratio.EH_capacity_fraction.append(EH_capacity/(EH_capacity+EL_capacity))
            peak_ratio.EL_capacity_fraction.append(EL_capacity/(EH_capacity+EL_capacity))
            peak_ratio.total_capacity.append(EH_capacity+EL_capacity)
            
        
        setattr(self, "peak_ratio", peak_ratio)
        


    def plot_peak_capacity_ratio(self, cycles_to_plot="all", skip_cycles=None, EH_color="gainsboro", EL_color="dimgrey", 
                                 skip_empty_cycles=False):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        y_EL = np.copy(self.peak_ratio.EL_capacity)
        y_EH = np.copy(self.peak_ratio.EH_capacity)

        if cycles_to_plot == "all":
            cycles_to_plot = np.arange(len(y_EL))
        else:
            cycles_to_plot = np.array(cycles_to_plot)

        if type(skip_cycles) != type(None):
            for n in skip_cycles:
                y_EL[n] = np.nan
                y_EH[n] = np.nan

        if skip_empty_cycles==True:
            cycles_to_plot = np.array([value for nvalue, value in enumerate(cycles_to_plot) if all([np.isfinite(self.peak_ratio.EH_capacity[nvalue]),
                                                                                           np.isfinite(self.peak_ratio.EL_capacity[nvalue])])])

        x_pos = np.arange(cycles_to_plot.shape[0])+1
        
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.bar(x=x_pos, height=[y_EL[n] for n in cycles_to_plot],
               color=EL_color, edgecolor="k", label="E$_{L}$")
        ax.bar(x=x_pos, height=[y_EH[n] for n in cycles_to_plot],
               bottom=[y_EL[n] for n in cycles_to_plot], color=EH_color, edgecolor="k", label="E$_{H}$")

        if type(skip_cycles) == type(None):
            skip_cycles = []

        [ax.annotate("{:.0f}".format(self.peak_ratio.EL_capacity_fraction[nvalue]*100),
                     xy=(x_pos[n], 0.5*self.peak_ratio.EL_capacity[nvalue]), ha="center") for n, nvalue in enumerate(cycles_to_plot) if n not in skip_cycles ]

        [ax.annotate("{:.0f}".format(self.peak_ratio.EH_capacity_fraction[nvalue]*100),
                     xy=(x_pos[n], self.peak_ratio.EL_capacity[nvalue]+0.5*self.peak_ratio.EH_capacity[nvalue]), ha="center") for n, nvalue in enumerate(cycles_to_plot) if n not in skip_cycles]

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Capacity at current peak (mAh)")
        ax.legend(loc="upper right")
        return f, ax
