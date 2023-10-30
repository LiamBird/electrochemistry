def find_plateaus(data, dx_step=4):

    """
    Finds the array indices at the start and end of voltage plateaus from galvanostatic cycling data.
    
    Inputs
    ----------
    data: Cycling data object (see LandData, ArbinData, BiologicData)
        has format: 
        self
           ┢ discharge
           │   ┢ capacity (list of arrays)
           │   └ voltage (list of arrays)
           └ charge
               ┢ capacity (list of arrays)
               └ voltage (list of arrays)
               
    dx_step: int (optional) (default = 4)
        the number of data points over which the gradient is calculated (higher value smooths the data more)
    """
    import numpy as np
    
    n_cycles = np.nanmin([len(data.discharge.capacity), len(data.charge.capacity)])
    plateau_idx = {"EH_start": [],
                   "EH_end": [],
                   "EL_start": [],
                   "EL_end": [],
                   "plateau_start": [],
                   "plateau_end": []}
    
    for cycle_number in range(n_cycles):
        x = data.discharge.capacity[cycle_number]
        y = data.discharge.voltage[cycle_number]

        dydx = (y[-1]-y[0])/(x[-1]-x[0])
        intercept = y[0]-(dydx*x[0])

        y_straight = dydx*x+intercept
        
        difference = y_straight-y
        dd_dx = difference[dx_step:]-difference[:-dx_step]
        x_dx = x[int(dx_step/2):x.shape[0]-int(dx_step/2)]
        
        plateau_begins = np.nonzero((np.sign(dd_dx[:-1])==1) & (np.sign(dd_dx[1:])==-1))[0]+int(dx_step/2)
        plateau_ends = np.nonzero((np.sign(dd_dx[:-1])==-1) & (np.sign(dd_dx[1:])==1))[0]-int(dx_step/2)

        if dd_dx[0] < 0: 
            plateau_begins = np.hstack((np.zeros(1, dtype=int), plateau_begins))
        if dd_dx[-1] < 0:
            plateau_ends = np.hstack((plateau_ends, np.full(dd_dx.shape[0], np.nan, dtype=int)))
            
        if plateau_begins.shape[0]>=1 and plateau_ends.shape[0]>=1:
            if plateau_ends[0] > plateau_begins[0]:
                plateau_idx["EH_start"].append(plateau_begins[0])
                plateau_idx["EH_end"].append(plateau_ends[0])
            else:
                plateau_idx["EH_start"].append(np.nan)
                plateau_idx["EH_end"].append(np.nan)
            if plateau_begins.shape[0]>1 and plateau_ends.shape[0]>1:
                if plateau_ends[1] > plateau_begins[1]:
                    plateau_idx["EL_start"].append(plateau_begins[1])
                    plateau_idx["EL_end"].append(plateau_ends[1])
            else:
                plateau_idx["EL_start"].append(np.nan)
                plateau_idx["EL_end"].append(np.nan)
            for keys in ["plateau_start", "plateau_end"]:
                plateau_idx[keys].append(np.nan)
        else:
            for keys in ["EH_start", "EH_end", "EL_start", "EL_end"]:
                plateau_idx[keys].append(np.nan)
            plateau_idx["plateau_start"].append(plateau_begins[0])
            plateau_idx["plateau_end"].append(plateau_ends[0])
            
    return plateau_idx



class _SubPlateaus:
    def __init__(sub_self):
        keys = ["_start_cap", "_end_cap", "capacity", "capacity_share",
                 "volt_max_err", "volt_min_err", "volt_median"]
        sub_self.__dict__.update([(key, []) for key in keys])  

class Plateaus(object):
    def __init__(self, data, dx_step=10):
        
        """
        Analyses the voltage plateaus from an ArbinData, LandData, or BiologicGalv object
        
        Returns object with two subclasses (EH for 2.4V plateau, EL for 2.1 V plateau)
        
        Example:
            plateau = Plateaus(data_object, dx_step=4)
            
            To access numerical data:
            plateau.EH.capacity = [100, 99, 99, 98...]
            
            To plot capacity shares:
            f, ax = plateau.plot_capacity_share()
            
            To plot voltage ranges:
            f, ax = plateau.plot_voltage_ranges()
        
        
        Inputs
        ----------
        data: ArbinData, LandData, or BiologicGalv object
        dx_step: int (default value = 10)
            Modifies the ranges of values used for gradient calculation
            Larger values reduce the sensitivity to short plateaus
            Recommended to set to higher values for shorter time interval data to reduce the noise
            
        Attributes
        ----------
        Two plateau objects (EH for 2.4V plateau, EL for 2.1 V plateau), each with attributes:
        
        capacity: array
            The capacity (in mAh or mAh/g, depending on input units) realised at each plateau
        capacity_share: array
            The percentage of the cycle's capacity realised at each plateau
        volt_median: array
            The median voltage of each plateau
        volt_max_err: array
            The maximum voltage of each plateau
        volt_min_err: array
            The minimum voltage of each plateau
        
        Methods
        ----------
        plot_capacity_share(self, EH_color="black", EL_color="white", EH_marker="o", EL_marker="d",
                            figsize=(8, 6), max_cycles=None, ideal_lines=True)
        
        plot_voltage_ranges(self, figsize=(8, 6), max_cycles=None, ideal_lines=True, 
                                EH_color="black", EL_color="grey")
        
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.rcParams.update({"font.size": 12})
        plt.rcParams.update({"xtick.direction" : "in"})
        plt.rcParams.update({"xtick.top" : True})
        plt.rcParams.update({"ytick.direction" : "in"})
        plt.rcParams.update({"ytick.right" : True})
        
        self._data = data
        self._plateaus = find_plateaus(data, dx_step=dx_step)
                
        for name in ["EH", "EL"]:
            setattr(self, name, _SubPlateaus())
            
            for cycle in range(len(data.discharge.summary_capacity)):
                try:
                    vars(self)[name]._start_cap.append(data.discharge.capacity[cycle][self._plateaus[name+"_start"][cycle]])
                    vars(self)[name]._end_cap.append(data.discharge.capacity[cycle][self._plateaus[name+"_end"][cycle]])
                    cycle_capacity = data.discharge.capacity[cycle][self._plateaus[name+"_end"][cycle]]-data.discharge.capacity[cycle][self._plateaus[name+"_start"][cycle]]
                    vars(self)[name].capacity.append(cycle_capacity)
                    vars(self)[name].capacity_share.append(cycle_capacity/data.discharge.summary_capacity[cycle])
                    
                    voltages = np.array(data.discharge.voltage[cycle][np.arange(self._plateaus[name+"_start"][cycle],
                                                                                self._plateaus[name+"_end"][cycle]
                                                                                 )])
                    vars(self)[name].volt_max_err.append(np.nanmax(voltages)-np.nanmedian(voltages))
                    vars(self)[name].volt_min_err.append(np.nanmedian(voltages)-np.nanmin(voltages))
                    vars(self)[name].volt_median.append(np.nanmedian(voltages))
                                        
                except:
                    for key in ["_start_cap", "_end_cap", "capacity", "capacity_share"]:
                        vars(vars(self)[name])[key].append(np.nan)
                        
            for key in vars(vars(self)[name]).keys():
                vars(vars(self)[name])[key] = np.array(vars(vars(self)[name])[key])
            

    def plot_capacity_share(self, EH_color="black", EL_color="white", EH_marker="o", EL_marker="d",
                             figsize=(8, 6), max_cycles=None, ideal_lines=True):
        
        """
        Generates a graph of capacity share (returns f, ax - attributes of ax can be set)
        
        Inputs
        ----------
        EH_color: color of the EH markers (mfc). Default=black
        EL_color: color of the EL markers (mfc). Default=white (with black edge)
        EH_marker: marker for EH markers. Default="o"
        EL_marker: marker for EL markers. Default="d"
        figsize: Size of the figure, inches. Default=(8, 6)
            Note that the text size can be changed using (replace <int> with required value):
            plt.rcParams.update({"font.size": <int>})
        max_cycles: the number of cycles to plot (e.g. set for consistently plotting 100 cycles for all samples)
        ideal_lines: includes lines showing ideal values (default=True)
                    EH: 25%
                    EL: 75%
        """
        import matplotlib.pyplot as plt
        
        f, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.EH.capacity_share[:max_cycles]*100, mfc=EH_color, mec="k", marker=EH_marker, ls="none", label="E$_{H}$")
        ax.plot(self.EL.capacity_share[:max_cycles]*100, mfc=EL_color, mec="k", marker=EL_marker, ls="none", label="E$_{L}$")
        ax.legend()
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Capacity share (%)")
        
        if ideal_lines==True:
            ax.axhline(25, color="k", ls="-")
            ax.axhline(75, color="k", ls=":")
            ax.set_ylim([0, 100])
            
        ax.set_xlim([0, None])
        
        return f, ax
    
    def plot_voltage_ranges(self, figsize=(8, 6), max_cycles=None, ideal_lines=True, 
                            EH_color="black", EL_color="grey"):
        """
        Generates a graph of voltage ranges associated with each plateau (returns f, ax - attributes of ax can be set)
        
        Inputs
        ----------
        EH_color: color of the EH lines. Default=black
        EL_color: color of the EL lines. Default=grey
        figsize: Size of the figure, inches. Default=(8, 6)
            Note that the text size can be changed using (replace <int> with required value):
            plt.rcParams.update({"font.size": <int>})
        max_cycles: the number of cycles to plot (e.g. set for consistently plotting 100 cycles for all samples)
        ideal_lines: includes lines showing ideal values (default=True)
                    EH: 2.4
                    EL: 2.1
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        f, ax = plt.subplots(figsize=figsize)
        ax.errorbar(x=np.arange(self.EH.volt_median[:max_cycles].shape[0]),
                    y=self.EH.volt_median[:max_cycles],
                    yerr=np.vstack((self.EH.volt_min_err[:max_cycles].flatten(),
                                    self.EH.volt_max_err[:max_cycles].flatten())),
                    ls="none", ecolor=EH_color, label="E$_{H}$")
        ax.errorbar(x=np.arange(self.EL.volt_median[:max_cycles].shape[0]),
                    y=self.EL.volt_median[:max_cycles],
                    yerr=np.vstack((self.EL.volt_min_err[:max_cycles].flatten(),
                                    self.EL.volt_max_err[:max_cycles].flatten())), ls="none", ecolor=EL_color,
                    label="E$_{L}$")
        
        ax.legend()
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Plateau voltage (V vs. Li/ Li$^{+}$)")
    
        if ideal_lines == True:
            ax.axhline(2.1, color=EL_color, ls=":")
            ax.axhline(2.4, color=EH_color)
            
        ax.set_xlim([0, None])
            
        return f, ax
