class _CycleData(object):
    def __init__(self):
        for keys in ["capacity", "voltage", "time", "summary_capacity"]:
            self.__dict__.update([(keys, [])])

class BiologicGalv(object):
    """
    Class for loading and displaying Galvanostatic data from Biologic cyclers
    Requires mpt_to_df.py in same directory
    
    Inputs
    ----------
    filename (str):
        The .mpt file (including its path)
        
    mass (float OR "ignore") (optional, default=None):
        Mass of active material in g (if entering in mg, set scale_mass to True)
        If mass == None, mass is read from header of Biologic .mpt file (if available, otherwise set to 1)
        If mass == "ignore", mass is set to 1
        Where mass is not set (i.e. mass defaults to '1'), graph outputs will be shown in mAh
        Where mass is set, graph outputs will be shown in mAh/g
        
    theoretical_capacity (float) (default=1675):
        The theoretical capacity of the cell in mAh/g
        
    rate_change (bool) (optional, default=False):
        If multiple CRates are present in the cycling (e.g. 0.1, 0.2, 0.5...), splits the data into different CRates and enables attributes for labels in plotting
    
    resume_label (str) (optional, default = " "):
        For use with rate change data with repeated CRate, e.g. (0.1, 0.2, 0.5, 0.1). Resume label will be appended to repeated CRates for unique dictionary keys
        
    truncate_at (int) (optional, default=None):
        Last cycle to include in the data (e.g. to avoid final failed cycle affecting capacities)
        
    scale_mass (bool) (optional, default=False):
        mass argument entered in g by default. To scale to mg, enter scale_mass=True
        
        
    Attributes
    ----------
    n_cycles (int): 
        Number of cycles for which discharge and charge data is available
        
    mass (float):
        Mass (in g) used to calculate gravimetric capacity 
    
    charge AND discharge (_CycleData objects):
        ┣ capacity: array of arrays of capacity data (one array per cycle)
        ┃           units: mAh/g (if mass set) or mAh (if mass not set)
        ┣ voltage: array of arrays of voltage data (one array per cycle)
        ┣ time: array of arrays of times for data collection after onset of cycling (cumulative)
        ┃           units: seconds (cumulative)
        └ summary_capacity: final capacity at the end of the cycle
                            units: mAh/g (if mass set) or mAh (if mass not set)
                            
    Methods
    ----------
    plot_capacities(y_max=1675):
        Plots the summary_capacity data as a function of cycle number
        
    plot_capacity_voltage(cycles_to_plot="all", formation_cycles=2)
        Plots the discharge/ charge curves for the specified cycles
    """
    def __init__(self, filename, mass=None, theoretical_capacity=1675, rate_change=False, resume_label = " ", truncate_at=None, scale_mass=False):
        from mpt_to_df import mpt_to_df
        import matplotlib.pyplot as plt
        import numpy as np
        import re
        
        import warnings
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        
        self._updated = "07.10.2023"
        self._update_record = ["25.08.2023: Prints cycle where import processing failed rather than terminating and failing",
                                "06.09.2023: if Battery capacity not present in .mpt, sets mass=1 by default rather than terminating and failing",
                              "07.10.2023: Added 'scale mass' boolean argument. If True, capacity is divided by 1.675; if False (default), capacity is divided by 1675. Where True, assumes capacity provided in mAh (i.e. mass returned in mg)"]
        
        
        plt.rcParams.update({"xtick.direction": "in"})
        plt.rcParams.update({"ytick.direction": "in"})
        plt.rcParams.update({"xtick.top": True})
        plt.rcParams.update({"ytick.right": True})
        self._data_file = mpt_to_df(filename)
        
        if mass == None:
            try:            
                self._mass_set = False
                with open(filename) as f:
                    for line in f.readlines():
                        if "Battery capacity" in line:
                            capacity_read = float(re.findall("\d+.\d+", line)[0])
                            if scale_mass == False:
                                self.mass = capacity_read/theoretical_capacity
                            else:
                                self.mass = capacity_read/(theoretical_capacity/1000)
                            self._mass_set = True
                if self._mass_set == False:
                    self.mass = 1
            except:
                self.mass = 1
                self._mass_set = False
        elif mass == "ignore":
            self.mass = 1
            self._mass_set = False
        else:
            self.mass = mass
            self._mass_set = True
            
        if "I/mA" in self._data_file.columns:
            current_header = "I/mA"
        elif "<I>/mA" in self._data_file.columns:
            current_header = "<I>/mA"
            
        if "Ecell/V" in self._data_file.columns:
            voltage_header = "Ecell/V"
        elif "Ewe/V" in self._data_file.columns:
            voltage_header = "Ewe/V"
            
        current = np.array(self._data_file[current_header])
        capacity = np.array(self._data_file["Capacity/mA.h"])/self.mass
        voltage = np.array(self._data_file[voltage_header])
        time = np.array(self._data_file["time/s"])
        
        charge_idx = np.argwhere(np.sign(current)==1).flatten()
        discharge_idx = np.argwhere(np.sign(current)==-1).flatten()
        
        self._charge_end_idx = charge_idx[np.argwhere(charge_idx[1:]-charge_idx[:-1]!=1).flatten()]
        self._discharge_end_idx = discharge_idx[np.argwhere(discharge_idx[1:]-discharge_idx[:-1]!=1).flatten()]
        
        if max(self._discharge_end_idx) > max(self._charge_end_idx):
            last_cycle = "discharge"
        else:
            last_cycle = "charge"
            
        self.n_cycles = np.min([self._discharge_end_idx.shape[0],
                                self._charge_end_idx.shape[0]])
            
        self._charge_end_idx = np.array(np.hstack((np.zeros(1), self._charge_end_idx)), dtype=int)
        self.discharge = _CycleData()
        self.charge = _CycleData()
        
        ### Truncate at added 25.03.2023
        if truncate_at != None:
            stop_at = truncate_at
        else:
            stop_at = self.n_cycles
        
        for cyc in range(stop_at):
            if cyc < self._charge_end_idx.shape[0] and cyc < self._discharge_end_idx.shape[0]:
                try:
                    self.discharge.capacity.append(capacity[self._charge_end_idx[cyc]+1: self._discharge_end_idx[cyc]])
                    self.discharge.voltage.append(voltage[self._charge_end_idx[cyc]+1: self._discharge_end_idx[cyc]])
                    self.discharge.time.append(time[self._charge_end_idx[cyc]+1: self._discharge_end_idx[cyc]])
                    self.discharge.summary_capacity.append(np.max(capacity[self._charge_end_idx[cyc]+1: self._discharge_end_idx[cyc]]))
                except:
                    print("Failed at discharge {}".format(cyc))
                    
            if cyc < self._charge_end_idx.shape[0]-1 and cyc < self._discharge_end_idx.shape[0]:
                try:
                    self.charge.voltage.append(voltage[self._discharge_end_idx[cyc]+1: self._charge_end_idx[cyc+1]])
                    self.charge.capacity.append(capacity[self._discharge_end_idx[cyc]+1: self._charge_end_idx[cyc+1]])
                    self.charge.time.append(time[self._discharge_end_idx[cyc]+1: self._charge_end_idx[cyc+1]])
                    self.charge.summary_capacity.append(np.max(capacity[self._discharge_end_idx[cyc]+1: self._charge_end_idx[cyc+1]]))
                except:
                    print("Failed at charge {}".format(cyc))
                
                
        for cycle_type in ["discharge", "charge"]:
            for varname in vars(vars(self)[cycle_type]).keys():
                if "capacity" in varname:
                    vars(vars(self)[cycle_type])[varname] = np.array(vars(vars(self)[cycle_type])[varname], dtype=object)
                else:
                    vars(vars(self)[cycle_type])[varname] = np.array(vars(vars(self)[cycle_type])[varname], dtype=object)
        
        if rate_change==True:
            data_read = []
            import re

#             filename = 'CellData\\155_c60s40\\rate_change\\LRB_155_60C_40S_cell1_CO3.mpt'
            with open(filename) as f:
                for nline, line in enumerate(f.readlines()):
                    data_read.append(line)

            for nline, line in enumerate(data_read): 
                if "ctrl_type" in line:
                    ctrl_type = [seg for seg in line.split(" ") if len(seg)>0][1:-1]
                if "ctrl1_val " in line:
                    n_repeats = [re.findall("\d+.\d+", seg)[0] for seg in data_read[nline].split(" ") if len(re.findall("\d+.\d+", seg))>0]
                if "ctrl3_val_vs" in line:
                    CRates_read = [re.findall("\d+.\d+", seg)[0] for seg in data_read[nline+1].split(" ") if len(re.findall("\d+.\d+", seg))>0]

            CRates_ordered = [CRates_read[nname][:CRates_read[nname].index(".")] for nname, name in enumerate(ctrl_type) if ctrl_type[nname-1]!="CC" and name!="Rest"]        
            for nname, name in enumerate(CRates_ordered):
                if name in (CRates_ordered[:nname]):
                    CRates_ordered[nname] = "{}{}".format(name, resume_label)

            n_repeats = [n_repeats[nname] for nname, name in enumerate(ctrl_type) if ctrl_type[nname-1]!="CC" and name!="Rest"]

            self.CRates_cycles = dict([(CRates_ordered[n], int(float(n_repeats[n]))) for n in range(len(n_repeats))])
            self.CRate_change_cycles = [int(np.sum([*self.CRates_cycles.values()][:n])+n) for n in range(len(self.CRates_cycles)+1)]
            
    def plot_capacities(self, y_max=1675):
        """
        Plots the discharge and charge capacities as a function of cycle number
        
        Inputs
        ----------
        y_max (float) (optional, default=1675):
            the upper y limit of the plot
            
        Returns
        ----------
        fig, ax (figure, subplots):
            Enables control of properties after plotting
            Example:
            f, ax = self.plot_capacities()
            ax.legend(loc="upper left")
            
        """
        import matplotlib.pyplot as plt
        import numpy as np
        f, ax = plt.subplots()
        ax.plot(self.discharge.summary_capacity, "o", mfc="k", mec="k", label="Discharge")
        ax.plot(self.charge.summary_capacity, "o", mec="k", mfc="none", label="Charge")
        ax.legend()
        ax.set_ylim([0, y_max])
        ax.set_xlabel("Cycle number")
        if self._mass_set == True:
            ax.set_ylabel("Gravimetric capacity (mAh/g)")
        else:
            ax.set_ylabel("Capacity (mAh)")
        return f, ax
            
    def plot_capacity_voltage(self, cycles_to_plot="all", formation_cycles=2):
        """
        Plots charge/ discharge curves for specified cycles, differentiated by colour

        Inputs
        ----------
        cycles_to_plot : (str or list of integers) (optional, default="all")
            If "all", plots all the available data
            If list of integers (e.g. [0, 1, 10, 20, 30...]) plots the specified cycles if available
        formation_cycles : (int) (optional, default=2)
            If slow formation cycles are used, shows these curves as dotted lines
            Set to 0 if no formation cycles are used. 

        Returns
        ----------
        fig, ax (figure, subplots):
            Enables control of properties after plotting
            Example:
            f, ax = self.plot_capacities()
            ax.legend(loc="upper left")

        """
        import matplotlib.pyplot as plt
        import numpy as np
        f, ax = plt.subplots()
        
        if cycles_to_plot == "all":
            cycles_to_plot = range(self.n_cycles)
        else:
            cycles_to_plot = np.array(cycles_to_plot)-1
        
        color_map = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(cycles_to_plot)-formation_cycles)]
        
        for ncyc, cyc in enumerate(cycles_to_plot):
            if cyc < formation_cycles:
                d, = ax.plot(self.discharge.capacity[cyc], self.discharge.voltage[cyc], ls=":", label=cyc+1, color="gray")
                c, = ax.plot(self.charge.capacity[cyc], self.charge.voltage[cyc], ls=":", color=d.get_color())
            else:
                d, = ax.plot(self.discharge.capacity[cyc], self.discharge.voltage[cyc], label=cyc+1, color=color_map[ncyc-formation_cycles])
                c, = ax.plot(self.charge.capacity[cyc], self.charge.voltage[cyc], color=d.get_color())
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if self._mass_set == True:
            ax.set_xlabel("Gravimetric capacity (mAh/g)")
        else:
            ax.set_xlabel("Capacity (mAh)")
        ax.set_ylabel("Voltage vs. Li/ Li$^{+}$(V)")
        return f, ax
        
