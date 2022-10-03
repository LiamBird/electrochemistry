"""
ArbinData
Uploaded 03/10/2022

Imports data from .xls files output from Arbin plugin for Excel
"""

def _make_cycle(charge_discharge, dataframe):
  """
  Accessed by ArbinData class
  Finds the end of charge/ discharge cycle capacities from galvanostatic cycling data
  """
    import numpy as np
    if charge_discharge == "discharge":
        column = "Discharge_Capacity(Ah)"
    elif charge_discharge == "charge":
        column = "Charge_Capacity(Ah)"
        
    all_capacities = np.array(dataframe[column])
    dQ = all_capacities[1:]-all_capacities[:-1]
    idx = np.argwhere(np.sign(dQ)==1).flatten()
    didx = idx[1:]-idx[:-1]
    ends_of_cycles = np.array(np.hstack((np.zeros(1),
                                         np.argwhere(didx!=1)[:, 0])), dtype=int)
    
    voltages = [np.array(dataframe["Voltage(V)"][idx][ends_of_cycles[cycle]+1:ends_of_cycles[cycle+1]])
                for cycle in range(ends_of_cycles.shape[0]-1)]
    
    capacities = [all_capacities[idx][ends_of_cycles[cycle]+1:
                                      ends_of_cycles[cycle+1]] for cycle in range(ends_of_cycles.shape[0]-1)]
    
    summary_capacities = all_capacities[idx][ends_of_cycles].flatten()
    
    class _CycleSummary(object):
        def __init__(self):
          """
          Subclass for Arbin data to contain galvanostatic cycling data. 
          Contains the experimental data for either the discharge or the charge sections of galvanostatic cycles.
          
          Parameters
          ----------
          none
          
          Attributes
          ----------
          voltages: list
            A list of arrays containing the measured voltages (V vs. Li/ Li+), with one array per cycle (array shapes may vary depending on cell capacity)
          capacities: list
            A list of arrays containing the measured capacities (mAh or mAh/g, array shapes may vary depending on cell capacity)
          summary_capacities: array
            Array containing the capacity at the end of each cycle step (mAh or mAh/g)
          capacity_units: str
            Where present, specifies whether capacities are in mAh or mAh/g (mAh/g requires sample mass to be specified when ArbinData class instantiated) 
          cycle_numbers: array
            Sequence of numbers for cycle numbers (1, 2, 3... N), useful for plotting cycle_numbers vs. summary_capacities
          capacity_retention: array
            Percentage of capacity retained compared to maximum capacity attained for specified cycle type (charge/ discharge)
          """
            self.voltages = voltages
            self.capacities = capacities
            self.summary_capacities = summary_capacities
            self.capacity_units = None
            self.cycle_numbers = np.arange(1, summary_capacities.shape[0])
            self.capacity_retention = None
    return _CycleSummary()


class ArbinData(object):
    def __init__(self, filename, sample_mass=None, cumulative=True):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        if sample_mass == None:
            self._sample_mass = 1000
            capacity_units = "mAh"
        else:
            self._sample_mass = sample_mass
            capacity_units = "mAh/g"
        
        data_file = pd.ExcelFile(filename)
        data = data_file.parse(sheet_name=[name for name in data_file.sheet_names if "Channel" in name][0])
        
        self.discharge = _make_cycle("discharge", data)
        self.charge = _make_cycle("charge", data)
        
        if cumulative == True:
            for state in ["discharge", "charge"]:
                adj_capacities = [vars(self)[state].capacities[0]]
                for n in vars(self)[state].cycle_numbers[:-1]:
                    adj_capacities.append(vars(self)[state].capacities[n]-vars(self)[state].summary_capacities[n])
                vars(self)[state].capacities = adj_capacities
                vars(self)[state].summary_capacities = vars(self)[state].summary_capacities[1:]-vars(self)[state].summary_capacities[:-1]
        else:
            for state in ["discharge", "charge"]:
                vars(self)[state].summary_capacities = vars(self)[state].summary_capacities[1:]
        
        for state in ["discharge", "charge"]:
            vars(self)[state].summary_capacities = vars(self)[state].summary_capacities*1000/(self._sample_mass/1000)
            vars(self)[state].capacities = [cycle_data*1000/(self._sample_mass/1000) for cycle_data in vars(self)[state].capacities]
            vars(self)[state].capacity_units = capacity_units
            vars(self)[state].capacity_retention = [cycle_capacity/vars(self)[state].summary_capacities[0] for cycle_capacity in vars(self)[state].summary_capacities]
            
        self.number_of_cycles = np.min([len(self.discharge.capacities), len(self.charge.capacities)])
        
    def plot_capacities(self, y_max=None, y_div_major=None, y_div_minor=None, y_formatter="%d",
                              x_max=None, x_div_major=10, x_div_minor=1):
        """
        Alternative formatters: "%.2f", "%d"
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

        f, ax = plt.subplots()
        ax.plot(self.discharge.cycle_numbers, self.discharge.summary_capacities, "o", mfc="k", mec="k", label="Discharge")
        ax.plot(self.charge.cycle_numbers, self.charge.summary_capacities, "o", mfc="white", mec="k", label="Charge")
        ax.legend()
        ax.tick_params(which="both", tickdir="in", top=True, right=True)
        ax.set_ylabel("Capacity ({})".format(self.discharge.capacity_units))
        ax.set_xlabel("Cycle number")
        
        ax.set_ylim([0, y_max])
        ax.set_xlim([0, x_max])
        
        ax.xaxis.set_major_locator(MultipleLocator(x_div_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_div_minor))
        
        if y_max == None: 
            y_max = np.max([np.max(self.charge.summary_capacities),
                                             np.max(self.discharge.summary_capacities)])
        elif y_max != None:
            y_max = y_max
            
        if y_div_major == None:
            ax.set_yticks(np.linspace(0, y_max , 11))
            ax.yaxis.set_major_formatter(FormatStrFormatter(y_formatter))
        elif y_div_major != None:
            ax.yaxis.set_major_locator(MultipleLocator(y_div_major))
        
        if y_div_minor != None:
            ax.yaxis.set_minor_locator(MultipleLocator(y_div_minor))
        elif y_div_minor == None:
            pass
        
        return f, ax
        

    def plot_capacity_voltage(self, cycles_to_plot="all", cmap="viridis", 
                              y_min=None, y_max=None, y_div_major=None, y_div_minor=None, y_formatter="%d",
                              x_max=None, x_div_major=50, x_div_minor=10):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                       AutoMinorLocator)
        
        f, ax = plt.subplots()
        if cycles_to_plot == "all":
            cycles = list(np.arange(1, self.number_of_cycles+1, dtype=int))
        else:
            cycles = list(cycles_to_plot)
            
        ax.plot([], [], ls="none", lw=0, label="Cycle number")
            
        ax.set_prop_cycle("color", [vars(plt.cm)[cmap](i) for i in np.linspace(0, 1, len(cycles))])
        for cyc in cycles:
            ax.plot(self.discharge.capacities[cyc-1], self.discharge.voltages[cyc-1], label=cyc)
        for cyc in cycles:
            ax.plot(self.charge.capacities[cyc-1], self.charge.voltages[cyc-1])
            
        ax.legend()
        
        ax.tick_params(which="both", tickdir="in", top=True, right=True)
        ax.set_xlabel("Capacity ({})".format(self.discharge.capacity_units))
        ax.set_ylabel("Voltage vs. Li/ Li$^{+}$")
        
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([0, x_max])
        
        for n, major_divs in enumerate([x_div_major, y_div_major]):
            axes = ["xaxis", "yaxis"]
            if major_divs != None:
                vars(ax)[axes[n]].set_major_locator(MultipleLocator(major_divs))
            elif major_divs == None:
                pass
        
        for n, minor_divs in enumerate([x_div_minor, y_div_minor]):
            axes = ["xaxis", "yaxis"]
            if minor_divs != None:
                vars(ax)[axes[n]].set_minor_locator(MultipleLocator(minor_divs))
            elif minor_divs == None:
                pass
        
        return f, ax
    
    def dQdV(self, charge_discharge, cycle_number, d_step=1, min_volt_change=0):
        dV = vars(self)[charge_discharge].voltages[cycle_number-1][d_step:]-vars(self)[charge_discharge].voltages[cycle_number-1][:-d_step]
        dQ = vars(self)[charge_discharge].capacities[cycle_number-1][d_step:]-vars(self)[charge_discharge].capacities[cycle_number-1][:-d_step]

        dQdV = []
        for i in range(dQ.shape[0]):
            if dV[i] == 0:
                dQdV.append(np.nan)
            elif abs(dV[i]) < min_volt_change:
                dQdV.append(np.nan)
            else:
                dQdV.append(dQ[i]/dV[i])
        return vars(self)[charge_discharge].voltages[cycle_number-1][d_step:], dQdV
