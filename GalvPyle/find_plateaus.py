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
            plateau_ends = np.hstack((plateau_ends, np.full(dd_dx.shape[0], dtype=int)))
            
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
