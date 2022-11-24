import re
import numpy as np
import pandas as pd

def mpt_to_df(filename, eis=False):
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
