import pandas as pd
import numpy as np


def normalization(filename:str):
    data = pd.read_csv(filename, header=None)
    new_data = data.iloc[:,:-1].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    newnew = pd.concat([new_data,data.iloc[:,-1]],axis=1)
    newnew.to_csv(filename + 'normal', header=None, columns=None, index=False)

def main():
    normalization("corrected_ok")
    normalization("kddcup.data_10_percent_ok")

if __name__ == "__main__":
    main()