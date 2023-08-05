import numpy as np
import pandas as pd
import os
import glob
path="/home/ashish/Desktop/projectcs668/doh/testing/input/"

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv"))), ignore_index= True)
df.to_csv(path+"final.csv",index=False)
