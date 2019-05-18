import pandas as pd
import glob
#path = r'C:\Users\KangYu\Downloads\ndsc-advanced\ans' # use your path

path = r'C:\Users\KangYu\Desktop\nationaldata\chongyan' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)


frame = pd.concat(li, axis=0, ignore_index=True)

frame = frame.sort_values("id")

frame = frame.rename(columns={'id': 'id', 'tagging': 'tagging'})
frame.to_csv('chongyan.csv', index = False)
