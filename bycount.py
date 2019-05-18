
import pandas as pd
import numpy as np

items = ['beauty', 'fashion', 'mobile']
rdf = pd.DataFrame()
dictz = {
'beauty':{'Benefits':3, 'Brand':3, 'Colour_group':3,
       'Product_texture':3, 'Skin_type':3},
'fashion':{'Pattern':3, 'Collar Type':3,
       'Fashion Trend':3, 'Clothing Material':3, 'Sleeves':3},
'mobile':{'Operating System':3, 'Features':3,
       'Network Connections':3, 'Memory RAM':3, 'Brand':3, 'Warranty Period':3,
       'Storage Capacity':3, 'Color Family':3, 'Phone Model':3, 'Camera':3,
       'Phone Screen Size':3}
}

for x in items:
    df = pd.read_csv(x+"_data_info_train_competition.csv")
    eval = pd.read_csv(x+"_new.csv")
    for i in df.columns[3:]:
        column = i
        num = dictz[x][i]
        value = df.groupby(column)['itemid'].nunique().nlargest(n=num).to_frame().reset_index()[column]
        # count = df.groupby(column)['itemid'].nunique().sort_values(ascending = False)
        # print(count)
        # print(value[num-2], value[num-1])

        index = eval['itemid'].astype(str)+"_"+str(column)
        r = pd.DataFrame(dict(itemid = index, r1 = int(value[num-2]), r2 = int(value[num-1])))
        rdf = pd.concat([rdf,r])

rdf.to_csv("new_countvar.csv", index = False)
 
