# Requires folder structure data/traingle,square,ellipse,annulus
# 

import numpy as np
from ClassFiles.GeneratedDataset import gen_data_polygons
from ClassFiles.GeneratedDataset import gen_data_round

        
FILEPATH = "./data6"
datasize = 10

gen_data_polygons(FILEPATH,datasize)
gen_data_round(FILEPATH,datasize)
