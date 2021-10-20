import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt

Data = {
    'Date':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'Demand':[37, 60, 85, 112, 132, 145, 179, 198, 150, 132],
    'Forecast':[0 ,0 ,0 ,61, 86, 110, 130, 152, 174, 176, 160, 160, 160]
}
print(Data)