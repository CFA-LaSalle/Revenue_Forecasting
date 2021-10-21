import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt

def moviming_average(d, periods=1, n=3):
    #historical period length
    col = len(d)
    #covering future demand with nan
    d = np.append(d, [np.nan]*periods)
    #forecast array
    f = np.full(col + periods, np.nan)

    #creating t+1 for all historical period
    for t in range(n, col):
        f[t] = np.mean(d[t-n:t])

    #forecast for all extra periods
    f[t+1] = np.mean(d[t-n+1:t+1])

    #returning forecasting as dataframe with demand, forecast and error
    forecast = pd.DataFrame.from_dict({'Demand':d, 'Forecast':f, 'Error':d-f})

    return forecast

def kpi(data):
    dem_ave = data.loc[data['Error'].notnull(), 'Demand'].mean()
    bias_abs = data['Error'].mean()
    bias_rel = bias_abs / dem_ave
    print ('Bias: {:0.2f}, {:.2%}'.format(bias_abs, bias_rel))

def mape(data):
    MAPE = (data['Error'].abs() / data['Demand']).mean()
    print('MAPE: {:.2%}'.format(MAPE))

def mae(data):
    dem_ave = data.loc[data['Error'].notnull(), 'Demand'].mean()
    MAE_abs = data['Error'].abs().mean()
    MAE_rel = MAE_abs / dem_ave
    print('MAE: {:0.2f}, {:.2%}'.format(MAE_abs, MAE_rel))

def rmse(data):
    dem_ave = data.loc[data['Error'].notnull(), 'Demand'].mean()
    RMSE_abs = np.sqrt((data['Error']**2).mean())
    RMSE_rel = RMSE_abs / dem_ave
    print('RMSE: {:0.2f}, {:.2%}'.format(RMSE_abs, RMSE_rel))

Data = [28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]
forecast = moviming_average(Data,1,3)
forecast[['Demand', 'Forecast']].plot()

#KPI Key performance indicator
kpi = kpi(forecast)

#MAPE Mean absolute percentage error, never used
Mape = mape(forecast)

#MAE Mean Absolute Error, MAE = 10 < Ave_demand = 1000, ashonishing
#MAE is amiming at demand median
#Minimizing it will result in a bias
Mae = mae(forecast)

#RMSE Root Mean Square Error
#RMSE is aiming at demand average
#Minimizing it will not result in bias
Rmse = rmse(forecast)

#in SES, a (alpha) exists, it its a ratio of how much importance
#the model will allocate to the most recent observation
#compared to the importance of demand history

#in ES the weight given to each demand observation is exponentially
#reduced

#if alpha is high, the model will allocate more importance
#to the most recent demand observation, the model will learn fast
#it will be reactive to a charge in the demand level
#but it will also be sensitive to outliers and noise

#if alpha es low, the model wont rapidly notice a change in
#level, but will also not overreact to noise and outliers

#a rasonable range for a is between 0.05 and 0.5
#if a is higher than 0.5 it means that the model is allocating
#nearly no importance to demand history
#the forecast will almost solely be based on the lastest observation
#that would be a hint that something is wrong with the model

#f0 = d0
#first forecast as the fist demand observation

#data leakage
#an initialization with multiple periods ahead
#-if you define the initial forecast as the average of the first five periods
#you face a data leakage

def ses(d, extra_periods, alpha):
    #historial period length
    col = len(d)
    #append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods)

    #forecast array
    f = np.full(col+extra_periods, np.nan)
    #initialization of first forecast
    f[1] = d[0]

    #create all the t+1 forecast until end of historical period
    for t in range(2, col+1):
        f[t] = alpha*d[t-1]+(1-alpha)*f[t-1]

    #forecast for all extra periods
    for t in range(col+1, col+extra_periods):
        #update the forecast as the previous forecast
        f[t] = f[t-1]

    ses = pd.DataFrame.from_dict({'Demand':d, 'Forecast':f, 'Error':d-f})

    return ses

SES = SES(Data, 1, 0.4)
SES_Bias = kpi(SES)
SES_MAPE = MAPE(SES)
SES_MAE = MAE(SES)
SES_RMSE = RMSE(SES)

SES.index.name = 'Period'
SES[['Demand', 'Forecast']].plot(
    figsize=(8,3),
    title='Simple Smoothing',
    xlim=(0, 25),
    ylim=(0, 30),
    style=['-', '--']
)

#a model is underfitted when it doesnt explain reality accuralety enough
#we divide the data set in two parts
#Training set
#Test Set

#Training Set
#Is used to train (fit) our model (optimize its parameters)

#Test Set
#The dataset that will assess the accuracy of our model agaisnt
#unseen data
#This dataset is kept aside from the model during its training phase
#So that the model is not aware of this data and can thus be tested agaisnt unseen data

#we never use the test set to optimize our model
#if we optimize our model on the test set
#we will never know what accuracy we can expect agaisnt new demand

#ft∗+λ = at∗ + λbt∗ for any future forecast in DES

#for initialize a0 and b0, we use a low number for n (e.g. 3, 5)
#n could be the average of 1/b and 1/a
#for Linear regression np.polyfit

def des(d, periods, alpha, beta):
    col = len(d)
    d = np.append(d, [np.nan]*periods)
    f, a, b = np.full((3, col+periods), np.nan)

    #initialization
    a[0] = d[0]
    b[0] = d[1] - d[0]

    #create all the t+1 forecast
    for t in range(1, col):
        f[t] = a[t-1] + b[t-1]
        a[t] = alpha*d[t] + (1-alpha)*(a[t-1]+b[t-1])
        b[t] = beta*(a[t]-a[t-1]) + (1-beta)*b[t-1]

    #forecast for all extra periods
    for t in range(col, col+periods):
        f[t] = a[t-1] + b[t-1]
        a[t] = f[t]
        b[t] = b[t-1]

    des = pd.DataFrame.from_dict({'Demand':d, 'Forecast':f, 'Level':a, 'Trend':b, 'Error':d-f})

    return des

Des = des(Data, 1, 0.8, 0.7)
DES_Bias = kpi(Des)
DES_MAPE = mape(Des)
DES_MAE = mae(Des)
DES_RMSE = rmse(Des)

Des.index.name = 'Period'
Des[['Demand', 'Forecast', 'Trend']].plot(
    figsize=(8,3),
    title='Double Smoothing',
    xlim=(0, 25),
    ylim=(0, 30),
    style=['-', '--','--']
)

#overreact to the initial trend a=0.4 b=0.4

def exp_smooth_opti(d, periods=6):
    params = [] #contains all the different parameter sets
    kpis = [] #contains the results of each model
    forecastframe = [] #contains all the dataframes returned by different models

    for alpha in np.linspace(0.05, 0.6, 100):

        forecast = ses(d, extra_periods=periods, alpha=alpha)
        params.append('Simple Smoothing, alpha: {}'.format(alpha))
        forecastframe.append(forecast)
        MAE = forecast['Error'].abs().mean()
        kpis.append(MAE)

        for beta in np.linspace(0.05, 0.6, 100):

            forecast = des(d, periods=periods, alpha=alpha, beta=beta)
            params.append('Double Smoothing alpha: {}, beta: {}'.format(alpha, beta))
            forecastframe.append(forecast)
            MAE = forecast['Error'].abs().mean()
            kpis.append(MAE)

    mini = np.argmin(kpis)
    print(f'Optimal solution found for {params[mini]} MAE of', round(kpis[mini], 2))

    return forecastframe[mini]

opt = exp_smooth_opti(Data)

opt.index.name = 'Period'
opt[['Demand', 'Forecast']].plot(
    figsize=(8,3),
    title='Best Model Found',
    xlim=(0, 25),
    ylim=(0, 30),
    style=['-', '--','--']
)

#we can change the MAE to RSME for best forecasting model
#a beginner mistake is to allow a very wide range (0,1) for a,b
#a reasonable range for a,b is between 0.05 and 0.6
#a value above 0.6 means that the model is allocating nearly
#no importance to demand history and the fc is almost solely
#based on the latest observations
