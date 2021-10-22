import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt

class KPI:

    def __init__(self, data):
        self.data = data
        self.dem_ave = self.data.loc[self.data['Error'].notnull(), 'Demand'].mean()

    def bias(self):
        bias_abs = self.data['Error'].mean()
        bias_rel = bias_abs / self.dem_ave
        print ('Bias: {:0.2f}, {:.2%}'.format(bias_abs, bias_rel))

    def mape(self):
        mape = (self.data['Error'].abs() / self.data['Demand']).mean()
        print('MAPE: {:.2%}'.format(mape))

    def mae(self):
        mae_abs = self.data['Error'].abs().mean()
        mae_rel = mae_abs / self.dem_ave
        print('MAE: {:0.2f}, {:.2%}'.format(mae_abs, mae_rel))

    def rmse(self):
        rmse_abs = np.sqrt((self.data['Error']**2).mean())
        rmse_rel = rmse_abs / self.dem_ave
        print('RMSE: {:0.2f}, {:.2%}'.format(rmse_abs, rmse_rel))

    def all_kpi(self):
        all_bias = self.bias()
        all_mape = self.mape()
        all_mae = self.mae()
        all_rmse = self.rmse()

class Plot_Graph:

    def __init__(self, model):
        self.model = model

    def plot_ses(self):
        self.model.index.name = 'Period'
        self.model[['Demand', 'Forecast']].plot(
            figsize=(8,3),
            title='Simple Smoothing',
            xlim=(0, 25),
            ylim=(0, 30),
            style=['-', '--']
        )

    def plot_des(self):
        self.model.index.name = 'Period'
        self.model[['Demand', 'Forecast', 'Trend']].plot(
            figsize=(8,3),
            title='Double Smoothing',
            xlim=(0, 25),
            ylim=(0, 30),
            style=['-', '--','--']
        )

    def plot_tes(self):
        self.model.index.name = 'Period'
        self.model[['Demand', 'Forecast', 'Trend', 'Season']].plot(
            figsize=(8,3),
            title='TES',
            xlim=(0, 25),
            ylim=(0, 30),
            style=['-', '--', '--', '-']
        )

    def plot_tes_add(self):
        self.model.index.name = 'Period'
        self.model[['Demand', 'Forecast', 'Trend', 'Season']].plot(
            figsize=(8, 3),
            title='TES Additive',
            xlim=(0, 250),
            ylim=(0, 400),
            style=['-', '--', '--', '-']
        )

class Forecasting:

    def __init__(self, data, periods, alpha):
        self.data = data
        self.periods = periods
        self.alpha = alpha
        self.beta = 0


    def ses(self):
        col = len(self.data)
        self.data = np.append(self.data,[np.nan]*self.periods)

        f = np.full(col+self.periods, np.nan)

        f[1] = self.data[0]

        for t in range(2, col+1):
            f[t] = self.alpha*self.data[t-1]+(1-self.alpha)*f[t-1]

        for t in range(col+1, col+self.periods):
            f[t] = f[t-1]

        ses = pd.DataFrame.from_dict({'Demand':self.data, 'Forecast':f, 'Error':self.data-f})

        return ses

    def des(self, beta):
        col = len(self.data)
        self.data = np.append(self.data, [np.nan]*self.periods)
        f, a, b = np.full((3, col+self.periods), np.nan)

        a[0] = self.data[0]
        b[0] = self.data[1] - self.data[0]

        for t in range(1, col):
            f[t] = a[t-1] + b[t-1]
            a[t] = self.alpha*self.data[t] + (1-self.alpha)*(a[t-1]+b[t-1])
            b[t] = beta*(a[t]-a[t-1]) + (1-beta)*b[t-1]

        for t in range(col, col+self.periods):
            f[t] = a[t-1] + b[t-1]
            a[t] = f[t]
            b[t] = b[t-1]

        des = pd.DataFrame.from_dict(
            {'Demand':self.data,
             'Forecast':f,
             'Level':a,
             'Trend':b,
             'Error':self.data-f})

        return des

    def seasonal_factors_mul(self):
        for i in range(self.slen):
            self.s[i] = np.mean(self.data[i:self.col:self.slen])
        self.s /= np.mean(self.s[:self.slen])

        return self.s

    def tes(self, beta, phi, gamma):
        self.slen = slen
        self.col = len(self.data)
        self.data = np.append(self.data, [np.nan]*self.periods)

        f, a, b, self.s = np.full((4, self.col+self.periods), np.nan)
        self.s = self.seasonal_factors_mul()

        a[0] = self.data[0]/self.s[0]
        b[0] = self.data[1]/self.s[1] - self.data[0]/s[0]

        for t in range(1, self.slen):
            f[t] = (a[t-1] + phi*b[t-1])*self.s[t]
            a[t] = self.alpha*self.data[t]/self.s[t] + (1-self.alpha)*(a[t-1]+phi*b[t-1])
            b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]

        for t in range(self.slen, self.col):
            f[t] = (a[t-1] + phi*b[t-1])*self.s[t-self.slen]
            a[t] = self.alpha*self.data[t]/self.s[t-self.slen] + (1-self.alpha)*(a[t-1]+phi*b[t-1])
            b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]
            self.s[t] = gamma*self.data[t]/a[t] + (1-gamma)*s[t-self.slen]

        for t in range(self.col, self.col+self.periods):
            f[t] = (a[t-1] + phi*b[t-1])*self.s[t-self.slen]
            a[t] = f[t]/self.s[t-self.slen]
            b[t] = phi*b[t-1]
            self.s[t] = self.s[t-self.slen]

        tes = pd.DataFrame.from_dict(
            {'Demand':self.data,
             'Forecast':f,
             'Level':a,
            'Trend':b,
            'Season':self.s,
            'Error':self.data-f}
        )

        return tes

    def seasonal_factors_add(self):
        for i in range(self.slen):
            self.s[i] = np.mean(self.data[i:self.col:self.slen])
        self.s -= np.mean(self.s[:self.slen])

        return self.s

    def tes_add(self, slen, beta, phi, gamma):
        self.slen = slen
        self.col = len(self.data)
        self.data = np.append(self.data, [np.nan]*self.periods)

        f, a, b, self.s = np.full((4, self.col+self.periods), np.nan)
        self.s = self.seasonal_factors_add()

        a[0] = self.data[0]-self.s[0]
        b[0] = (self.data[1]-self.s[1]) - (self.data[0]-self.s[0])

        for t in range(1, self.slen):
            f[t] = a[t-1] + phi*b[t-1] + self.s[t]
            a[t] = self.alpha*(self.data[t]-self.s[t]) + (1-self.alpha)*(a[t-1]+phi*b[t-1])
            b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]

        for t in range(self.slen, self.col):
            f[t] = a[t-1] + phi*b[t-1] + self.s[t-self.slen]
            a[t] = self.alpha*(self.data[t]-self.s[t-self.slen]) + (1-self.alpha)*(a[t-1]+phi*b[t-1])
            b[t] = beta*(a[t]-a[t-1]) + (1-beta)*phi*b[t-1]
            self.s[t] = gamma*(self.data[t]-a[t]) + (1-gamma)*self.s[t-self.slen]

        for t in range(self.col, self.col+self.periods):
            f[t] = a[t-1] + phi*b[t-1] + self.s[t-self.slen]
            a[t] = f[t] - self.s[t-self.slen]
            b[t] = phi*b[t-1]
            self.s[t] = self.s[t-self.slen]

        tes_add = pd.DataFrame.from_dict(
            {'Demand':self.data,
             'Forecast':f,
             'Level':a,
            'Trend':b,
            'Season':self.s,
            'Error':self.data-f}
        )

        return tes_add

    def mae_opti(self, beta):
        self.beta = beta
        params = []
        kpi_mae = []
        forecastframe = []

        for alpha in np.linspace(0.05, 0.6, 100):

            forecast = self.ses()
            params.append('Simple Smoothing, alpha: {}'.format(self.alpha))
            forecastframe.append(forecast)
            mae = forecast['Error'].abs().mean()
            kpi_mae.append(mae)

            for beta in np.linspace(0.05, 0.6, 100):

                forecast = self.des(beta=self.beta)
                params.append('Double Smoothing alpha: {}, beta: {}'.format(self.alpha, self.beta))
                forecastframe.append(forecast)
                mae = forecast['Error'].abs().mean()
                kpi_mae.append(mae)

        mini = np.argmin(kpis)
        print(f'Optimal solution found for {params[mini]} MAE of', round(kpis[mini], 2))

        return forecastframe[mini]

Data = [21060.51, 22133.60, 22855.00, 24548.82, 25688.19, 26534.00, 28150.00, 29075.28, 31375.00, 32376.00, 33822.44, 35099.48, 36290.00, 35909.00, 35826.00, 36340.08, 37713.00, 36650.82, 37478.00, 38029.33, 38391.66, 39795.93]

Forecast = Forecasting(Data, 6, 0.6)

ses = Forecast.ses()
ses.plot()

des = Forecast.des(0.6)
des[['Demand', 'Forecast']].plot()

tes = Forecast.tes_add(4, 0.53443243, 0.54323, 0.578)
tes.plot()

Kpi_tes = KPI(tes)
Kpi_tes.all_kpi()
