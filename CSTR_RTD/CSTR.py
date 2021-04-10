import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from calc_tool import calc_tool

path = os.path.abspath(__file__)
python_file = path.split('/')[-1]
path = path.replace(f'/{python_file}', '')
os.chdir(path)

data_60 = pd.read_excel(f'{path}/CSTR.xlsx', sheet_name = '2,5(60rpm)')
data_90 = pd.read_excel(f'{path}/CSTR.xlsx', sheet_name = '2,5(90rpm)')

class calculation():
    def __init__(self, path, file_name):
        self.data = pd.read_csv(f'{path}/{file_name}')
        self.regression_data = None
        self.name = file_name[0:-4]
        self.conc_convert = False
        self.RTD_data = None
        self.fitted_data = None

    def conductance_to_conc(self):
        min_max_info = []
        max_index = self.data.shape[0]-1
        min_max_info.append([self.data.at[0, 'tank1'], self.data.at[max_index, 'tank1']])
        min_max_info.append([self.data.at[0, 'tank2'], self.data.at[max_index, 'tank2']])
        min_max_info.append([self.data.at[0, 'tank3'], self.data.at[max_index, 'tank3']])
        
        for i in range(self.data.shape[0]):
            self.data.at[i, 'tank1'] = (0.02-0.0025)/(min_max_info[0][1]-min_max_info[0][0])*(self.data.at[i, 'tank1'] - min_max_info[0][0]) + 0.0025
            self.data.at[i, 'tank2'] = (0.02-0.0025)/(min_max_info[1][1]-min_max_info[1][0])*(self.data.at[i, 'tank2'] - min_max_info[1][0]) + 0.0025
            self.data.at[i, 'tank3'] = (0.02-0.0025)/(min_max_info[2][1]-min_max_info[2][0])*(self.data.at[i, 'tank3'] - min_max_info[2][0]) + 0.0025

        self.conc_convert = True
        
    def regression(self):
        t_array = self.data['t']
        tank1_array = self.data['tank1']
        tank2_array = self.data['tank2']
        tank3_array = self.data['tank3']

        tank1_reg = np.polyfit(t_array, tank1_array, 10)
        tank2_reg = np.polyfit(t_array, tank2_array, 10)
        tank3_reg = np.polyfit(t_array, tank3_array, 10)

        self.regression_data = {'tank1' : tank1_reg, 'tank2' : tank2_reg, 'tank3' : tank3_reg}

    def plot_conc(self):
        plt.close()
        plt.plot(self.data['t'], self.data['tank1'], 'ro-', label = 'tank1')
        plt.plot(self.data['t'], self.data['tank2'], 'go-', label = 'tank2')
        plt.plot(self.data['t'], self.data['tank3'], 'bo-', label = 'tank3')
        plt.grid()
        plt.xlabel('time (min)')
        plt.ylabel('Concentration (N)')
        plt.title(f'{self.name} result')
        plt.legend()
        plt.show()
    
    def tau_by_RTD(self):
        E_1 = self.RTD_data['RTD_E1']
        E_2 = self.RTD_data['RTD_E2']
        E_3 = self.RTD_data['RTD_E3']
        t_array = self.fitted_data['t']
        
        tE_1 = t_array * E_1
        tE_2 = t_array * E_2
        tE_3 = t_array * E_3

        def numeric_int(x, y):
            A = 0
            for i in range(len(x)-1):
                A += 0.5*(y[i+1]+y[i])*(x[i+1]-x[i])
            return A
        
        tau1 = numeric_int(t_array, tE_1)/numeric_int(t_array, E_1)
        tau2 = numeric_int(t_array, tE_2)/numeric_int(t_array, E_2)
        tau3 = numeric_int(t_array, tE_3)/numeric_int(t_array, E_3)
    
        print('residence time by RTD')
        print(tau1)
        print(tau2)
        print(tau3)

    def RTD_calc(self):
        if self.conc_convert == False:
            self.conductance_to_conc()

        CB = 0.02

        t_array = self.fitted_data['t']

        c0_array = np.zeros_like(t_array) + CB
        c1_array = self.fitted_data['C1']
        c2_array = self.fitted_data['C2']
        c3_array = self.fitted_data['C3']

        E1_array = np.zeros_like(c1_array)
        E2_array = np.zeros_like(c2_array)
        E3_array = np.zeros_like(c3_array)

        for i in range(len(E1_array)):
            if i == 0:
                E1_array[i] = (c1_array[i+1]-c1_array[i])/(t_array[i+1]-t_array[i])/c0_array[i]
                E2_array[i] = (c2_array[i+1]-c2_array[i])/(t_array[i+1]-t_array[i])/c1_array[i]
                E3_array[i] = (c3_array[i+1]-c3_array[i])/(t_array[i+1]-t_array[i])/c2_array[i]
            else:
                E1_array[i] = (c1_array[i]-c1_array[i-1])/(t_array[i]-t_array[i-1])/c0_array[i]
                E2_array[i] = (c2_array[i]-c2_array[i-1])/(t_array[i]-t_array[i-1])/c1_array[i]
                E3_array[i] = (c3_array[i]-c3_array[i-1])/(t_array[i]-t_array[i-1])/c2_array[i]

        for i in range(len(E1_array)):
            if E1_array[i]<0:
                E1_array[i] = 0
            if E2_array[i]<0:
                E2_array[i] = 0
            if E3_array[i]<0:
                E3_array[i] = 0

        
        plt.close()
        plt.plot(t_array, E1_array, 'r-', label = 'tank1 RTD')
        plt.plot(t_array, E2_array, 'g-', label = 'tank2 RTD')
        plt.plot(t_array, E3_array, 'b-', label = 'tank3 RTD')
        plt.xlabel('time(min)')
        plt.ylabel('RTD')
        plt.title(f'RTD for each tank, {self.name}')
        plt.grid()
        plt.legend()
        plt.show()

        result = {'RTD_E1' : E1_array, 'RTD_E2' : E2_array, 'RTD_E3' : E3_array}
        self.RTD_data = result
        
        return result

    def therotical(self, max_time):
        tau = 5
        CB = 0.02
        C0 = 0.0025
        t_array = np.linspace(0, max_time, 100)
        C1_array = CB*(1-np.exp(-t_array/tau))+C0*np.exp(-t_array/tau)
        C2_array = CB*(1-np.exp(-t_array/tau)*(t_array+tau)/tau)+C0*np.exp(-t_array/tau)*(t_array/tau+1)
        C3_array = CB*(1-0.5*(2*tau**2+2*tau*t_array+t_array**2)*np.exp(-t_array/tau)/tau**2)+C0*(t_array**2/tau**2+t_array/tau+1)*np.exp(-t_array/tau)

        plt.close()
        plt.plot(t_array, C1_array, 'r-', label = 'tank1_the')
        plt.plot(t_array, C2_array, 'g-', label = 'tank2_the')
        plt.plot(t_array, C3_array, 'b-', label = 'tank3_the')
        plt.plot(self.data['t'], self.data['tank1'], 'r:', label = 'tank1_exp')
        plt.plot(self.data['t'], self.data['tank2'], 'g:', label = 'tank2_exp')
        plt.plot(self.data['t'], self.data['tank3'], 'b:', label = 'tank3_exp')
        plt.grid()
        plt.xlabel('time (min)')
        plt.ylabel('Concentration (N)')
        plt.title(f'{self.name} Theoritical prediction')
        plt.legend()
        plt.show()

    def residence_time_fitting(self):
        CB = 0.02
        C0 = 0.0025

        def f1(t, tau1):
            return CB*(1-np.exp(-t/tau1))+C0*np.exp(-t/tau1)
        
        def f2(X, tau2):
            t, tau1 = X
            result = CB/(tau1-tau2)*(tau1*(1-np.exp(-t/tau1))+tau2*(np.exp(-t/tau2)-1))+\
                tau1*C0/(tau1-tau2)*(np.exp(-t/tau1)-np.exp(-t/tau2))+\
                    C0*np.exp(-t/tau2)
            return result
        
        def f3(X, tau3):
            t, tau1, tau2 = X
            result = CB*(1-tau1**2*np.exp(-t/tau1)/((tau1-tau2)*(tau1-tau3))+tau2**2*np.exp(-t/tau2)/((tau1-tau2)*(tau2-tau3))+tau3**2*np.exp(-t/tau3)/((tau1-tau3)*(tau3-tau2)))+\
                C0/(tau2*tau3)*(tau1**2*tau2*tau3*np.exp(-t/tau1)/((tau1-tau2)*(tau1-tau3))-tau1*tau2**2*tau3*np.exp(-t/tau2)/((tau1-tau2)*(tau2-tau3))-tau1*tau2*tau3**2*np.exp(-t/tau3)/((tau1-tau3)*(tau3-tau2)))+\
                    tau2*C0/(tau2-tau3)*(np.exp(-t/tau2)-np.exp(-t/tau3))+C0*np.exp(-t/tau3)
            return result

        t_data = self.data['t']
        f1_data = self.data['tank1']
        f2_data = self.data['tank2']
        f3_data = self.data['tank3']

        tau1 = curve_fit(f1, t_data, f1_data)[0][0]
        tau1_data = np.zeros_like(t_data)+tau1
        tau2 = curve_fit(f2, (t_data, tau1_data), f2_data)[0][0]
        tau2_data = np.zeros_like(t_data)+tau2
        tau3 = curve_fit(f3, (t_data, tau1_data, tau2_data), f3_data)[0][0]

        print(tau1)
        print(tau2)
        print(tau3)

        max_time = self.data['t'][self.data.shape[0]-1]
        t_array = np.linspace(0,max_time,500)
        
        C1_array = CB*(1-np.exp(-t_array/tau1))+C0*np.exp(-t_array/tau1)
        C2_array = CB/(tau1-tau2)*(tau1*(1-np.exp(-t_array/tau1))+tau2*(np.exp(-t_array/tau2)-1))+\
                tau1*C0/(tau1-tau2)*(np.exp(-t_array/tau1)-np.exp(-t_array/tau2))+\
                    C0*np.exp(-t_array/tau2)
        C3_array = CB*(1-tau1**2*np.exp(-t_array/tau1)/((tau1-tau2)*(tau1-tau3))+tau2**2*np.exp(-t_array/tau2)/((tau1-tau2)*(tau2-tau3))+tau3**2*np.exp(-t_array/tau3)/((tau1-tau3)*(tau3-tau2)))+\
                C0/(tau2*tau3)*(tau1**2*tau2*tau3*np.exp(-t_array/tau1)/((tau1-tau2)*(tau1-tau3))-tau1*tau2**2*tau3*np.exp(-t_array/tau2)/((tau1-tau2)*(tau2-tau3))-tau1*tau2*tau3**2*np.exp(-t_array/tau3)/((tau1-tau3)*(tau3-tau2)))+\
                    tau2*C0/(tau2-tau3)*(np.exp(-t_array/tau2)-np.exp(-t_array/tau3))+C0*np.exp(-t_array/tau3)
        
        fitted_data = {'t' : t_array, 'C1' : C1_array, 'C2' : C2_array, 'C3' : C3_array}
        self.fitted_data = pd.DataFrame(fitted_data)

        plt.close()
        plt.plot(t_array, C1_array, 'r-', label = f'tank1_fit, tau1 = {calc_tool.round_sig(tau1, 4)}')
        plt.plot(t_array, C2_array, 'g-', label = f'tank2_fit, tau2 = {calc_tool.round_sig(tau2, 4)}')
        plt.plot(t_array, C3_array, 'b-', label = f'tank3_fit, tau3 = {calc_tool.round_sig(tau3, 4)}')
        plt.plot(self.data['t'], self.data['tank1'], 'r:', label = 'tank1_exp')
        plt.plot(self.data['t'], self.data['tank2'], 'g:', label = 'tank2_exp')
        plt.plot(self.data['t'], self.data['tank3'], 'b:', label = 'tank3_exp')
        plt.grid()
        plt.xlabel('time (min)')
        plt.ylabel('Concentration (N)')
        plt.title(f'{self.name} Residence time fitting')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    rpm_60 = calculation(path, 'CSTR_60.csv')
    rpm_60.conductance_to_conc()
    rpm_60.plot_conc()
    rpm_60.regression()
    rpm_60.therotical(rpm_60.data['t'][rpm_60.data.shape[0]-1])
    rpm_60.residence_time_fitting()
    rpm_60.RTD_calc()
    rpm_60.tau_by_RTD()


    rpm_90 = calculation(path, 'CSTR_90.csv')
    rpm_90.conductance_to_conc()
    rpm_90.plot_conc()
    rpm_90.regression()
    rpm_90.therotical(rpm_90.data['t'][rpm_90.data.shape[0]-1])
    rpm_90.residence_time_fitting()
    rpm_90.RTD_calc()
    rpm_90.tau_by_RTD()
