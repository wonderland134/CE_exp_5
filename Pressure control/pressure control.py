import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

path = os.path.abspath(__file__)
python_file = path.split('/')[-1]
path = path.replace(f'/{python_file}', '')
os.chdir(path)

def round_sig(x, sig):
	a = np.log10(abs(x))
	if a >= 0: n = np.trunc(a)
	elif a < 0: n = np.trunc(a)-1
	return np.round_(x, int(-1*(n+1-sig)))

class analize():
    def __init__(self, folder):
        self.folder = folder
        self.file_list = os.listdir(folder)
        self.file_list.sort()
        self.file_list = [t for t in self.file_list if t[0] != '.']
        path = os.path.abspath(__file__)
        python_file = path.split('/')[-1]
        path = path.replace(f'/{python_file}', '')
        self.result_path = path + '/result'
        self.path = path + f'/{self.folder}'

        self.result = pd.DataFrame(columns = ['Type', 'P', 'I', 'D', 'IAE', 'ISE', 'ITAE', 'ITSE'])
        self.raw_data = []

    def calc_all(self):
        for f in self.file_list:
            trial = Process_data(f, self.path) 
            
            '''
            #역동작 시간통일 시 주석 해제
            trial.data_file = trial.data_file.loc[range(0,37), :]
            '''
            '''
            #정동작 시간통일 시 주석 해제
            trial.data_file = trial.data_file.loc[range(0,51), :]
            '''
            print(trial.data_file.shape[0])
            self.raw_data.append(trial)

            err = trial.calc_err()
            data = {'Type' : trial.type_, 'P' : trial.p, 'I' : trial.i, 'D' : trial.d,\
                'IAE' : err['IAE'], 'ISE' : err['ISE'], 'ITAE' : err['ITAE'], 'ITSE' : err['ITSE']}
            self.result = self.result.append(data, ignore_index = True)

    def save_result(self):
        self.result.to_excel(f'{self.result_path}/{self.folder}.xlsx')
    
    def data_plot(self):
        plt.close()
        color_chart = ['b', 'g', 'r', 'y']
        for i, data_set in enumerate(self.raw_data):
            #제어출력도 plot하기 원하면 밑의 주석 해제
            t_array = data_set.data_file['실험시간']
            p_array = data_set.data_file['현재압력']
            #control_array = data_set.data_file['제어출력[%]']/100
            p_array_set = data_set.data_file['설정압력']
            name = f'{data_set.type_} P:{data_set.p},I:{data_set.i},D:{data_set.d}'
            plt.plot(t_array, p_array, color_chart[i], label = name)
            #plt.plot(t_array, control_array, color_chart[i]+':', label = f'{name}, Control')
            plt.plot(t_array, p_array_set, 'k:')
            
        plt.grid()
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('Pressure')
        plt.title(self.folder)
        plt.savefig(f'{self.result_path}/{self.folder}.png', dpi = 600)
        plt.show()

class Process_data():
    def __init__(self, file_name, path):
        self.file_name = file_name
        data = file_name[0:-4].split('&')
        if len(data) == 4:
            self.type_ = data[0]
            self.p = data[1]
            self.i = data[2]
            self.d = data[3]
        elif len(data) == 5:
            self.type_ = data[0]
            self.p = data[1]
            self.i = data[2]
            self.d = data[3]
        elif len(data) == 2:
            self.type_ = 'ONOFF'
            self.p = '-'
            self.i = '-'
            self.d = '-'

        self.data_file = pd.read_csv(f'{path}/{file_name}', header = 3)
        self.data_file = self.refine_time_data(self.data_file)
        self.cut_data()
        self.parameters = None
        self.err = None

    def refine_time_data(self, data):
        for i in range(data.shape[0]):
            if i == 0:
                data.at[i, '실험시간'] = self.time_stemp_to_sec(data.at[i,'실험시간'])
                start_point = data.at[i, '실험시간']
                data.at[i, '실험시간'] = 0
            else:
                data.at[i, '실험시간'] = self.time_stemp_to_sec(data.at[i,'실험시간']) - start_point
        return (data.astype(float))
    
    def cut_data(self):
        for i in range(self.data_file['설정압력'].shape[0]):
            if self.data_file['설정압력'][i] > 0:
                self.data_file = self.data_file.loc[range(i,self.data_file.shape[0]), :]
                self.data_file.reset_index(drop = True, inplace = True)

                start_point = self.data_file['실험시간'][0]
                for j in range(self.data_file.shape[0]):
                    self.data_file.at[j, '실험시간'] = self.data_file.at[j, '실험시간'] - start_point
                break
    
    def show_plot(self):
        t_array = self.data_file['실험시간']
        P_array = self.data_file['현재압력']
        P_set_array = self.data_file['설정압력']
        output_array = self.data_file['제어출력[%]']/100

        plt.close()
        plt.plot(t_array, P_array, 'b-', label = 'Current pressure')
        plt.plot(t_array, P_set_array, 'g:', label = 'Target pressure')
        plt.plot(t_array, output_array, 'k:', label = 'Output')
        plt.legend()
        plt.grid()
        plt.xlabel('time(s)')
        plt.ylabel('Pressure or Cont.Output')
        plt.title('Pressure control')
        plt.show()

    def calc_err(self):
        t_array = self.data_file['실험시간']
        e_array = self.data_file['설정압력']-self.data_file['현재압력']
        e_sq_array = e_array**2
        te_array = t_array * e_array
        te_sq_array = t_array * e_sq_array
        
        '''
        IAE
        '''
        IAE = 0
        for i in range(len(t_array)-1):
            IAE += 0.5*abs(e_array[i]+e_array[i+1])*(t_array[i+1]-t_array[i])
        
        '''
        ISE
        '''
        ISE = 0
        for i in range(len(t_array)-1):
            ISE += 0.5*(e_sq_array[i]+e_sq_array[i+1])*(t_array[i+1]-t_array[i])
        
        '''
        ITAE
        '''
        ITAE = 0
        for i in range(len(t_array)-1):
            ITAE += 0.5*abs(te_array[i]+te_array[i+1])*(t_array[i+1]-t_array[i])
        
        '''
        ITSE
        '''
        ITSE = 0
        for i in range(len(t_array)-1):
            ITSE += 0.5*(te_sq_array[i]+te_sq_array[i+1])*(t_array[i+1]-t_array[i])
        
        self.err = {'IAE' : IAE, 'ISE' : ISE, 'ITAE' : ITAE, 'ITSE' : ITSE}
        return self.err
        
    
    def first_order_fitting(self):
        def func(t, a, b, c, d):
            #a : 정상상태 이득
            #b : 겉보기 시간지연
            #c : 겉보기 1차 시간상수
            #d*a : 정상상태 수렴값
            return a*d*(1-np.exp(-(t-b)/c))*np.heaviside((t-b), 0)
        t_array = self.data_file['실험시간']
        y_exp_array = self.data_file['현재압력']
        constants = curve_fit(func, t_array, y_exp_array)[0]
        parameters = {'Kp' : constants[0], 'Td' : constants[1], 'T' : constants[2], 'Bu' : constants[3]*constants[0]}
        self.parameters = parameters
        print(f'Kp : {parameters["Kp"]}')
        print(f'Td : {parameters["Td"]}')
        print(f'T : {parameters["T"]}')
        print(f'Bu : {parameters["Bu"]}')

        tan_line_x = [parameters['Td'], parameters['T']+parameters['Td']]
        tan_line_y = [0,  parameters['Bu']]
        
        a, b, c, d = tuple(constants)
        y_fit_array = a*d*(1-np.exp(-(t_array-b)/c))*np.heaviside((t_array-b), 0)
        a, b, c, d = (round_sig(a, 3), round_sig(b, 3), round_sig(c, 3), round_sig(d, 3))
        plt.close()
        plt.plot(t_array, y_exp_array, 'ro', label = 'exp data')
        plt.plot(t_array, y_fit_array, 'b-', label = f'fit data\n{round_sig(a*d, 3)}(1-exp(-(t-{b})/{c})u(t-{b})')
        plt.plot(t_array, np.zeros_like(t_array)+self.data_file['설정압력'][0], 'g:', label = 'set value')
        plt.plot(tan_line_x, tan_line_y, 'k', label = 'tangent line')
        plt.legend()
        plt.grid()
        plt.title('Step input & respond')
        plt.show()
        
        return parameters
    
    def z_n_tuning_ultimate(self, Ku, Pu):
        z_n_PID = {}
        '''
        P
        '''
        Kc = 0.5*Ku
        z_n_PID['P'] = {'Kc' : Kc, 'tau_i' : 0, 'tau_d' : 0, 'Ki' : 0, 'Kd' : 0}
        '''
        PI
        '''
        Kc = 0.45*Ku
        tau_i = Pu/1.2
        Ki = Kc/tau_i
        z_n_PID['PI'] = {'Kc' : Kc, 'tau_i' : tau_i, 'tau_d' : 0, 'Ki' : Ki, 'Kd' : 0}
        '''
        PD
        '''
        Kc = 0.8*Ku
        tau_d = Pu/8
        Kd = Kc*tau_d
        z_n_PID['PI'] = {'Kc' : Kc, 'tau_i' : 0, 'tau_d' : tau_d, 'Ki' : 0, 'Kd' : Kd}
        '''
        PID
        '''
        Kc = 0.6*Ku
        tau_i = Pu/2
        tau_d = Pu/8
        Ki = Kc/tau_i
        Kd = Kc*tau_d
        z_n_PID['PID'] = {'Kc' : Kc, 'tau_i' : tau_i, 'tau_d' : tau_d, 'Ki' : Ki, 'Kd' : Kd}
        
        return z_n_PID
        
    
    def z_n_tuning(self, parameters):
        T = parameters['T']
        Td = parameters['Td']
        Kp = parameters['Kp']
        z_n_PID = {}
        '''
        P
        '''
        Kc = T/(Kp*Td)
        z_n_PID['P'] = {'Kc' : Kc, 'tau_i' : 0, 'tau_d' : 0, 'Ki' : 0, 'Kd' : 0}
        '''
        PI
        '''
        Kc = 0.9*(T/(Kp*Td))
        tau_i = 3.33*Td
        Ki = Kc/tau_i
        z_n_PID['PI'] = {'Kc' : Kc, 'tau_i' : tau_i, 'tau_d' : 0, 'Ki' : Ki, 'Kd' : 0}
        '''
        PID
        '''
        Kc = 1.2*(T/(Kp*Td))
        tau_i = 2*Td
        tau_d = 0.5*Td
        Ki = Kc/tau_i
        Kd = Kc*tau_d
        z_n_PID['PID'] = {'Kc' : Kc, 'tau_i' : tau_i, 'tau_d' : tau_d, 'Ki' : Ki, 'Kd' : Kd}
        
        return z_n_PID
    
    def c_c_tuning(self, parameters):
        T = parameters['T']
        Td = parameters['Td']
        Kp = parameters['Kp']
        c_c_PID = {}
        
        '''
        P
        '''
        Kc = T/(Kp*Td)*(1 + Td/(3*T))
        c_c_PID['P'] = {'Kc' : Kc, 'tau_i' : 0, 'tau_d' : 0, 'Ki' : 0, 'Kd' : 0}
        '''
        PI
        '''
        Kc = T/(Kp*Td)*(9/10 + Td/(12*T))
        tau_i = Td*(30 + 3*Td/T)/(9 + 20*Td/T)
        Ki = Kc/tau_i
        c_c_PID['PI'] = {'Kc' : Kc, 'tau_i' : tau_i, 'tau_d' : 0, 'Ki' : Ki, 'Kd' : 0}
        '''
        PD
        '''
        Kc = T/(Kp*Td)*(5/4 + Td/(6*T))
        tau_d = Td*(6 - 2*Td/T)/(22 + 3*Td/T)
        Kd = Kc*tau_d
        c_c_PID['PD'] = {'Kc' : Kc, 'tau_i' : 0, 'tau_d' : tau_d, 'Ki' : 0, 'Kd' : Kd}
        '''
        PID
        '''
        Kc = T/(Kp*Td)*(4/3 + Td/(4*T))
        tau_i = Td*(32 + 6*Td/T)/(13 + 8*Td/T)
        tau_d = Td*4/(11 + 2*Td/T)
        Ki = Kc/tau_i
        Kd = Kc*tau_d
        c_c_PID['PID'] = {'Kc' : Kc, 'tau_i' : tau_i, 'tau_d' : tau_d, 'Ki' : Ki, 'Kd' : Kd}
        
        return c_c_PID
        
    
    def time_stemp_to_sec(self, stemp):
        sec = 0
        if stemp[0:2] == '오전':
            sec += 0
        elif stemp[0:2] == '오후':
            sec += 12 * 3600
        
        temp = stemp[3:]
        temp = temp.split(':')
        sec += int(temp[0])*3600
        sec += int(temp[1])*60
        sec += int(temp[2])
        
        return sec

if __name__ == '__main__':
    '''
    p = Process_data('B&OPEN.csv', '/Users/junhee/Desktop/압력제어')
    p.first_order_fitting()
    z_n = pd.DataFrame(p.z_n_tuning(p.parameters))
    z_n.to_excel('zn.xlsx')
    print(z_n)
    c_c = pd.DataFrame(p.c_c_tuning(p.parameters))
    c_c.to_excel('cc.xlsx')
    print(c_c)
    '''

    data_sample0 = analize('B P=4')
    data_sample0.calc_all()
    data_sample0.save_result()
    data_sample0.data_plot()

    data_sample1 = analize('B P=4, I=0.1')
    data_sample1.calc_all()
    data_sample1.save_result()
    data_sample1.data_plot()

    data_sample2 = analize('B P=4, I=0.2')
    data_sample2.calc_all()
    data_sample2.save_result()
    data_sample2.data_plot()

    data_sample3 = analize('B P=4, I=0.3')
    data_sample3.calc_all()
    data_sample3.save_result()
    data_sample3.data_plot()
    
    data_sample4 = analize('B P=8')
    data_sample4.calc_all()
    data_sample4.save_result()
    data_sample4.data_plot()

    data_sample5 = analize('B P=8, I=0.1')
    data_sample5.calc_all()
    data_sample5.save_result()
    data_sample5.data_plot()

    data_sample6 = analize('B P=8, I=0.2')
    data_sample6.calc_all()
    data_sample6.save_result()
    data_sample6.data_plot()

    data_sample7 = analize('B P=8, I=0.3')
    data_sample7.calc_all()
    data_sample7.save_result()
    data_sample7.data_plot()
    
    data_sample8 = analize('B P=12')
    data_sample8.calc_all()
    data_sample8.save_result()
    data_sample8.data_plot()

    data_sample9 = analize('B P=12, I=0.1')
    data_sample9.calc_all()
    data_sample9.save_result()
    data_sample9.data_plot()

    data_sample10 = analize('B P=12, I=0.2')
    data_sample10.calc_all()
    data_sample10.save_result()
    data_sample10.data_plot()

    data_sample11 = analize('B P=12, I=0.3')
    data_sample11.calc_all()
    data_sample11.save_result()
    data_sample11.data_plot()

    data_sample12 = analize('B ONOFF')
    data_sample12.calc_all()
    data_sample12.save_result()
    data_sample12.data_plot()


    
    data_sample0 = analize('F P=4')
    data_sample0.calc_all()
    data_sample0.save_result()
    data_sample0.data_plot()

    data_sample1 = analize('F P=4, I=0.1')
    data_sample1.calc_all()
    data_sample1.save_result()
    data_sample1.data_plot()

    data_sample2 = analize('F P=4, I=0.2')
    data_sample2.calc_all()
    data_sample2.save_result()
    data_sample2.data_plot()

    data_sample3 = analize('F P=4, I=0.3')
    data_sample3.calc_all()
    data_sample3.save_result()
    data_sample3.data_plot()
    
    data_sample4 = analize('F P=8')
    data_sample4.calc_all()
    data_sample4.save_result()
    data_sample4.data_plot()

    data_sample5 = analize('F P=8, I=0.1')
    data_sample5.calc_all()
    data_sample5.save_result()
    data_sample5.data_plot()

    data_sample6 = analize('F P=8, I=0.2')
    data_sample6.calc_all()
    data_sample6.save_result()
    data_sample6.data_plot()

    data_sample7 = analize('F P=8, I=0.3')
    data_sample7.calc_all()
    data_sample7.save_result()
    data_sample7.data_plot()
    
    data_sample8 = analize('F P=12')
    data_sample8.calc_all()
    data_sample8.save_result()
    data_sample8.data_plot()

    data_sample9 = analize('F P=12, I=0.1')
    data_sample9.calc_all()
    data_sample9.save_result()
    data_sample9.data_plot()

    data_sample10 = analize('F P=12, I=0.2')
    data_sample10.calc_all()
    data_sample10.save_result()
    data_sample10.data_plot()

    data_sample11 = analize('F P=12, I=0.3')
    data_sample11.calc_all()
    data_sample11.save_result()
    data_sample11.data_plot()

    data_sample12 = analize('F ONOFF')
    data_sample12.calc_all()
    data_sample12.save_result()
    data_sample12.data_plot()
    
