# Master Thesis - Shubham Mamgain (AIP, Potsdam)
# Contact: mamgain@uni-potsdam.de
# Supervisor: Dr. Jesper Storm (jstorm@aip.de)

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#### INPUT: CONSTANTS AND DATA ###########
Ab_v = 1.31   ;Av_v = 1	;Ai_v = 0.608  ;Aj_v = 0.292  ;Ah_v = 0.181  ;Ak_v = 0.119 ; 		A = [Ab_v, Av_v, Ai_v, Aj_v, Ah_v, Ak_v];
Ab = []; Av = []; Ai = []; Aj = []; Ah = []; Ak = [] ; ext = [Ab, Av, Ai, Aj, Ah, Ak]; 		ex_name = ['Ab', 'Av', 'Ai', 'Aj', 'Ah', 'Ak'] ;
R_v = 3.23;R_b = Ab_v*R_v ;R_i = Ai_v*R_v ;R_j = Aj_v*R_v ;R_h = Ah_v*R_v ;R_k = Ak_v*R_v; 	R = [R_b, R_v, R_i, R_j, R_h, R_k];
bands = ['M_B', 'M_V', 'M_I', 'M_J', 'M_H', 'M_K']; 						nband = len(bands);
ap_bands = ['B_mag', 'V_mag' ,'I_mag', 'J_mag', 'H_mag', 'K_mag'];				m = ['B', 'V','I','J','H','K'];
col_dot = ['b.', 'g.', 'y+', 'r*', 'c+', 'k.', 'y+', 'b+'];
col_lin = ['b-', 'g-', 'y-', 'r-', 'c-', 'k-', 'y-', 'b-'] ;
col_das = ['b--', 'g--', 'y--', 'r--', 'c--', 'k--', 'y--', 'b--'];
print('Selective to total absorption ratio (Sandage, 2004): R_V = 3.23'); print('Extinction law (ratio) for B, V, I, J, H, K:'); print(A);

filepath = './125_gaia_based_absolute.csv';
#filepath = '/home/shubham/Desktop/Thesis/Final/Jupyter/125_gaia_based_absolute.csv';
print('Data from: ' + filepath)	; 	data = pd.read_csv(filepath);	data = pd.read_csv(filepath); 	cepheid = len(data.index);

#### Estimating extinction in each band for every star ###
for i in range(0,nband):
    for star in range(0,cepheid):
        ex = R[i]*data.EBV[star];	 ext[i].append(ex);

extinction = pd.DataFrame(list(zip(ext[0], ext[1], ext[2], ext[3], ext[4], ext[5])), columns = ex_name)
extinction['name'] = data.name; print(data); 							print(extinction)
data = pd.merge(data, extinction, on = ['name']);
########################################################


pl = input('If you wish to see plots, type y')


##### DEFINING FUNCTIONS regression; plot #####
def regression(x_data, y_data, index, y_invert_flag_0, x_name, y_name, pl):
    # index
    regression_line = stats.linregress(x_data, y_data); intercept = regression_line.intercept
    slope = regression_line.slope; prediction = slope * x_data + intercept; residue = y_data  - prediction
    slope_error = regression_line.stderr; intercept_error = regression_line.intercept_stderr
    print('%s = %f %s ( %f) + %f ( %f)'%(y_name, slope, x_name, slope_error, intercept, intercept_error))
    print('****************************************************************')
    if pl == 'y':
        plot(x_data, y_data, index, y_invert_flag_0, x_name, y_name,  intercept, slope, prediction, residue, slope_error, intercept_error)
    return slope, intercept, prediction, residue, slope_error, intercept_error

def scatter_plot(x_data, y_data):
    fig, [ax1, ax2] = plt.subplots(figsize=(8,12), nrows=2, ncols = 1, sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(x_data, y_data , col_dot[index])
    inp = input('Press y key to utilise more functions')
    if inp == '':
        y_invert_flag_0 = input('Press 0 if you inverted y axis')
        plot(x_data, y_data, index, y_invert_flag_0, x_axis_str, y_axis_str,  intercept, slope, prediction, residue, slope_error, intercept_error))

def plot(x_data, y_data, index, y_invert_flag_0, x_axis_str, y_axis_str,  intercept, slope, prediction, residue, slope_error, intercept_error):
#    intercept, slope, prediction, residue, slope_error, intercept_error = regression(x_data, y_data, nil1, nil2, x_name, y_name)
    fig, [ax1, ax2] = plt.subplots(figsize=(8,12), nrows=2, ncols = 1, sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(x_data, y_data , col_dot[index])
    ax1.plot(x_data, prediction , col_lin[index], label = '%s = %f ($\pm$ %f) %s + %f ($\pm$ %f)'%(y_axis_str, slope, slope_error, x_axis_str, intercept, intercept_error))
    ax1.plot(x_data, prediction - y_data.std(), col_das[index], label = ' $\sigma$ = %f'%(y_data.std()))
    ax1.plot(x_data, prediction + y_data.std(), col_das[index]) # line of standard deviation
    ax1.set_xlabel(x_axis_str);    ax1.set_ylabel(y_axis_str);							ax1.legend()
    if y_invert_flag_0 == 0:                                    # if flag value given in input, then only invert y-axis
        ax1.invert_yaxis()
    ax2.plot(x_data, residue , col_dot[index], label = 'sigma $%s$ = %f'%(bands[index], y_data.std()));  	ax2.legend();
    ax2.set_xlabel(x_axis_str); ax2.set_ylabel('Deviation from model line')
    ax2.axhline(y=0, color='r', linestyle='--'); ax2.axvline(x=0, color='r', linestyle='--');
    plt.show();

####### PERIOD - LUMINOSITY RELATION ###########
def PL_relation():
    PL_slope = [];	PL_intercept = [];	i_error = [];	s_error = [];	M_predict = pd.DataFrame(); M_residue = pd.DataFrame();
    for wavelength in range(0,6):
        print(' ');    print('Leavitt Law for %s band'%(bands[wavelength]));
        slp, incp, M_predict[bands[wavelength]], M_residue[bands[wavelength]], error_s, error_i = regression(data.logP, data[bands[wavelength]], wavelength, 0, '(logP-1)',  '$%s$'%(bands[wavelength]), pl);
        PL_slope.append(slp);	PL_intercept.append(incp);	i_error.append(error_i);		s_error.append(error_s);
    return PL_slope, PL_intercept, i_error, s_error, M_predict, M_residue
PL_slope, PL_intercept, i_error, s_error, M_predict, M_residue = PL_relation();

####### PERIOD - WESENHEIT RELATION ############
def wesenheit(M_1,M_2, M_3, R_1,R_2, R_3):
    W123 = M_1 - (R_1/(R_2 -R_3))*(M_2-M_3)
    return W123

W = ['BBI','VVI'];

def PW_relation():
    wes = pd.DataFrame();	PW_slope = [];	PW_intercept = [];	wi_error = [];	ws_error = [];	rvi = [];
    W_predict = pd.DataFrame();	W_residue = pd.DataFrame();	wes['name'] = data.name;	lw = len(W);
    for i in range(2,3):                                         # loop for m_3, taking I band only
        for j in range(0,i):                                     # loop for m_2, taking V band only
            for k in range(0,2):                                 # loop for m_1, take all the band
                wes[m[k]+m[j]+m[i]] = wesenheit(data[bands[k]]+data[ex_name[k]],data[bands[j]]+data[ex_name[j]], data[bands[i]]+data[ex_name[i]], R[k],R[j], R[i])
                print(' '); print('Period-Wesenheit relation for W_{%s%s%s}'%(m[k],m[j],m[i]));
                slp, incp, W_predict[m[k]+m[j]+m[i]], W_residue[m[k]+m[j]+m[i]], es, ei = regression(data.logP-1, wes[m[k]+m[j]+m[i]], k, 0, '(logP - 1)', '$W_{%s%s%s}$'%(m[k],m[j],m[i]), pl)
                PW_slope.append(slp); PW_intercept.append(incp);	wi_error.append(ei);	ws_error.append(es);	rvi.append(R[k]/(R[j]-R[i]));
    return PW_slope, PW_intercept, wi_error, ws_error, rvi, wes, W_predict, W_residue

PW_slope, PW_intercept, wi_error, ws_error, rvi, wes, W_predict, W_residue = PW_relation()

################## Delta - Delta plot ###############
def del_plot():
    Del_slope = []	;Del_intercept = []	;Del_ierr = []	;Del_serr = []	;Del_predict = pd.DataFrame()	;Del_residue = pd.DataFrame()	;
    for i in range(0,nband):
        print(' '); print('regression line of Del-Del plot')
        slp, incp, Del_predict[bands[i]], Del_residue[bands[i]], es, ei =  regression(W_residue[W[1]], M_residue[bands[i]], i, 1, '$\Delta W_{%s}$'%(W[1]), '$\Delta %s$'%(bands[i]), pl)
        Del_slope.append(slp)
        Del_intercept.append(incp)
        Del_ierr.append(ei)
        Del_serr.append(ei)
    return Del_slope, Del_intercept, Del_ierr, Del_serr, Del_predict, Del_residue

Del_slope, Del_intercept, Del_ierr, Del_serr, Del_predict, Del_residue = del_plot()

def star(star_num):
    name = data.name.iloc[[star_num]].to_string() #  Getting star name
    period = data.logP.iloc[star_num]             #  period of the star
    res_W = W_residue.iloc[star_num]                      #  getting residue from PW relation
    res_M = M_residue.iloc[star_num]                      #  getting residue from PL relation
    res_del = Del_residue.iloc[star_num]                  #  residue from Del_M-Del_W plot (residue = extinction) (for all bands)
    pre_del = Del_predict.iloc[star_num]                  #  predicted value lies on the line of regression
    excess = data.EBV.iloc[star_num]              #  color excess of the star
    mod = data['mod'].iloc[star_num]              #  modulus of the star
    del_EBV_mean = data['del_EBV'].iloc[star_num]
    D_slope = Del_slope
    D_intercept = Del_intercept
    return name, period, excess, mod, res_W, res_M, res_del, del_EBV_mean, D_slope, D_intercept 

def show_del_E(k, pl):
    print(star(k)[0], '| period (logP) = ', star(k)[1], '| E_BV = ', star(k)[2], '| modulus = ', star(k)[3])
    print()
    if pl =='y':
        fig, ax1 = plt.subplots(figsize=(8,2))
        for i in range(0,nband):
            ax1.plot(i, excess.iloc[k][bands[i]], col_dot[i])
        ax1.axhline(y=data['del_EBV'].iloc[k], color='r', linestyle='--', label = '$mean = %f$'%(data['del_EBV'].iloc[k]))    
        ax1.set_xlabel('%s'%(star(k)[0]))
        ax1.set_ylabel('Reddening')
        ax1.legend()
        plt.show()
##############################################

excess = pd.DataFrame()
for i in range(0,nband):
    excess[bands[i]] = (Del_residue[bands[i]]/R[i])
data['del_EBV'] = excess['avg'] = excess.mean(axis=1)
int = input('Press Enter to see estimated error in color excess')
if int == '':
    print(excess)
#############################################
def plot_star(k, del_mu):
    mu_cor = pd.DataFrame()                                 # correction of each star
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel('$\Delta$ $\mu$ for %s'%(star(k)[0]))
    ax.set_ylabel('$\Delta$ $E_{BV}$')
    ax.axhline(y=0, color='y', linestyle='--')
    ax.axvline(x=0, color='y', linestyle='--')
    ax.invert_yaxis()
    for i in range(0,band):                                                             # loop for bands
        mu_cor['del_mu'] = del_mu                                                       # possible corrections in modulus
        mu_cor[W[1]] = star(k)[4][W[1]] - del_mu                                        # array select wesenheit residue WVI of kth star
        mu_cor[bands[i]] =  star(k)[5][bands[i]] - del_mu                               # PL residue + del_mu
        mu_cor['reg_'+bands[i]] = star(k)[8][i]*mu_cor[W[1]] + star(k)[8][i]            # predicting magnitude error for new modulus
        mu_cor['extinction_'+bands[i]] = (mu_cor[bands[i]] - mu_cor['reg_'+bands[i]])   # Extinction for each band
        ax.plot(del_mu, mu_cor['extinction_'+bands[i]]/R[i], col_lin[i], label = '%s band'%(m[i]))
        ax.plot(star(k)[4][W[1]], (star(k)[5][bands[i]] - Del_slope[i]*star(k)[4][W[1]] -  Del_intercept[i])/R[i], col_dot[i])
    ax.legend()
############################################


