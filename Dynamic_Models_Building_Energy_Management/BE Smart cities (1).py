

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

# Variables 
Va = 107.3
nb_branches = 28
nb_noeuds = 22

# Physical properties
material = {'Conductivity': [1.75, 0.25, 0.04, 0.952, 0.024, 0.2, 0.065],
        'Density': [2500.0, 825, 18, 1185, 1, 800, 2500],
        'Specific heat': [1008, 1008, 1030, 1080, 1005, 2385, 720]}

material = pd.DataFrame(material, index=['beton arme', 'BA13', 'laine de verre', 'parpaing de ciment', 'air', 'bois lourd','verre'])
# print("\n materials \n", material)

# Constantes

h=pd.DataFrame([{'in':4., 'out':10}])

# air intérieur de la pièce
air = {'Density': 1.2,
       'Specific heat': 1000}


wall1 = {'Conductivity': [material['Conductivity'][1], material['Conductivity'][4], material['Conductivity'][1]],
        'Density': [material['Density'][1], material['Density'][4], material['Density'][1]],
        'Specific heat': [material['Specific heat'][1], material['Specific heat'][4], material['Specific heat'][1]],
        'Width': [0.0125, 0.108, 0.0125],
        'Surface': 9.8046}

wall1 = pd.DataFrame(wall1, index=['BA13', 'air', 'BA13'])

# print("\n wall1 \n", wall1)

wall2 = {'Conductivity': [material['Conductivity'][3], material['Conductivity'][2], material['Conductivity'][1]],
        'Density': [material['Density'][3], material['Density'][2], material['Density'][1]],
        'Specific heat': [material['Specific heat'][3], material['Specific heat'][2], material['Specific heat'][1]],
        'Width': [0.2, 0.283-0.2-0.0125, 0.0125],
        'Surface': 23.8602-8.3232}

wall2 = pd.DataFrame(wall2, index=['parpaing de ciment', 'laine de verre', 'BA13'])

# print("\n wall2 \n", wall2)

wall3 = {'Conductivity': [material['Conductivity'][0], material['Conductivity'][2], material['Conductivity'][1]],
        'Density': [material['Density'][0], material['Density'][2], material['Density'][1]],
        'Specific heat': [material['Specific heat'][0], material['Specific heat'][2], material['Specific heat'][1]],
        'Width': [0.35, 0.429-0.35-0.0125, 0.0125],
        'Surface': 14.7966}

wall3 = pd.DataFrame(wall3, index=['beton arme', 'laine de verre', 'BA13'])

# print("\n wall3 \n", wall3)

wall4 = {'Conductivity': [material['Conductivity'][0], material['Conductivity'][2], material['Conductivity'][1]],
        'Density': [material['Density'][0], material['Density'][2], material['Density'][1]],
        'Specific heat': [material['Specific heat'][0], material['Specific heat'][2], material['Specific heat'][1]],
        'Width': [0.35, 0.429-0.35-0.0125, 0.0125],
        'Surface': 9.1312-2.7248}

wall4 = pd.DataFrame(wall4, index=['beton arme', 'laine de verre', 'BA13'])

# print("\n wall4 \n", wall4)

#Mur 5 et 6 assimilé au même mur

wall5 = {'Conductivity': [material['Conductivity'][1], material['Conductivity'][4], material['Conductivity'][1]],
        'Density': [material['Density'][1], material['Density'][4], material['Density'][1]],
        'Specific heat': [material['Specific heat'][1], material['Specific heat'][4], material['Specific heat'][1]],
        'Width': [0.0125, (0.094-2*0.0125+0.133-2*0.0125)/2, 0.0125],
        'Surface': 4.94-2.01564+14.703}

wall5 = pd.DataFrame(wall5, index=['BA13', 'isolant', 'BA13'])

# print("\n wall5 \n", wall5)


entrydoor = {'Conductivity': material['Conductivity'][5],
        'Density': material['Density'][5],
        'Specific heat': material['Specific heat'][5],
        'Width': 0.04,
        'Surface': 2.7248}

storagedoor = {'Conductivity': material['Conductivity'][5],
        'Density': material['Density'][5],
        'Specific heat': material['Specific heat'][5],
        'Width': 0.039,
        'Surface': 2.01564}

window = {'Conductivity': material['Conductivity'][6],
        'Density': material['Density'][6],
        'Specific heat': material['Specific heat'][6],
        'Width': 0.015,
        'Surface': 8.3232}

#Le BA13 a été négligé dans le modèle (il n'y a qu'un isolant par mur)

#Calcul des matrices

#Code matrice A
A = np.zeros([nb_branches,nb_noeuds])
A[0,0]=1
A[1,0],A[1,1]=-1,1
A[2,1],A[2,2]=-1,1
A[3,2],A[3,3]=-1,1
A[4,3],A[4,4]=-1,1
A[5,4],A[5,20]=-1,1
A[1,0],A[1,1]=-1,1

A[6,5]=1
A[7,5],A[7,6]=-1,1
A[8,6],A[8,7]=-1,1
A[9,7],A[9,8]=-1,1
A[10,8],A[10,9]=-1,1
A[11,9],A[11,20]=-1,1

A[12,10]=1
A[13,10],A[13,11]=-1,1
A[14,11],A[14,12]=-1,1
A[15,12],A[15,13]=-1,1
A[16,13],A[16,14]=-1,1
A[17,14],A[17,20]=-1,1
A[18,15],A[18,20]=-1,1

A[19,15],A[19,16]=-1,1
A[20,16],A[20,17]=-1,1
A[21, 17], A[21, 18] = -1, 1
A[22,18],A[22,19]=-1,1
A[23,19],A[23,21]=-1,1
A[24,20],A[24,21]=-1,1
A[25,20]=1
A[26,20]=1
A[27,20]=1

#Code matrice G
G=np.zeros([nb_branches,nb_branches])

#Calcul des conductance de conduction par mur :
    
#Wall 2

G[0,0]=h['out']*wall2['Surface'][0]
G[1,1]=wall2['Conductivity'][0]/(((wall2['Width'][0])/2)*wall2['Surface'][0])
G[2,2]=G[1,1]
G[3,3]=wall2['Conductivity'][1]/(((wall2['Width'][1])/2)*wall2['Surface'][1])
G[4,4]=G[3,3]
G[5,5]=h['in']*wall2['Surface'][0]

#Wall 3

G[6,6]=h['out']*wall3['Surface'][0]
G[7,7]=wall3['Conductivity'][0]/(((wall3['Width'][0])/2)*wall3['Surface'][0])
G[8,8]=G[7,7]
G[9,9]=wall3['Conductivity'][1]/(((wall3['Width'][1])/2)*wall3['Surface'][1])
G[10,10]=G[9,9]
G[11,11]=h['in']*wall3['Surface'][0]

#Wall 4

G[12,12]=h['out']*wall4['Surface'][0]
G[13,13]=wall4['Conductivity'][0]/(((wall4['Width'][0])/2)*wall4['Surface'][0])
G[14,14]=G[13,13]
G[15,15]=wall4['Conductivity'][1]/(((wall4['Width'][1])/2)*wall4['Surface'][1])
G[16,16]=G[15,15]
G[17,17]=h['in']*wall4['Surface'][0]

#Wall 5

G[18,18]=h['out']*wall5['Surface'][0]
G[19,19]=wall5['Conductivity'][0]/(((wall5['Width'][0])/2)*wall5['Surface'][0])
G[20,20]=G[19,19]
G[21,21]=wall4['Conductivity'][1]/(((wall5['Width'][1])/2)*wall5['Surface'][1])
G[22,22]=G[21,21]
G[23,23]=h['in']*wall4['Surface'][0]

# Door
G[24,24]=1/(1/(h['in']*entrydoor['Surface']))+1/(entrydoor['Conductivity']/((entrydoor['Width'])*entrydoor['Surface']))
             
G[25,25]=1/(1/(h['in']*storagedoor['Surface']))+1/(storagedoor['Conductivity']/((storagedoor['Width'])*storagedoor['Surface']))


# window

G[26,26]=1/(1/(h['in']*window['Surface']))+1/(window['Conductivity']/((window['Width'])*window['Surface']))

# HVAC
Kp = 1000
G[27,27] = Kp

#matrice b
T0 = 19.5
TH = 16.4
Tventil = 20.0
b = np.zeros(nb_branches)

b[0] = b[6] = b[26] = T0
b[12] = b[25] = TH
b[27] = Tventil
#b[[0, 6, 12]]= 1

σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

ε_wLW = 0.9     # long wave wall emmisivity (concrete)
α_wSW = 0.2     # absortivity white surface
ε_gLW = 0.9     # long wave glass emmisivity (glass pyrex)
τ_gSW = 0.83    # short wave glass transmitance (glass)
α_gSW = 0.1     # short wave glass absortivity


#Calcul de C
C2 = wall2['Density'] * wall2['Specific heat'] * wall2['Surface'] * wall2['Width']
C3 = wall3['Density'] * wall3['Specific heat'] * wall3['Surface'] * wall3['Width']
C4 = wall4['Density'] * wall4['Specific heat'] * wall4['Surface'] * wall4['Width']
C5 = wall5['Density'] * wall5['Specific heat'] * wall5['Surface'] * wall5['Width']
Cair = air['Density'] * air['Specific heat'] * Va

C = np.diag([0, C2[0], 0, C2[1], 0, 0, C3[0], 0, C3[1], 0, 0, C4[0], 0, C4[1], 0, 0, C5[0], 0, C5[1], 0, Cair, 0])


# #matrice f
# f=np.zeros(22)
# f[[0,4,5,9,10,20]] = np.array([1000,1000,1000,500,500,4000])
# #f[[0,4,5,9,10,20]] = 1
# #vecteur y
# y=np.ones(22)

f=np.zeros(nb_noeuds)

f[[0,4,5,9,10,14,20]] = np.array([1000,1000,1000,500,500,500,4000])

#f[[0,4,5,9,10,20]] = 1

#vecteur y
y = np.ones(nb_noeuds)

#vecteur u
u = np.hstack([b[np.nonzero(b)], f[np.nonzero(f)]])


### State Space Model ###
[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)

yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
ytc = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)

# print(np.array_str(yss, precision=3, suppress_small=True))
# print(np.array_str(ytc, precision=3, suppress_small=True))
# print(f'Max error in steady-state between thermal circuit and state-space:\
#  {max(abs(yss - ytc)):.2e}')


### Dynamic Simulation ###
y = np.zeros(nb_noeuds)
y[[20,21]] = 1
[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)

yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u

print(np.array_str(yss, precision=3, suppress_small=True))



### Time step ###
dtmax = min(-2./np.linalg.eig(As)[0])
print(f'Maximum time step: {dtmax:.2f} s')
dt = 50


### Step response ###
duration = 3600 * 24 * 2

# Number of steps
n = int(np.floor(duration/dt))
t = np.arange(0,n*dt,dt) #time

# Vectors of state and input (in time)
n_tC = As.shape[0]              # no of state variables (temps with capacity)
# u = [To To To Tsp Phio Phii Qaux Phia]
u = np.zeros([13, n])
u[0:6, :] = np.ones([6, n])

temp_exp = np.zeros([n_tC, t.shape[0]])
temp_imp = np.zeros([n_tC, t.shape[0]])

I = np.eye(n_tC)
for k in range(n - 1):
    temp_exp[:, k + 1] = (I + dt * As) @\
        temp_exp[:, k] + dt * Bs @ u[:, k]
    temp_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (temp_imp[:, k] + dt * Bs @ u[:, k])
       

y_exp = Cs @ temp_exp + Ds @  u
y_imp = Cs @ temp_imp + Ds @  u

fig, ax = plt.subplots()
ax.plot(t/60, y_exp.T, t/60, y_imp.T)
ax.set(xlabel='Time [min]',
        ylabel='$T_i$ [°C]',
        title='Step input: To = 1°C')
# plt.show()

b = np.zeros(nb_branches)

b[0] = b[6] = b[26] = 1
b[12] = b[25] = 1
b[27] = 1

f=np.zeros(nb_noeuds)

ytc = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {ytc[20]:.4f} °C')
print('- response to step input:', y_exp[0,len(y_exp[0])-1], '°C')


### Simulation with weather station ###

filename = 'FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2004-2018.epw'
start_date = '2017-07-03 12:00:00'
end_date = '2017-08-05 18:00:00'

# Read weather data from Energyplus .epw file
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2017))
weather = weather[(weather.index >= start_date) & (
    weather.index < end_date)]

surface_orientation2 = {'slope': 90,
                        'azimuth': 0,
                        'latitude': 45}
albedo = 0.5
rad_surf2 = dm4bem.sol_rad_tilt_surf(weather, surface_orientation2, albedo)
rad_surf2['Qray0'] = rad_surf2.sum(axis=1)

surface_orientation3 = {'slope': 90,
                        'azimuth': 270,
                        'latitude': 45}
albedo = 0.5
rad_surf3 = dm4bem.sol_rad_tilt_surf(weather, surface_orientation3, albedo)
rad_surf3['Qray5'] = rad_surf3.sum(axis=1)

data = pd.concat([weather['temp_air'], rad_surf2['Qray0'],rad_surf3['Qray5']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'T0'})


data['Tventil'] = 20 * np.ones(data.shape[0])
data['TH'] = 16.4 * np.ones(data.shape[0])
data['Q4'] = 1000 * np.ones(data.shape[0])
data['Q9'] = 500 * np.ones(data.shape[0])
data['Q10'] = 500 * np.ones(data.shape[0])
data['Q14'] = 500 * np.ones(data.shape[0])
data['Q20'] = 4000 * np.ones(data.shape[0])

t = dt * np.arange(data.shape[0])

u = pd.concat([data['T0'], data['T0'], data['T0'],data['TH'],data['TH'], data['Tventil'],
                α_wSW * wall2['Surface']['parpaing de ciment'] * data['Qray0'],  data['Q4']
                ,α_wSW * wall3['Surface']['beton arme'] * data['Qray5'], data['Q9'], data['Q10'], data['Q14'], data['Q20']], axis=1)




temp_exp = 20 * np.ones([As.shape[0], u.shape[0]])
y_exp = Cs @ temp_exp + Ds @ u.to_numpy().T
q_HVAC = Kp * (data['Tventil'] - y_exp[0, :])

# fig, axs = plt.subplots(2, 1)
# # plot indoor and outdoor temperature
# axs[0].plot(t / (3600 * 24), y_exp[0, :], label='$T_{indoor}$')
# axs[0].plot(t / (3600 * 24), data['T0'], label='$T_{outdoor}$')
# axs[0].set(xlabel='Time [days]',
#             ylabel='Temperatures [°C]',
#             title='Simulation for weather')
# axs[0].legend(loc='upper right')

# # plot total solar radiation and HVAC heat flow
# axs[1].plot(t / (3600 * 24),  q_HVAC, label='$q_{HVAC}$')
# axs[1].plot(t / (3600 * 24), data['Qray0'], label='$Φ_{ray0}$')
# axs[1].plot(t / (3600 * 24), data['Qray5'], label='$Φ_{ray4}$')
# axs[1].plot(t / (3600 * 24), data['Qray5'], label='$Φ_{ray5}$')
# axs[1].plot(t / (3600 * 24), data['Qray5'], label='$Φ_{ray10}$')
# axs[1].plot(t / (3600 * 24), data['Qray5'], label='$Φ_{ray14}$')
# axs[1].set(xlabel='Time [days]',
#             ylabel='Heat flows [W]')
# axs[1].legend(loc='upper right')

# fig.tight_layout()
# for k in range(u.shape[0] - 1):
#     temp_exp[:, k + 1] = (I + dt * As) @ temp_exp[:, k]\
#         + dt * Bs @ u.iloc[k, :]




