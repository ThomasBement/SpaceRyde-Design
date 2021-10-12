# ---------------------------------------- #
# Isentropic [Python File]
# Written By: Thomas Bement
# Created On: 2021-10-08
# ---------------------------------------- #

"""IMPORTS"""

import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton

"""FUNCTIONS"""
def atm(h):
    if (h > 25000):
        T = -131 + (0.00299*h)
        P = 2.488*(((T + 273.1)/216.6)**-11.388)
    elif (h > 11000):
        T = -56.46
        P = 22.65*math.exp(1.73 - (0.000157*h))
    else:
        T = 15.40 - 0.00649*h
        P = 101.29*((T + 273.15)/288.08)**5.256
    return [(T + 273.15), (1000*P)]

coeff_lis = [1.05, -0.365, 0.85, -0.39]
def Cp_poly(temp, c0, c1, c2, c3, k):
    t = temp/1000
    Cp = 1000*(c0 + c1*t + c2*(t**2) + c3*(t**3))
    Cv = k/Cp
    R = Cp - Cv
    return [Cp, Cv, R]

def mach(chamb, throat, ex, K):
    chamb['mach'] = 0
    throat['mach'] = 1
    ex['mach'] = math.sqrt(((chamb['press']/ex['press'])**((K-1)/K) - 1)*(2/(K-1)))

def press(chamb, throat, ex, K):
    throat['press'] = chamb['press']/((1 + ((K-1)/2)*(throat['mach']**2))**(K/(K-1)))

def temp(chamb, throat, ex, K):
    throat['temp'] = chamb['temp']/(1 + ((K-1)/2)*(throat['mach']**2))
    ex['temp'] = chamb['temp']/(1 + ((K-1)/2)*(ex['mach']**2))

def area(chamb, throat, ex, K, T, R):
    throat['area'] = T/(((throat['press'])/math.sqrt(throat['temp']))*math.sqrt(K/R)*((1 + ((K - 1)/2))**((K + 1)/(-2*(K - 1))))*ex['c']*ex['mach'])
    ex['area'] = throat['area']*((1/ex['mach'])*((2 + (K - 1)*(ex['mach']**2))/(K + 1))**((K + 1)/(2*(K - 1))))

def mass_max(chamb, throat, ex, K, R):
    mdot = ((throat['area']*throat['press'])/math.sqrt(throat['temp']))*math.sqrt(K/R)*((1 + ((K - 1)/2))**((K + 1)/(-2*(K - 1))))
    chamb['mdot'] = mdot
    throat['mdot'] = mdot
    ex['mdot'] = mdot

def sound_speed(chamb, throat, ex, K):
    chamb['c'] = math.sqrt(K*Cp_poly(chamb['temp'], *coeff_lis, K)[2]*chamb['temp'])
    throat['c'] = math.sqrt(K*Cp_poly(throat['temp'], *coeff_lis, K)[2]*throat['temp'])
    ex['c'] = math.sqrt(K*Cp_poly(ex['temp'], *coeff_lis, K)[2]*ex['temp'])

def thrust(mdot, v, exit_press, atm_press, A):
    return (mdot*v) #+ (exit_press - atm_press)*A

def impulse(chamb, throat, ex, K, R, g = 9.81):
    return (ex['mach']*math.sqrt(K*R*ex['temp']))/g

def radius(x, r_throat, r_exit, alpha, beta):
    x_converge = r_throat/math.tan(beta) 
    x_diverge = (r_exit - r_throat)/math.tan(alpha)
    if (x <= x_converge):
        return (-1*math.tan(beta)*x) + 2*r_throat
    else:
        return (math.tan(alpha)*x) + r_throat*(1-(math.tan(alpha)/math.tan(beta)))
    
def area_ratio(mach, A, At, K):
    return ((1/mach)*(((2+(K-1)*(mach**2))/(K+1))**((K+1)/(2*(K-1))))) - (A/At)

def total_mass(chamb, throat, ex, alpha, beta, stress, den, K, L, burn_time, tank_press, mN2, boltz):
    r_throat = math.sqrt(throat['area']/math.pi)
    r_exit = math.sqrt(ex['area']/math.pi)
    r_chamb = radius(0, r_throat, r_exit, alpha, beta)
    chamb['r'] = r_chamb 
    x_converge = r_throat/math.tan(beta) 
    x_diverge = (r_exit - r_throat)/math.tan(alpha)

    l_converge = r_throat/math.sin(beta) 
    l_diverge = (r_exit - r_throat)/math.sin(alpha)

    chamb['thick'] = ((chamb['press']-ex['press'])*radius(0, r_throat, r_exit, alpha, beta))/stress
    throat['thick'] = ((throat['press']-ex['press'])*radius(x_converge, r_throat, r_exit, alpha, beta))/stress
    ex['thick'] = 0
    t = max([chamb['thick'], throat['thick'], ex['thick']])

    m_converge = t*(l_converge)*(2*math.pi*((r_throat + r_chamb)/2))*den
    m_diverge = t*(l_diverge)*(2*math.pi*((r_throat + r_exit)/2))*den

    m_nozz = m_converge + m_diverge

    tube_r = radius(0, r_throat, r_exit, alpha, beta)
    tube_t = ((chamb['press']-ex['press'])*tube_r)/stress
    m_tube = t*(L)*(2*math.pi*(tube_r/2))*den

    m_dot = throat['mdot']
    m_out = burn_time*m_dot
    Ns = m_out/mN2
    m_tank = (3*tank_press*den*Ns*boltz*chamb['temp'])/(2*stress*(tank_press - chamb['press']))
    return m_nozz + m_tube + m_tank

"""CONSTANTS"""
nom_T = 50
altitude = 30*(10**3)
atm_press = atm(altitude)[1]
K = 1.4
R = 296.8
mN2 = 28.0134/(1000*6.0221409E+23)
boltz = 1.38064852E-23
alpha = 15*(math.pi/180)
beta = 60*(math.pi/180)
CF = 0.9830

L = 2
ss_ty = 215E+6
ss_den = 7500

burn_time = 60
tank_press = 2E+7

chamb = {'temp': 300}
throat = {}
ex = {'press': atm_press} 

"""CALCULATIONS"""
steps = 200
press_rng = np.linspace(5000, 2000000, steps)

impulse_sp = []
mass = []
ratio = []
results = {'chamb': [], 'throat': [], 'ex': []}
for chamb_press in press_rng:
    chamb['press'] = chamb_press

    mach(chamb, throat, ex, K)
    press(chamb, throat, ex, K)
    temp(chamb, throat, ex, K)
    sound_speed(chamb, throat, ex, K)
    area(chamb, throat, ex, K, nom_T, R)
    mass_max(chamb, throat, ex, K, R)

    T = thrust(ex['mdot'], ex['c']*ex['mach'], ex['press'], atm_press, ex['area'])
    impulse_sp.append(impulse(chamb, throat, ex, K, R))

    mass.append(total_mass(chamb, throat, ex, alpha, beta, ss_ty, ss_den, K, L, burn_time, tank_press, mN2, boltz))
    
    results['chamb'].append(chamb)
    results['throat'].append(throat)
    results['ex'].append(ex)

impulse_sp = np.array(impulse_sp)
press = np.array(press_rng)
mass = np.array(mass)

mass_min = min(mass)
op_idx = np.where(mass == mass_min)[0][0]

plt.title('System Mass vs. Chamber Pressure')
plt.plot(press_rng, mass, marker='.')
plt.scatter([press_rng[op_idx]], [mass[op_idx]], color='r')
plt.xlabel('Chamber Pressure [Pa]')
plt.ylabel('System Mass [kg]')
plt.show()
plt.close()

print(atm(altitude))
print(results['chamb'][op_idx])
print(results['throat'][op_idx])
print(results['ex'][op_idx])