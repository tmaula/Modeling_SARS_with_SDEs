import numpy as np
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#import SIMPLIFIED_CORRMATRIX as SDE
import COMPLETE_CORRELATION_MATRIX as SDE
import sys

## DEFINE GLOBAL VARIABLES
global V, k1, k2p, k2m, k3p, k3m, k4, k5, k6, k7, k8, k9, k10p, k10m, k11p, k11m, k12p, k12m, k13p, k13m

##### CHOICES #####
"""
0 := RURAL
1 := SUBURBAN
2 := URBAN
"""
choice = int(sys.argv[1])
name = sys.argv[2]

# import STOCHASTIC_TERM as Stoch

# DEFINE SIMULATION TIMES
Nsteps = 17000
stepsize = 0.01
time = [0]*Nsteps
det = [0]*Nsteps
sto = [0]*Nsteps
Ctot = [0]*Nsteps

# DENSITY DEPENDENT INFECTION RATE
k1_rural = 0.020
k1_suburban = 0.059
k1_urban = 0.25
    
# DEFINE RATES CONSTANTS #
k1 = 0.25 #0.06 #INFECTION
k2p = 0.032 #0.04 #RECOVERY
k2m = 0#.02*k1 #RELAPSE
k3p = 0# 4.94 #VACCINATION
k3m = 0 # INEFFECTIVE VACCINATION
k4 = k3p # VACCINATION
k5 = 0.0279 # DEATH RATE
k6 = k4 #VACCINATION
k7 = 0 #HOSPITALIZATION
k8 = 0.1*k3m*k5 #DEATH RATE OF HOSPITALIZED
k9 = 0 # RELEASE OF HOSPITALIZED
k10p = 0.00
k10m = 0.00
k11p = 0.00
k11m = 0.00
k12p = 0.000
k12m = 0.000
k13p = 0.000
k13m = 0.000
krates = [k1, k2p, k2m, k3p, k3m, k4, k5, k6, k7, k8, k9, k10p, k10m, k11p, k11m, k12p, k12m, k13p, k13m]

# DEFINE VOLUMES
Vmat = [1,1,1]
V = Vmat[choice]

# DEFINE POPULATION NUMBERS
ninit = [10,3*10, 12*10] 
n1 = ninit[choice] # susceptible
n2 = 1 # infected
n3 = 0 # recovered
n4 = 0 # vaccinated
n5 = 0 # dead
n6 = 0 # hospitalized

# DEFINE CONCENTRATIONS
c1 = [0]*Nsteps; c1[0] = n1/V
c2 = [0]*Nsteps; c2[0] = n2/V
c3 = [0]*Nsteps; c3[0] = n3/V
c4 = [0]*Nsteps; c4[0] = n4/V
c5 = [0]*Nsteps; c5[0] = n5/V
c6 = [0]*Nsteps; c6[0] = n6/V
Ctot[0] = c1[0]+c2[0]+c3[0]+c4[0]+c5[0]+c6[0]
det[0] = 0
# DEFINE DETERMINISTIC PARTS OF SDE (in terms of concentration)
def F1(x, t):
    x1, x2, x3, x4, x5, x6 = x
    return V*(-k1*x1 - k4*x1 - k10p*x1 + k10m) 
def F2(x, t):
    x1, x2, x3, x4, x5, x6 = x
    return V*(k1*x1 - (k2p*x2 - k2m*x3) - (k3p*x2 - k3m*x4) - (k5*x2) - k7*x2 - (k11p*x2 - k11m))
def F3(x, t):
    x1, x2, x3, x4, x5, x6 = x
    return V*((k2p*x2 - k2m*x3) - k6*x3 - (k13p*x3 - k13m))
def F4(x, t):
    x1, x2, x3, x4, x5, x6 = x
    return V*(k3p*x2 - k3m*x4 + k4*x1 + k6*x3 + k9*x6 - (k12p*x4 - k12m))
def F5(x, t):
    x1, x2, x3, x4, x5, x6 = x
    return V*(k5*x2 + k8*x6)
def F6(x, t):
    x1, x2, x3, x4, x5, x6 = x
    return V*(k7*x2 - k8*x6 - k9*x6)

def bigFFF(x, t):
    global V, k1, k2p, k2m, k3p, k3m, k4, k5, k6, k7, k8, k9, k10p, k10m, k11p, k11m, k12p, k12m, k13p, k13m
    d1 = F1(x,t)
    d2 = F2(x,t)
    d3 = F3(x,t)
    d4 = F4(x,t)
    d5 = F5(x,t)
    d6 = F6(x,t)
    return [d1, d2, d3, d4, d5, d6]

# RUN SIMULATIONS
for i in range(0,Nsteps-1):
    time[i+1] = time[i]+stepsize
    #if c2[i] <= 0:
    #    continue
    xi1, xi2, xi3, xi4, xi5, xi6 = np.random.normal(0,1,6)
    XI = [xi1, xi2, xi3, xi4, xi5, xi6]
    time[i+1] = time[i]+stepsize
    conc = [c1[i], c2[i], c3[i], c4[i], c5[i], c6[i]]

    det1 = stepsize*F1(conc, time)
    det2 = stepsize*F2(conc, time)
    det3 = stepsize*F3(conc, time)
    det4 = stepsize*F4(conc, time)
    det5 = stepsize*F5(conc, time)
    det6 = stepsize*F6(conc, time)

    sto1 = SDE.WHITE_NOISE(krates, conc, XI, 1)*np.sqrt(stepsize)*V
    sto2 = SDE.WHITE_NOISE(krates, conc, XI, 2)*np.sqrt(stepsize)*V
    sto3 = SDE.WHITE_NOISE(krates, conc, XI, 3)*np.sqrt(stepsize)*V
    sto4 = SDE.WHITE_NOISE(krates, conc, XI, 4)*np.sqrt(stepsize)*V
    sto5 = SDE.WHITE_NOISE(krates, conc, XI, 5)*np.sqrt(stepsize)*V
    sto6 = SDE.WHITE_NOISE(krates, conc, XI, 6)*np.sqrt(stepsize)*V
    
    c1[i+1] = c1[i] + det1 - sto1
    c2[i+1] = c2[i] + det2 - sto2
    c3[i+1] = c3[i] + det3 - sto3
    c4[i+1] = c4[i] + det4 - sto4
    c5[i+1] = c5[i] + det5 - sto5
    c6[i+1] = c6[i] + det6 - sto6
    if c2[i] <= 0:
        c2[i+1] = c2[i]
    det[i+1] = det1+det2+det3+det4+det5+det6
    sto[i+1] = sto1+sto2+sto3+sto4+sto5+sto6
    Ctot[i+1] = c1[i+1]+c2[i+1]+c3[i+1]+c4[i+1]+c5[i+1]+c6[i+1]
    
# CALCULATE ODES!!!
myODE = odeint(bigFFF, [c1[0], c2[0], c3[0], c4[0], c5[0], c6[0]], time)
c1_ode = myODE[:, 0]
c2_ode = myODE[:, 1]
c3_ode = myODE[:, 2]
c4_ode = myODE[:, 3]
c5_ode = myODE[:, 4]
c6_ode = myODE[:, 5]

print 

# PLOT RESULTS
fig, ax = plt.subplots(2,3, sharex=True)
ax[0,0].plot(time, c1_ode, ls='--', label='Susceptible', linewidth=3.0, color='k')
ax[0,1].plot(time, c2_ode, ls='--', label='Infected', linewidth=3.0,color='k')
ax[0,2].plot(time, c3_ode, ls='--', label='Recovered', linewidth=3.0, color='k')
ax[1,0].plot(time, c4_ode, ls='--', label='Vaccinated', linewidth=3.0, color='k')
ax[1,1].plot(time, c5_ode, ls='--', label='Dead', linewidth=3.0, color='k')
ax[1,2].plot(time, c6_ode, ls='--', label='Hospitalized', linewidth=3.0, color='k')

ax[0,0].plot(time, c1, ls='-', label='Susceptible', linewidth=3.0, color='g', alpha=0.5)

ax[0,1].plot(time, c2, ls='-', label='Infected', linewidth=3.0,color='peru', alpha=0.5)
ax[0,2].plot(time, c3, ls='-', label='Recovered', linewidth=3.0, color='b', alpha=0.5)
ax[1,0].plot(time, c4, ls='-', label='Vaccinated', linewidth=3.0, color='magenta', alpha=0.5)
ax[1,1].plot(time, c5, ls='-', label='Dead', linewidth=3.0, color='r', alpha=0.5)
ax[1,2].plot(time, c6, ls='-', label='Hospitalized', linewidth=3.0, color='cyan', alpha=0.5)

ax[1,0].set_xlabel('Days')
ax[1,1].set_xlabel('Days')
ax[1,2].set_xlabel('Days')
ax[0,0].set_ylabel('Susceptible (1/km)')
ax[0,1].set_ylabel('Infected (1/km)')
ax[0,2].set_ylabel('Recovered (1/km)')
ax[1,0].set_ylabel('Vaccinated (1/km)')
ax[1,1].set_ylabel('Dead (1/km)')
ax[1,2].set_ylabel('Hospitalized(1/km)')
plt.show(); plt.close()

fig, ax = plt.subplots(1,3)
ax[0].plot(time,det)
ax[1].plot(time,sto)
ax[2].plot(time, Ctot)
x = sum(Ctot)/len(Ctot)
ax[2].plot(time, [x]*(Nsteps))
plt.show()


PDF1, bins1 = np.histogram(c1,bins=30, density=True)
PDF2, bins2 = np.histogram(c2,bins=30, density=True)
PDF3, bins3 = np.histogram(c3,bins=30, density=True)
PDF4, bins4 = np.histogram(c4,bins=30, density=True)
PDF5, bins5 = np.histogram(c5,bins=30, density=True)
PDF6, bins6 = np.histogram(c6,bins=30, density=True)

def mean_bins(bins):
    Bins = [0]*(len(bins)-1)
    for i in range(0, len(bins)-1):
        Bins[i] = (bins[i]+bins[i+1])/2
    return Bins
print (PDF1)
Bins1 = mean_bins(bins1)
Bins2 = mean_bins(bins2)
Bins3 = mean_bins(bins3)
Bins4 = mean_bins(bins4)
Bins5 = mean_bins(bins5)
Bins6 = mean_bins(bins6)

fig, ax = plt.subplots(2,3, sharex=False)
ax[0,0].plot(Bins1, PDF1, ls='-', label='Susceptible', marker='o', color='g')
ax[0,1].plot(Bins2, PDF2, ls='-', label='Infected', marker='o',color='peru')
ax[0,2].plot(Bins3, PDF3, ls='-', label='Recovered', marker='o', color='b')
ax[1,0].plot(Bins4, PDF4, ls='-', label='Vaccinated', marker='o', color='magenta')
ax[1,1].plot(Bins5, PDF5, ls='-', label='Dead', marker='o', color='r')
ax[1,2].plot(Bins6, PDF6, ls='-', label='Hospitalized', marker='o', color='cyan')
plt.show()


concentrations = (np.array([c1, c2, c3, c4, c5, c6])).T
np.savetxt(name, concentrations)
