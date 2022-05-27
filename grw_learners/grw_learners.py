# -*- coding: utf-8 -*-

import os
import sys
from contextlib import contextmanager
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt 
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import pickle

#####plotting parameters
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 18})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

# @contextmanager
# def suppress_stdout():
#     with open(os.devnull, "w") as devnull:
#         old_stdout = sys.stdout
#         sys.stdout = devnull
#         try:  
#             yield
#         finally:
#             sys.stdout = old_stdout

# ############################## Import and prepare epochs ######################
os.chdir("/grw_learners/")

# files_l = glob.glob("/epochs_l/*t4_epo.fif")
# files = files_l

# ########### Load Learners ################
# epochs = []
# event = '54'
# no1 = ['1','10']
# no2 = ['2','20']
# no3 = ['3','30']
# # chans = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
# #          'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'] #reduce to biosemi 16 array
# amps_t = [] #target tone (falling) from learners
# amps_n1 = []
# amps_n2 = []
# amps_n3 = []
# print("loading epochs...")
# for file in tqdm(files):
#     with suppress_stdout():
#         epo = mne.read_epochs(file, proj=True, preload=True, verbose=None)
#         epo.apply_baseline(baseline=(None, 0), verbose=None) #apply mean subtraction baseline
#         chans = epo.info.ch_names[:-5]
#         amps_t.append(epo[event].get_data(picks=chans).mean(axis=0)*1e6)
#         amps_n1.append(epo[no1].get_data(picks=chans).mean(axis=0)*1e6)
#         amps_n2.append(epo[no2].get_data(picks=chans).mean(axis=0)*1e6)
#         amps_n3.append(epo[no3].get_data(picks=chans).mean(axis=0)*1e6)

# amps_t = np.array(amps_t)
# amps_n1 = np.array(amps_n1)
# amps_n2 = np.array(amps_n2)
# amps_n3 = np.array(amps_n3)


# amps = np.stack([amps_t,  amps_n1, amps_n2, amps_n3])
# amps = np.swapaxes(amps,0,1)

# amps = amps.mean(axis=0)

# times = epo.times

# np.save('learners_averaged_epochs.npy', amps)
# np.save('times.npy', times)

amps = np.load('/data/learners_averaged_epochs.npy')

times = np.load('/grw_learners/data/times.npy')

C = amps.shape[0] #number of conditions C
E = amps.shape[1] #number of electrodes E 
S = amps.shape[2] #number of time-samples S

ts = np.arange(S)/256

from pymc3.distributions.timeseries import GaussianRandomWalk as GRW

#### Model Learners
with pm.Model() as mod:
    # create a theano shared variable of observed voltages, may be handy for predictions
    y_obs = pm.Data('y_obs', amps)
    # covariances: diagonal matrices of ones, 1 matrix per tone (i.e. 4 total)
    Σ = [tt.eye(E) for c in range(C)]
    # Gaussian random walks (GRWs) 1 per tone
    g = [GRW("g"+str(c+1), shape=(S,E)) for c in range(C)]
    # Multivariate GRWs, 1 per tone
    x = [pm.Deterministic('x'+str(c+1), tt.dot(Σ[c], g[c].T)) for c in range(C)]
    # S (282) by E (32) by C (4) matrix (beta parameter), i.e. samples by electrode by condition(tone)
    B = pm.Deterministic("B", tt.stack(x))
    # intercept of stationary Gaussian noise across samples S (i.e. time-samples)
    α = pm.Normal('α', mu=0, sigma=0.05, shape=(S))
    # Estimate location parameter for Normal distribution likelihood y
    μ = pm.Deterministic('μ', α + B)
    # Estimate error parameter for likelihood y
    σ = pm.HalfNormal('σ', 0.05)+1
    # Likelihood or distribition for observed variable: amplitudes/volatges SxExC
    y = pm.Normal("y", mu=μ, sigma=σ, observed=y_obs) 
    
with mod:
    trace = pm.sample(1000, tune=1000, cores=8, chains=4, init='adapt_diag', target_accept=0.9)

    
tracedir = "/grw_learners/trace/"
pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)

# with mod:
#     trace = pm.load_trace(tracedir)

###### Plot Posteriors #####
fig, axs = plt.subplots(2,2, figsize=(14,10))
for i in range(3):
    if i == 0:
        ax = axs[0,0]
        t = 1
        c='teal'
    if i == 1:
        ax = axs[0,1]#
        t = 2
        c='limegreen'
    if i == 2:
        ax = axs[1,0]
        t = 3
        c='sienna'
    odiff = amps[0,12,:]-amps[t,12,:]
    pdiff = trace['μ'][:,0,12,:]-trace['μ'][:,t,12,:]
    postm = pdiff.mean(axis=0)
    posth5, posth95 = az.hdi(pdiff, hdi_prob=0.9).T
    ax.set_ylim([-3,9])
    ax.grid(alpha=0.2, zorder=-1)
    ax.axvline(0, color='k', zorder=-1, linestyle=':')
    ax.axhline(0, color='k', zorder=-1, linestyle=':')
    ax.plot(times, odiff, alpha=0.3, color='k', label="observed voltage")
    ax.plot(times, postm, color=c, label="posterior mean")
    ax.fill_between(times, posth5, posth95, color=c, alpha=0.3, label="90% HDI")
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=16, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Pz: Tone4 - Tone'+str(i+1))
axs[1,1].axis("off")
plt.tight_layout()
plt.savefig('posteriors_learners.png', dpi=300)
plt.close()

###### Plot Predictions #####
with mod:
    preds = pm.sample_posterior_predictive(trace)

fig, axs = plt.subplots(2,2, figsize=(14,10))
for i in range(3):
    if i == 0:
        ax = axs[0,0]
        t = 1
        c='teal'
    if i == 1:
        ax = axs[0,1]#
        t = 2
        c='limegreen'
    if i == 2:
        ax = axs[1,0]
        t = 3
        c='sienna'
    odiff = amps[0,12,:]-amps[t,12,:]
    pdiff = preds['y'][:,0,12,:]-preds['y'][:,t,12,:]
    predm = pdiff.mean(axis=0)
    pred_sdl = predm - pdiff.std(axis=0)
    pred_sdh = predm + pdiff.std(axis=0)
    ax.set_ylim([-3,9])
    ax.grid(alpha=0.2, zorder=-1)
    ax.axvline(0, color='k', zorder=-1, linestyle=':')
    ax.axhline(0, color='k', zorder=-1, linestyle=':')
    ax.plot(times, odiff, alpha=0.3, color='k', label="observed voltage")
    ax.plot(times, predm, color=c, label="predicted mean")
    ax.fill_between(times, pred_sdl, pred_sdh, color=c, alpha=0.3, label="predicted SD")
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=16, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Pz: Tone4 - Tone'+str(i+1))
axs[1,1].axis("off")
plt.tight_layout()
plt.savefig('predictions_learners.png', dpi=300)
plt.close()


###### Plot Topomaps #####
non_targets = np.array([trace['μ'][:,1,:,:],trace['μ'][:,2,:,:],trace['μ'][:,3,:,:]]).mean(axis=0)
pdiff = trace['μ'][:,0,:,:]-non_targets
#pdiff = pdiff[:,:,77:205].mean(axis=2)
mdiff = pdiff.mean(axis=0)
h5diff,h95diff = np.array([az.hdi(pdiff[:,e,:], hdi_prob=0.9) for e in range(E)]).T

# info = epo.average(picks='eeg').info
# with open('info.pickle', 'wb') as handle:
#     pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
info_path = "/grw_learners/data/info.pickle"
with open(info_path, 'rb') as handle:
    info = pickle.load(handle)
    
h5ev = mne.EvokedArray(h5diff.T, info)
mev = mne.EvokedArray(mdiff, info)
h95ev = mne.EvokedArray(h95diff.T, info)

selt = [0.2,0.4,0.6,0.8]

mne.viz.plot_evoked_topomap(h5ev, times=selt,scalings=1, show=False)
plt.savefig('topomap_learners_h5.png', dpi=300)
plt.close()
mne.viz.plot_evoked_topomap(mev, times=selt,scalings=1, show=False)
plt.savefig('topomap_learners_mean.png', dpi=300)
plt.close()
mne.viz.plot_evoked_topomap(h95ev, times=selt,scalings=1, vmin=-5, vmax=5, show=False)
plt.savefig('topomap_learners_h95.png', dpi=300)
plt.close()



#############################################

######### Save summaries ##########
summ = az.summary(trace, hdi_prob=0.9, round_to=4)
summ = pd.DataFrame(summ)
summ.to_csv('summary.csv')
print("summary saved")

bfmi = az.bfmi(trace)
bfmi = pd.DataFrame(bfmi)
bfmi.to_csv('bfmi.csv')
print("bfmi saved") 

ener = az.plot_energy(trace)
plt.savefig("energy.png", dpi=300)
plt.close()

########### Model fit

loo = az.loo(trace, pointwise=True)
loo = pd.DataFrame(loo)
loo.to_csv("loo.csv")

waic = az.waic(trace, pointwise=True)
waic = pd.DataFrame(waic)
waic.to_csv('waic.csv')

###plot rank
path = "/grw_learners/tranks/"
varias = [v for v in trace.varnames if not "__" in v]
for var in tqdm(varias):
    err = az.plot_rank(trace, var_names=[var], kind='vlines', ref_line=True,
                       vlines_kwargs={'lw':1}, marker_vlines_kwargs={'lw':2})
    plt.savefig(path+var+'_trank.png', dpi=300)
    plt.close()
