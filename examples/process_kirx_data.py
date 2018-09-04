import pandas as pd
import numpy as np
import os
import sys
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from estimates import (calculate_B_kino2014, calculate_B_marscher,
                       calculate_B_zdzr2015)
import astropy.units as u


freq_dict = {"c1": 4.60849, "c2": 5.00349, "x1": 8.10849, "x2": 8.42949,
             "u1": 15.3655, "k1": 23.8045, "q1": 43.2175}


def calculate_theta_m(nu_m):
    k = np.random.normal(0.91, 0.04, size=size)
    return 2*nu_m**(-1/k)


# Directory with files with data
data_dir = "/home/ilya/Dropbox/Bfield_estimates"

# Size os random samples to propagate uncertainties
size = 1000
# Fractional errors on values for spectral maximum
sigma_S_m_frac = 0.1
sigma_nu_m_frac = 0.1
# One estimate of superluminal speed
beta_app = 40.1
sigma_beta_app = 6.6
# Assuming sin(theta) = 1/Gamma
gamma = np.sqrt(np.random.normal(beta_app, sigma_beta_app, size=size)**2+1)
delta = gamma
# Using relation from MOJAVE to estimate half-open angle [rad]
hoangle = np.random.normal(0.13, 0.02)/gamma
z = 1.038
p = 2.5
# Distance where to estimate B using Zdziarski et al. method
h_u = 1*u.pc

# Calculate Kino's B estimates
df = pd.read_csv(os.path.join(data_dir, "kino_data.txt"),
                 names=["epoch", "nu_obs", "S_core", "sigma_S_core",
                        "theta_core", "sigma_theta_core", "limit", "resolved"],
                 sep='\s+', skiprows=1)

df_result = pd.DataFrame(columns=["epoch", "nu_obs", "lgBkino", "sigma_lgBkino",
                                  "resolved"])
for index, row in df.iterrows():
    S_core = np.random.normal(row["S_core"], row["sigma_S_core"], size=size)
    if row["resolved"]:
        theta_core = np.random.normal(row["theta_core"], row["sigma_theta_core"], size=size)
    else:
        theta_core = np.ones(size)*row["limit"]
    B_kino = calculate_B_kino2014(row["nu_obs"], theta_core, S_core, delta, z, p)
    df_result.loc[index] = [row["epoch"], row["nu_obs"],
                            np.mean(np.log10(B_kino)),
                            np.std(np.log10(B_kino)),
                            row["resolved"]]

df_result.to_csv(os.path.join(data_dir, "B_kino.txt"), sep=" ", index=False)



# Cacluate Marscher's estimates
df = pd.read_csv(os.path.join(data_dir, "marscher_data.txt"),
                 names=["epoch", "nu_m", "S_m"], sep="\s+", skiprows=1)
df_result = pd.DataFrame(columns=["epoch", "lgBmarscher", "sigma_lgBmarscher"])

for index, row in df.iterrows():
    theta_m = calculate_theta_m(row["nu_m"])
    nu_m = np.random.normal(row["nu_m"], row["nu_m"]*sigma_nu_m_frac, size=size)
    S_m = np.random.normal(row["S_m"], row["S_m"]*sigma_S_m_frac, size=size)
    B_marscher = calculate_B_marscher(nu_m, theta_m, S_m, delta, z, p)
    df_result.loc[index] = [row["epoch"], np.mean(np.log10(B_marscher)),
                            np.std(np.log10(B_marscher))]
df_result.to_csv(os.path.join(data_dir, "B_marscher.txt"), sep=" ", index=False)



# Calculate Zdzr's estimates
df = pd.read_csv(os.path.join(data_dir, "zdzr_data.txt"),
                 names=["epoch", "nu_1", "nu_2", "shift", "sigma_shift"],
                 sep="\s+", skiprows=1)
df_flux = pd.read_csv(os.path.join(data_dir, "kino_data.txt"),
                      names=["epoch", "nu_obs", "S_core", "sigma_S_core",
                             "theta_core", "sigma_theta_core", "limit",
                             "resolved"], sep='\s+', skiprows=1)
df_result = pd.DataFrame(columns=["epoch", "nu1", "nu2", "lgB1_zdzr",
                                  "sigma_lgB1_zdzr"])
for index, row in df.iterrows():

    nu_1, nu_2 = sorted([freq_dict[row["nu_1"]], freq_dict[row["nu_2"]]])
    # Find flux as mean of the core flux for two bands
    epoch = row["epoch"]
    df1 = df_flux[(df_flux["epoch"] == epoch) & (df_flux["nu_obs"] == nu_1)]
    flux1 = df1["S_core"].values[0]
    sigma_flux1 = df1["sigma_S_core"].values[0]
    df2 = df_flux[(df_flux["epoch"] == epoch) & (df_flux["nu_obs"] == nu_2)]
    flux2 = df2["S_core"].values[0]
    sigma_flux2 = df2["sigma_S_core"].values[0]
    flux = np.random.normal(0.5*(flux1+flux2),
                            np.hypot(sigma_flux1, sigma_flux2),
                            size=size)

    shift = np.random.normal(row["shift"], row["sigma_shift"], size=size)
    B_zdrz = calculate_B_zdzr2015(h_u, nu_1*u.GHz, nu_2*u.GHz, z, delta,
                                  shift*u.mas, hoangle, np.arcsin(1/gamma),
                                  flux*u.Jy, p=2)
    df_result.loc[index] = [epoch, nu_1, nu_2, np.nanmean(np.log10(B_zdrz)),
                            np.nanstd(np.log10(B_zdrz))]
df_result.to_csv(os.path.join(data_dir, "B_zdzr.txt"), sep=" ",
                 index=False)