import astropy.constants as const
import astropy.units as u
from astropy.cosmology import WMAP9
import numpy as np


def calculate_B_kino2014(nu_ssa_ghz, theta_obs_mas, flux_nu_ssa_obs_jy, delta,
                         z, p):
    """
    Calculate B using (11) from Kino et al. 2014 (doi:10.1088/0004-637X/786/1/5).

    :param nu_ssa_ghz:
        Frequency of observation (it is SSA frequency for radio core).
    :param theta_obs_mas:
        Width of the VLBI core (see paper for discussion).
    :param flux_nu_ssa_obs_jy:
        Flux of the core.
    :param delta:
        Doppler factor.
    :param z:
        Redshift.
    :param p:
        Exponent of particles energy distribution. Must be 2.5, 3 or 3.5 here.
    :return:
        Value of B [G].
    """
    if p not in (2.5, 3.0, 3.5):
        raise Exception("p must be 2.5, 3.0 or 3.5")
    b_p = {2.5: 3.3*10**(-5), 3.0: 1.9*10**(-5), 3.5: 1.2*10**(-5)}
    return b_p[p] * nu_ssa_ghz**5 * theta_obs_mas**4 * flux_nu_ssa_obs_jy**(-2) * delta / (1+z)


def calculate_K_kino2014(nu_ssa_ghz, theta_obs_mas, flux_nu_ssa_obs_jy, delta,
                         z, p):
    """
    Calculate B using (11) from Kino et al. 2014 (doi:10.1088/0004-637X/786/1/5).

    :param nu_ssa_ghz:
        Frequency of observation (it is SSA frequency for radio core).
    :param theta_obs_mas:
        Width of the VLBI core (see paper for discussion).
    :param flux_nu_ssa_obs_jy:
        Flux of the core.
    :param delta:
        Doppler factor.
    :param z:
        Redshift.
    :param p:
        Exponent of particles energy distribution. Must be 2.5, 3 or 3.5 here.
    :return:
        Value of the K_e - normalization factor of the electron energy density
        distribution [erg**(p-1) * cm**(-3].
    """
    if p not in (2.5, 3.0, 3.5):
        raise Exception("p must be 2.5, 3.0 or 3.5")
    k_p = {2.5: 1.4*10**(-2), 3.0: 2.3*10**(-3), 3.5: 3.6*10**(-4)}
    D_A_Gpc =  WMAP9.angular_diameter_distance(z).to(u.Gpc).value
    return k_p[p] * (D_A_Gpc)**(-1) * nu_ssa_ghz**(-2*p-3) * theta_obs_mas**(-2*p-5) *\
           flux_nu_ssa_obs_jy**(p+2) * (delta / (1+z))**(-p-3)


def calculate_Ue_to_UB_kino2014(nu_ssa_ghz, theta_obs_mas, flux_nu_ssa_obs_jy,
                                delta, z, p, gamma_e_min):
    """
    Calculate B using (11) from Kino et al. 2014 (doi:10.1088/0004-637X/786/1/5).

    :param nu_ssa_ghz:
        Frequency of observation (it is SSA frequency for radio core).
    :param theta_obs_mas:
        Width of the VLBI core (see paper for discussion).
    :param flux_nu_ssa_obs_jy:
        Flux of the core.
    :param delta:
        Doppler factor.
    :param z:
        Redshift.
    :param p:
        Exponent of particles energy distribution. Must be 2.5, 3 or 3.5 here.
    :param gamma_e_min:

    :return:
        Value of the particles energy to B energy ratio.

    :note:
        U_B = B_{tot}^2/(8*pi), where B_{tot} = sqrt(3)*B => assumes isotropic
        tangled magnetic field here.
    """
    if p not in (2.5, 3.0, 3.5):
        raise Exception("p must be 2.5, 3.0 or 3.5")
    k_p = {2.5: 1.4*10**(-2), 3.0: 2.3*10**(-3), 3.5: 3.6*10**(-4)}
    b_p = {2.5: 3.3*10**(-5), 3.0: 1.9*10**(-5), 3.5: 1.2*10**(-5)}
    E_e_min_erg = (gamma_e_min*const.m_e*const.c**2).to(u.erg).value
    D_A_Gpc = WMAP9.angular_diameter_distance(z).to(u.Gpc).value
    return (8*np.pi/(3*b_p[p]**2)) * (k_p[p]*E_e_min_erg**(-p+2)/(p-2)) *\
           (D_A_Gpc)**(-1) * nu_ssa_ghz**(-2*p-13) * theta_obs_mas**(-2*p-13) *\
           flux_nu_ssa_obs_jy**(p+6) * (delta / (1+z))**(-p-5)


def calculate_B_marscher(nu_ssa_ghz, theta_obs_mas, flux_extrapol_nu_ssa_obs_jy,
                         delta, z, p):
    """
    Calculate B using (2) from Marscher 1983 (1983ApJ...264..296M).

   :param nu_ssa_ghz:
        Frequency of observation (it is SSA frequency for radio core).
    :param theta_obs_mas:
        Width of the VLBI core (see paper for discussion).
    :param flux_extrapol_nu_ssa_obs_jy:
        Core flux density extrapolated from optical thin spectrum [Jy].
    :param delta:
        Doppler factor.
    :param z:
        Redshift.
    :param p:
        Exponent of particles energy distribution. Must be 1.5, 2.0, 2.5 or 3.

    :return:
        Value of B [G].
    """
    if p not in (2.5, 3.0, 3.5):
        raise Exception("p must be 1.5, 2.0, 2.5 or 3")
    alpha = (p-1.0)/2
    b = {0.25: 1.8, 0.5: 3.2, 0.75: 3.6, 1.0: 3.8}
    return 10**(-5)*b[alpha] * theta_obs_mas**4 * nu_ssa_ghz**5 *\
           flux_extrapol_nu_ssa_obs_jy**(-2) * delta / (1+z)


def calculate_Ke_marscher(nu_ssa_ghz, theta_obs_mas, flux_extrapol_nu_ssa_obs_jy,
                         delta, z, p):
    """
    Calculate Ke using (3) from Marscher 1983 (1983ApJ...264..296M).

   :param nu_ssa_ghz:
        Frequency of observation (it is SSA frequency for radio core).
    :param theta_obs_mas:
        Width of the VLBI core (see paper for discussion).
    :param flux_extrapol_nu_ssa_obs_jy:
        Core flux density extrapolated from optical thin spectrum [Jy].
    :param delta:
        Doppler factor.
    :param z:
        Redshift.
    :param p:
        Exponent of particles energy distribution. Must be 1.5, 2.0, 2.5 or 3.

    :return:
        Value of B [G].
    """
    if p not in (2.5, 3.0, 3.5):
        raise Exception("p must be 1.5, 2.0, 2.5 or 3")
    alpha = (p-1.0)/2
    n = {0.25: 7.9, 0.5: 0.27, 0.75: 0.012, 1.0: 0.00059}
    D_L_Gpc = WMAP9.luminosity_distance(z).to(u.Gpc).value
    return n[alpha] * D_L_Gpc**(-1) ** theta_obs_mas**(-4*alpha-7.0) *\
           nu_ssa_ghz**(-4*alpha-5.0) * flux_extrapol_nu_ssa_obs_jy**(2*alpha+3.0) *\
           (1+z)**(2*alpha+6.0) * delta*(-2*alpha-4.0)


def calculate_B_zdzr2015(h_u, nu1_u, nu2_u, z, delta, dr_core_ang_u, hoangle_rad,
                         theta_los_rad, flux_u, p):
    """
    Calculate B using (8) from paper Zdziarski et al. 2015 (10.1093/mnras/stv986).
    Arguments with ``u`` postfix are values with units:

    >>> import astropy.units as u
    >>> flux_u = 1.0*u.Jy
    >>> nu1_u = 15.4*u.GHz

    :param h_u:
        Distance from BH.
    :param nu1_u:
        Lower frequency.
    :param nu2_u:
        Higher frequency.
    :param z:
        redshift.
    :param delta:
        Doppler factor.
    :param dr_core_ang_u:
        Core shift.
    :param hoangle_rad:
        Half-opening angle [rad].
    :param theta_los_rad:
        LOS angle [rad]
    :param flux_u:
        Flux.
    :param p:
        Exponent of particles energy distribution. Must be 2 or 3 here.

    :return:
        Value of B [G].

    """
    if p not in (2.0, 3.0):
        raise Exception("p must be 2.0 or 3.0")
    c3 = {2: 3.61, 3: 2.10, 4: 1.61}
    c1 = {2: 1.14, 3: 1}
    c2 = {2: 2./3, 3: 1}
    D_L = WMAP9.luminosity_distance(z)
    # B_cr = 2*np.pi*(const.m_e**2*const.c**3/(const.e.gauss*const.h)).cgs.value
    result = (D_L * delta * (const.h/const.m_e)**7 / (h_u*((1+z)*np.sin(theta_los_rad))**3*const.c**12) *
              (2*np.pi*(const.m_e**2*const.c**3/(const.e.gauss*const.h)) * dr_core_ang_u / (nu1_u**(-1) - nu2_u**(-1)))**5 * (const.alpha*c1[p]*c3[p]*np.tan(hoangle_rad)/(flux_u*24*np.pi**3*c2[p]))**2).cgs
    return result.value