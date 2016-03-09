#reference values; Teff from IAU 2015 Resolution B3
delta_nu_sol = 135.1
nu_max_sol = 3090.0
Teff_sol = 5772.0 

def delta_nu_func(M_Mo,Teff,L_Lo):
    return delta_nu_sol*sqrt(M_Mo)*pow(Teff/Teff_sol,3)*pow(L_Lo,-0.75)

def nu_max_func(M_Mo,Teff,L_Lo):
    return nu_max_sol*M_Mo*pow(Teff/Teff_sol,3.5)/L_Lo
