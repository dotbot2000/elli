from __future__ import print_function
from numpy import *
from numpy.linalg import inv, det
import emcee
from numpy.random import rand, seed
from scipy.interpolate import interp1d
from Isochrones import DSED_Isochrones
from asteroseismic import *

#define constants
log2pi=log(2*pi)

class star:
    def __init__(self):
        self.Teff=None; self.sigma_Teff=None
        self.logg=None; self.sigma_logg=None
        self.FeH=None; self.sigma_FeH = None
        self.kmag=None; self.sigma_kmag = None #apparent
        self.Kmag=None; self.sigma_Kmag = None #absolute
        self.parallax=None; self.sigma_parallax=None
        self.delta_nu=None; self.sigma_dnu=None
        self.nu_max=None; self.sigma_numax=None
        self.ID=None

    def pack(self):
        self.data=[]
        self.mask=[]

        if self.Teff is not None:
            self.data.append(self.Teff)
            self.mask.append(True)
        else:
            self.mask.append(False)

        if self.logg is not None:
            self.data.append(self.logg)
            self.mask.append(True)
        else:
            self.mask.append(False)

        if self.FeH is not None:
            self.data.append(self.FeH)
            self.mask.append(True)
        else:
            self.mask.append(False)

        if self.Kmag is not None:
            self.mask.append(True)
            self.data.append(self.Kmag)
        else:
            self.mask.append(False)

        if self.delta_nu is not None:
            self.data.append(self.delta_nu)
            self.mask.append(True)
        else:
            self.mask.append(False)

        if self.nu_max is not None:
            self.data.append(self.nu_max)
            self.mask.append(True)
        else:
            self.mask.append(False)

        self.data=array(self.data)
        self.mask=array(self.mask)
        self.dim = len(self.data)

    def set_absolute_Kmag(self):
        self.distance, self.sigma_distance = parallax_distance(self.parallax, self.sigma_parallax)
        self.Kmag, self.sigma_Kmag = Kmag_from_distance(self.kmag, self.sigma_kmag, self.distance, self.sigma_distance)

    def set_covariance_matrix(self,cov):
        self.cov = cov
        #then invert it
        self.icov=inv(self.cov)
        #and take the determinant, and the log of that...
        self.det_cov = det(self.cov)
        self.log_det_cov = log(self.det_cov)



def parallax_distance(parallax,err_parallax): #both in mas
    d=(parallax*u.mas).to(u.parsec, equivalencies=u.parallax())
    d=d.value 
    sigma_d=abs(d*err_parallax/float(parallax))
    print('distance: '+str(d)+'+/-'+str(sigma_d)+' pc')
    return d,sigma_d #both in pc

def Kmag_from_distance(kmag,err_kmag,d,err_d):
    Kmag= kmag-5*(log10(d)-1 )
    sigma_Kmag=sqrt( pow(err_kmag,2)+ 25*pow( err_d/ (float(d)*log(10) ),2)  )
    print('abs K mag: '+ str(Kmag)+'+/-'+str(sigma_Kmag))
    return Kmag, sigma_Kmag #absolute kmag

def interp(x,y):
    #return interp1d(x=x,y=y,kind='nearest')
    return interp1d(x=x,y=y,kind='linear')

def get_star(input_params,y):
    age=input_params[0]
    mass=input_params[1]
    FeH = input_params[2]
    ok=False
    result=empty(6)
    #y is a sorted list of isochrones ordered by increasing [Fe/H]
    if y[0].FeH <= FeH <= y[-1].FeH: 
        met=[]; Teff=[]; logg=[]; Kmag=[]; dnu=[]; numax=[]
        for i in range(len(y)):
            if y[i].FeH <= FeH < y[i+1].FeH: 
                iy=i
                break

        for x in y[iy:iy+2]:
            ok,params=get_one_star(age,mass,x)
            if ok:
                met.append(x.FeH)
                Teff.append(params[0])
                logg.append(params[1])
                Kmag.append(params[2])
                dnu.append(params[3])
                numax.append(params[4])
        #now we have two lists, 
        #one filled with mets and the other with results

        if len(met)>1 and met[0] <= FeH <= met[-1]:
            result[0] = interp(met,Teff)(FeH)
            result[1] = interp(met,logg)(FeH)
            result[2] = FeH
            result[3] = interp(met,Kmag)(FeH)
            result[4] = interp(met,dnu)(FeH)
            result[5] = interp(met,numax)(FeH)
    return ok, result

def get_one_star(age,mass,x):
    ok=False
    params=empty(3)
    if x.ages[0] <= age <= x.ages[-1]:
        for i in range(len(x.ages)-1):
            if x.ages[i] <= age <= x.ages[i+1]: 
                i0=i
                i1=i+1
                break
        #linear age interpolation
        m0=x.data[i0]['M_Mo']
        m1=x.data[i1]['M_Mo']
        if m0[0] <= mass <= m0[-1] and m1[0] <= mass <= m1[-1]:
            T0=interp(m0,x.data[i0]['LogTeff'])(mass)
            T1=interp(m1,x.data[i1]['LogTeff'])(mass)

            g0=interp(m0,x.data[i0]['LogG'])(mass)
            g1=interp(m1,x.data[i1]['LogG'])(mass)

            K0=interp(m0,x.data[i0]['Ks      '])(mass)
            K1=interp(m1,x.data[i1]['Ks      '])(mass)

            L0=interp(m0,x.data[i0]['LogL_Lo'])(mass)
            L1=interp(m1,x.data[i1]['LogL_Lo'])(mass)

            alfa=(age-x.ages[i0])/(x.ages[i1]-x.ages[i0])
            beta=1.0-alfa
            Teff=alfa*pow(10,T1) + beta*pow(10,T0)
            logg=alfa*g1 + beta*g0
            Kmag=alfa*K1 + beta*K0
            logL=alfa*L1 + beta*L0
            luminosity = pow(10,logL)
            dnu = delta_nu_func(mass,Teff,luminosity)
            numax= nu_max_func(mass,Teff,luminosity)
            params=array([Teff,logg,Kmag,dnu,numax])
            ok=True
    return ok,params

def lnPrior(m):
    alpha=-2.35 #Salpeter IMF slope                                                                        
    m0=0.1
    norm=-(alpha+1.)/pow(m0,alpha+1.)
    return log(norm*pow(m,alpha))

def lnProb(params,star):
    #params = [age, mass, feh]
    ok,model=get_star(params,y)
    N = star.dim
    if ok:
        #shrink the model array to only those values present in the data
        mod = array([model[i] for i in range(len(star.mask)) if star.mask[i]])
        diff = star.data - mod
        #now calculate ln(probability)
        return -0.5 * ( dot(diff, dot(star.icov,diff) ) + star.log_det_cov + N*log2pi ), model
    else:
        return -inf, model


def run_emcee(s):
    print(s.ID)
    #each thread has an independent random number sequence
    seed() 

    #for a, typically 2 is a good number; must be > 1
    nwalkers=200
    nburn=100
    nrun=500
    ndim=3
    my_a=3 #typically 2

    age_guess=7.0
    mass_guess=0.8
    feh_guess=s.FeH+1e-5

    guess=array([age_guess,mass_guess,feh_guess])
    print(' guess = ', guess)
    p0=[guess*(1-0.2*(0.5-rand(ndim))) for i in range(nwalkers)]

    #create an instance 
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnProb,args=[s],a=my_a)

    #burn-in: save end state and reset
    pos,prob,state,blob=sampler.run_mcmc(p0,nburn)
    sampler.reset()

    #main run
    sampler.run_mcmc(pos,nrun)
    lnp_flat=sampler.flatlnprobability

    Teff_dist=[]
    logg_dist=[]
    kmag_dist=[]
    for i in range(nrun):
        for j in range(nwalkers):
            Teff_dist.append(sampler.blobs[i][j][0])
            logg_dist.append(sampler.blobs[i][j][1])
            kmag_dist.append(sampler.blobs[i][j][2])

    logg_dist=array(logg_dist)
    Teff_dist=array(Teff_dist)
    kmag_dist=array(kmag_dist)

    #print basic results
    print()
    print("Result: ", s.ID)
    print("Mean acceptance fraction: {0:10.4g}".format(mean(sampler.acceptance_fraction)))

    result=[]

    age_dist=sampler.flatchain[:,0]
    print("len(age_dist)=", len(age_dist))
    print(min(age_dist))
    print(mean(age_dist))
    print(std(age_dist))
    print(max(age_dist))

    mass_dist=sampler.flatchain[:,1]
    print("len(mass_dist)=", len(mass_dist))
    print(min(mass_dist))
    print(mean(mass_dist))
    print(std(mass_dist))
    print(max(mass_dist))
    print("len(mass_dist)=", len(mass_dist))

    feh_dist = sampler.flatchain[:,2]
    print("len(feh_dist)=", len(feh_dist))
    print(min(feh_dist))
    print(mean(feh_dist))
    print(std(feh_dist))
    print(max(feh_dist))
    print("len(feh_dist)=", len(feh_dist))

    print("len(kmag_dist)=", len(kmag_dist))
    print(min(kmag_dist))
    print(max(kmag_dist))
    print("len(kmag_dist)=", len(kmag_dist))

    if False:
        good=where( (lnp_flat>-15)& (kmag_dist!=99.99) & (logg_dist!=99.99) & (Teff_dist!=0) )

        print(good)

        if len(good)==0:
            logg_dist=logg_dist[0:0]
            Teff_dist=Teff_dist[0:0]
            kmag_dist=kmag_dist[0:0]
            age_dist=age_dist[0:0]
            mass_dist=mass_dist[0:0]
            feh_dist=feh_dist[0:0]
            lnp_flat=lnp_flat[0:0]
        else:
            logg_dist=logg_dist[good]
            Teff_dist=Teff_dist[good]
            kmag_dist=kmag_dist[good]
            age_dist=age_dist[good]
            mass_dist=mass_dist[good]
            feh_dist=feh_dist[good]
            lnp_flat=lnp_flat[good]

        result.append(age_guess)
        if len(age_dist)>0: 
            pct=prctile(age_dist,p=[2.5,50,97.5])
            result.append(mean(age_dist))
            result.append(std(age_dist))
            result.append(pct[0]) #2.5%
            result.append(pct[1]) #50%
            result.append(pct[2]) #97.5%
        else:
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)

        result.append(mass_guess)
        if len(mass_dist)>0:
            pct=prctile(mass_dist,p=[2.5,50,97.5])
            result.append(mean(mass_dist))
            result.append(std(mass_dist))
            result.append(pct[0]) #2.5th percentile
            result.append(pct[1]) #50th percentile==median
            result.append(pct[2]) #97.5%
        else:
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)

        if len(Teff_dist) > 0:
            pct=prctile(Teff_dist,p=[2.5,50,97.5])
            result.append(mean(Teff_dist))
            result.append(std(Teff_dist))
            result.append(pct[0])
            result.append(pct[1])
            result.append(pct[2])
        else:
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)

        if len(logg_dist) > 0:
            pct=prctile(logg_dist,p=[2.5,50,97.5])
            result.append(mean(logg_dist))
            result.append(std(logg_dist))
            result.append(pct[0])
            result.append(pct[1])
            result.append(pct[2])
        else:
            result.append(99.0)
            result.append(99.0)
            result.append(99.0)
            result.append(99.0)
            result.append(99.0)

        if len(kmag_dist)>0: 
            pct=prctile(kmag_dist,p=[2.5,50,97.5])
            print('mean kmag')
            print(mean(kmag_dist))
            result.append(mean(kmag_dist))
            result.append(std(kmag_dist))
            result.append(pct[0]) #2.5%
            result.append(pct[1]) #50%
            result.append(pct[2]) #97.5%
        else:
            result.append(99.0)
            result.append(99.0)
            result.append(99.0)
            result.append(99.0)
            result.append(99.0)        

        if len(feh_dist)>0:
            mean_feh = mean(feh_dist)
            std_feh = std(feh_dist)
            pct=prctile(feh_dist,p=[2.5,50,97.5])
            result.append(mean_feh)
            result.append(std_feh)
            result.append(pct[0]) #5th percentile
            result.append(pct[1]) #50th percentile==median
            result.append(pct[2]) #95%

        else:
            mean_feh = 0.0
            std_feh = 0.0
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)
            result.append(0.0)

    return age_dist, mass_dist, feh_dist, lnp_flat




iso_dir='/home/dotter/science/mcmc/iso'
iso_list=['fehm25afep6.UBVRIJHKsKp',
          'fehm20afep4.UBVRIJHKsKp',
          'fehm15afep4.UBVRIJHKsKp',
          'fehm10afep4.UBVRIJHKsKp',
          'fehm05afep2.UBVRIJHKsKp',          
          'fehp00afep0.UBVRIJHKsKp',
          'fehp02afep0.UBVRIJHKsKp',
          'fehp03afep0.UBVRIJHKsKp',
          'fehp05afep0.UBVRIJHKsKp']
y=[]
for iso in iso_list:
    y.append(DSED_Isochrones(iso_dir+'/'+iso))
