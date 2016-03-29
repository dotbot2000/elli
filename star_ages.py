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
        self.delta_nu=None; self.sigma_delta_nu=None
        self.nu_max=None; self.sigma_nu_max=None
        self.ID=None; self.initial_guess=None
        self.sampler=None

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

    def run_emcee(self,nwalkers=200,nburn=100,nrun=400,a=3.):
        #each thread has an independent random number sequence
        seed() 
        ndim=3

        #initial guess for this star
        age_guess=7.0
        mass_guess=0.8
        feh_guess=self.FeH

        guess=array([age_guess,mass_guess,feh_guess])
        self.initial_guess = guess

        #create a cloud around the guess for the walkers
        p0=[guess*(1-0.2*(0.5-rand(ndim))) for i in range(nwalkers)]

        #create an instance 
        self.sampler=emcee.EnsembleSampler(nwalkers,ndim,lnProb,args=[self],a=a)

        #burn-in: save end state and reset
        pos,prob,state,blob=self.sampler.run_mcmc(p0,nburn)
        self.sampler.reset()

        #main run
        self.sampler.run_mcmc(pos,nrun)

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



if __name__ == '__main__':

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

    if False:
        print("Example: Sun")
        x=star()
        x.Teff=5777; x.sigma_Teff=3.
        x.logg=4.4;  x.sigma_logg=0.03
        x.FeH=0.0;   x.sigma_FeH=0.01
        x.Kmag=3.302; x.sigma_Kmag=0.005
        x.delta_nu = 135.1; x.sigma_delta_nu=0.1
        x.nu_max = 3090.0; x.sigma_nu_max=30

        covariance=zeros((6,6))
        covariance[0,0]=pow(x.sigma_Teff,2)
        covariance[1,1]=pow(x.sigma_logg,2)
        covariance[2,2]=pow(x.sigma_FeH,2)
        covariance[3,3]=pow(x.sigma_Kmag,2)
        covariance[4,4]=pow(x.sigma_delta_nu,2)
        covariance[5,5]=pow(x.sigma_nu_max,2)
        #add off diagonal terms as needed...

        x.set_covariance_matrix(covariance)
        x.pack()
        x.run_emcee(nwalkers=100,nrun=250)


        C=cov(x.sampler.flatchain.T)

        print(matrix(C))
        print('mean age = ', mean(x.sampler.flatchain[:,0]))
        print('std  age = ', std(x.sampler.flatchain[:,0]))
        print('mean mass= ', mean(x.sampler.flatchain[:,1]))
        print('std  mass= ', std(x.sampler.flatchain[:,1]))
        print('mean Fe/H= ', mean(x.sampler.flatchain[:,2]))
        print('std  Fe/H= ', std(x.sampler.flatchain[:,2]))


    print("\nExample: GALAH+Cannon+covariance")
    w=star()
    w.Teff=6224.146
    w.logg=3.78807
    w.FeH=-0.7095807
    cov=array([[  1.37034154e+01,   2.54319931e-03,   7.33258078e-03],
               [  2.54319931e-03,   6.03449909e-05,  -5.09215250e-07],
               [  7.33258078e-03,  -5.09215250e-07,   1.39315419e-05]])
    w.pack()
    w.set_covariance_matrix(cov)
    w.run_emcee(nwalkers=100,nrun=250)
    

    print('mean age = ', mean(w.sampler.flatchain[:,0]))
    print('std  age = ', std(w.sampler.flatchain[:,0]))
    print('mean mass= ', mean(w.sampler.flatchain[:,1]))
    print('std  mass= ', std(w.sampler.flatchain[:,1]))
    print('mean Fe/H= ', mean(w.sampler.flatchain[:,2]))
    print('std  Fe/H= ', std(w.sampler.flatchain[:,2]))



    print("\nExample: GALAH+Cannon+ no covariance")
    z=star()
    z.Teff=6224.146
    z.logg=3.78807
    z.FeH=-0.7095807
    cov=array([[  1.37034154e+01,              0.0,              0.0],
               [             0.0,   6.03449909e-05,              0.0],
               [             0.0,              0.0,   1.39315419e-05]])
    z.pack()
    z.set_covariance_matrix(cov)
    z.run_emcee(nwalkers=100,nrun=250)

    print('mean age = ', mean(z.sampler.flatchain[:,0]))
    print('std  age = ', std(z.sampler.flatchain[:,0]))
    print('mean mass= ', mean(z.sampler.flatchain[:,1]))
    print('std  mass= ', std(z.sampler.flatchain[:,1]))
    print('mean Fe/H= ', mean(z.sampler.flatchain[:,2]))
    print('std  Fe/H= ', std(z.sampler.flatchain[:,2]))


    from multiprocessing import Process

    #read data from files or whatever into a list of stars() called star_data
    #but don't be greedy about memory!

    max_threads=4
    for thread in range(max_threads):
        p=Process(target=do_run_emcee, args=(star_data,thread,max_threads))
        p.start()
    p.join()
