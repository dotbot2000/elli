#parameters:
#include_teff=1
#include_logg=1
#include_kmag=1
#include_feh=1
#nwalkers=200
#nburn=200
#nrun=500

import glob
import os
import linecache
from numpy import *
from multiprocessing import Process
from mcmc_ages_2 import * 
import  matplotlib.pyplot as plt
import numpy as np

num_stars= 1 #'number of stars, 'all' or interger
max_threads = 1 #number of threads 
out_dir="test" # where to put the output 
out_prefix='test'#names of the outputs 

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#data file
data=genfromtxt( 'test_stars.csv'  ,names=True,dtype=None,delimiter=',')
#locate some useful columns
FehS='FeH'
eFehS='eFeH'
TeffS='Teff'
loggS='logg'
eTeffS='eTeff'
eloggS='elogg'
ID='ID'
kmagS='kmag'
ekmagS='ekmag'
parallaxS='parallax_T'
eparallaxS='eparallax_T'

if num_stars=='all':
    data=data[:]
else:
    data=data[:int(num_stars)]

for thread in range(max_threads):
#use do_run_emcee_full to get full output!
    p=Process(target=do_run_emcee, args=(data,out_dir,out_prefix,FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,thread,max_threads))
    p.start()

p.join()

