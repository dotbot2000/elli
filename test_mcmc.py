import glob
import os
import linecache
import time
from numpy import *
from multiprocessing import Process
from mcmc_ages import * 
import  matplotlib.pyplot as plt
import numpy as np

num_stars= 50 #'all' or int
max_threads = 16
out_dir="test_lnp_cut4"
out_prefix='test_lnp_cut4'#'47tuc_test5'
merged=out_dir+"/testc4.dat"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#data file
data=genfromtxt( '/home/jlin/Dropbox/galah/AMR/data/parallax/ramirez_with_parallax.csv'  ,names=True,dtype=None,delimiter='|')
#data=genfromtxt( 'test.tsv'  ,names=True,dtype=None,delimiter='|')

#locate some useful columns
FehS='FeH'
eFehS='e_FeH'
TeffS='Teff'
loggS='logg'
eTeffS='e_Teff'
eloggS='e_logg'
ID='HIP'
kmagS='kmag'
ekmagS='ekmag'
parallaxS='parallax'
eparallaxS='eparallax'

if num_stars=='all':
    data=data[:]
else:
    data=data[:int(num_stars)]


start=time.time()

for thread in range(max_threads):
#use do_run_emcee_full to get full output!
    p=Process(target=do_run_emcee_full, args=(data,out_dir,out_prefix,FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,thread,max_threads))
    p.start()

p.join()


end=time.time()

print(' Time to complete (s): ',  end-start)

# ha
if True:
    read_files = glob.glob('%s/%s_*' %(out_dir,out_prefix))
    line1=linecache.getline(read_files[0],1)

    with open(merged, "wb") as outfile:
        outfile.write(line1)
        for f in read_files:
            with open(f, "rb") as infile:
                next(infile)
                for line in infile:
                    outfile.write(line)


# plt.figure()
# a=np.genfromtxt('test_kmag/HD220507',delimiter=',',skip_header=42)
# titles=['age','mass','teff','logg','kmag','feh']
# for i in [0,1,2,3,4,5]:
#         plt.subplot(2,3,i+1)
#         plt.hist(a[:,i])
#         plt.xlabel(titles[i])
#         plt.locator_params(nbins=4)
#         plt.tight_layout()
# plt.show()

# c=pickle.load(open('test.p','rb')) #flatchain, flatchain, flat lnp
# #chain is age mass kmag feh
# names=['age','mass','kmag','feh']

# plt.figure()
# for i in [0,1,2,3]:
#     plt.subplot(2,2,i+1)
#     H, xedges, yedges = np.histogram2d(c[0][:,i],c[2][:,0],bins=50)
#     H = np.rot90(H)
#     H = np.flipud(H)
#     Hmasked = np.ma.masked_where(H==0,H)
#     plt.pcolormesh(xedges,yedges,Hmasked)
#     cbar = plt.colorbar()
#     cbar.ax.set_ylabel('steps')
#     plt.xlabel(names[i])        
#     plt.locator_params(nbins=5)
#     plt.ylabel('lnp')
# plt.tight_layout()
# plt.show()
