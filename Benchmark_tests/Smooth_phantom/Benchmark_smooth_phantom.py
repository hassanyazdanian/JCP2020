#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    This program is part of codes written for doi: 10.1016/j.jcp.2020.109408
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    
    Fast evaluation of Biot-Savart Integral using FFTCONVOLVE method!
    
    This is the benchmark test 1 (Smooth phantom) to o validate the technique
    developed to compute the Biot-Savart integral.
    Important notice: if our computation domain resides on -s to s,
    the current density field should be zero outside of (at least) -s/2 to s/2.

    If you have any question, please contact: hassanyazdanian@gmial.com.


    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

#-------------- Import_libraries
import numpy as np
from scipy.fftpack import fft, fft2, fftn
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")   # Be carefull about this!!! OFF all warnings


#-------------- Generate_computational_domain
def GRID(s, m):
    
    M = 2**m		# number of grid points in each direction. we choose M as a power of 2 to speed up fft computation.
    h = s*2./M   	# step size
    x = np.linspace((-2**(m-1)+1./2)*h,(2**(m-1)-1./2)*h,M)	 # grid points in each direction
    x, y, z = np.meshgrid(x, x, x, indexing='ij', sparse = True)	 # Evaluation cells. we attach a cell to each grid point.
    return h, M, x, y, z

#-------------- Define J field
def J_field(r, x, y, z):
    ''' Define analytical smooth current density field '''
           
    ## Define a mask, values outside of the mask set to be zero
    mask=((x**2+y**2+z**2)<r**2) # sphere mask
  
    Jx=(14*y*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2 - (8*y**3*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 - (4*y*z**3*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4 - (4*y**3*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4 - (8*y*z**3*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 - (8*x**2*y*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 - (4*x**2*y*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4
    Jy =(14*x*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2 - (8*x**3*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 - (4*x*z**3*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4 - (4*x**3*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4 - (8*x*z**3*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 - (8*x*y**2*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 - (4*x*y**2*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4
    Jz =(16*x*y**3*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 + (16*x**3*y*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 + (8*x*y**3*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4 + (8*x**3*y*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4 - (28*x*y*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2 + (16*x*y*z**2*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**3 + (8*x*y*z**2*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**4
    
    return mask, Jx, Jy, Jz

#------------------ Circular convolution by using FFT method 
def FFTCONV(h,Jx,Jy,Jz):
    ''' Calculates the magnetic field caused by a current density distribution Jx, Jy, and Jz '''
    
    
    #Fourier transform of the Biot-Savart kernel
    k = 2*np.pi*np.fft.fftfreq(Jx.shape[0],h)
    kx, ky, kz = np.meshgrid(k,k,k,indexing='ij',sparse=True)
    kx[0,0,0] = 1e-9
    K=1j/(kx**2+ky**2+kz**2)
    K[0,0,0]=0
    
    Jx_fft = np.fft.rfftn(Jx)
    Byh = np.multiply((K*kz)[:,:,0:len(K)//2+1],Jx_fft)
    Bzh = np.multiply(-(K*ky)[:,:,0:len(K)//2+1],Jx_fft)
    del Jx_fft

    Jy_fft = np.fft.rfftn(Jy)
    Bxh = np.multiply(-(K*kz)[:,:,0:len(K)//2+1],Jy_fft)
    Bzh += np.multiply((K*kx)[:,:,0:len(K)//2+1],Jy_fft)
    del Jy_fft

    Jz_fft = np.fft.rfftn(Jz)
    Bxh += np.multiply((K*ky)[:,:,0:len(K)//2+1],Jz_fft)
    Byh += np.multiply(-(K*kx)[:,:,0:len(K)//2+1],Jz_fft)
    del Jz_fft

    Bxh = np.fft.irfftn(Bxh)
    Byh = np.fft.irfftn(Byh)
    Bzh = np.fft.irfftn(Bzh)
      
    return Bxh, Byh, Bzh

#-------------- Initialization
s = 2	                    # define the size ofcomputional domain  (-s to s)               
r = 1	                    # radii of mask
mu0 = 1	                    # we assumed that mu0=1
mR = [5, 6, 7, 8, 9]        # M=2^mR[i] where M is umber of grid points in each direction.

Erms = np.zeros(len(mR))    # Error vector
TIME = np.zeros(len(mR))    # Time vector


for idx in range(0,len(mR)):
    m = mR[idx]
    h, M, x, y, z = GRID(s,m)
    M_2 = M/2	                # half of grid points
    M_4 = M/4	                # quarter of grid points
    
    mask, Jx, Jy, Jz = J_field(r,x,y,z)
    
    time_start = time . time ()
    Bxh, Byh ,Bzh = FFTCONV(h,Jx,Jy,Jz)
    TIME [idx] = time . time ()-time_start
    
    #-------------- Analytical_magnetic_flux_density
    Bx = (4*x*y**2*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2 - 3*x*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)) + (2*x*z**2*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2
    By = 3*y*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)) - (4*x**2*y*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2 - (2*y*z**2*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2
    Bz = (2*y**2*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2 - (2*x**2*z*np.exp(-1/abs((r**2-(x**2+y**2+z**2))*mask)))/(- r**2 + x**2 + y**2 + z**2)**2
    
    #-------------- L2-norm_error_calculation
    Erms[idx] = 100*np.sqrt(np.sum((Bx-Bxh)**2)+np.sum((By-Byh)**2)+np.sum((Bz-Bzh)**2))/np.sqrt(np.sum(Bx**2)+np.sum(By**2)+np.sum(Bz**2))
   
#-------------- Plot_results
print (Erms)
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
fig, ax1 = plt.subplots()
ax1.set_xlabel('$M$')
ax1.set_ylabel(r'$ \mathbf E $ (\%)',fontsize=16)
ax1.tick_params('y')
ax1.set_xscale('log', basex=2)
ax1.set_yscale('log', basey=10)
plt.xticks(2**np.array(mR),['32','64','128','256','512'])
plt.grid(True, which="major",axis="both",ls="-")
plt.legend((r'Smooth $ \mathbf J$', r'Piecewise smooth $ \mathbf J$'),loc='lower center',fontsize=12)

ax2 = ax1.twinx()
ax2.loglog(np.power(2,mR),TIME,'r-D')
ax2.set_ylabel(r'Time ($s$)', color='r',fontsize=16)
ax2.tick_params('y', colors='r')
ax2.set_xscale('log', basex=2)
ax2.set_yscale('log', basey=10)
plt.xticks(2**np.array(mR),['32','64','128','256','512'])

# plt.savefig('SmoothPSmooshvsANA.jpg', dpi = 200, bbox_inches='tight')

plt.show()
