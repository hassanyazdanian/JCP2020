#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    This program is part of codes written for doi: 10.1016/j.jcp.2020.109408
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    
    Direct evaluation of Biot-Savart Integral using 1/3 Simpson rule.
    
    This code was written to be compared with fftconv method in the benchmark 
    test 1 (Smooth phantom).

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
import time


def xSimpson(a, b, n):
    ''' Calculates abscissas x[] and weights w[] for 1/3 Simpson’s rule with 
        n integration points on interval [a,b]'''

    c13 = 1e0/3e0; c23 = 2e0/3e0; c43 = 4e0/3e0
    if (n % 2 == 0): n += 1 # increment n if even
    h = (b-a)/(n-1)
    w = np.ones(n)*c23*h
    i = np.mgrid[0:n]
    w[i%2==1] = c43*h
    w[0] = w[n-1] = h * c13

    return w


def qSimpson3D(Integrand, ax, bx, nx, ay, by, ny, az, bz, nz):

    '''Integrates function Func(x,y,z) in the cuboid [ax,bx] x [ay,by] x [az,bz]
        using Simpson’s rule with (nx x ny x nz) integration points'''

    wx = [0]*nx
    wy = [0]*ny
    wz = [0]*nz
    wx = xSimpson(ax,bx,nx) # generate integartion points
    wy = xSimpson(ay,by,ny)
    wz = xSimpson(az,bz,nz)
    s1 = 0e0
    sx = 0e0  
    sy = 0e0
    sy = np.dot(Integrand,wz)
    sx = np.dot(sy,wy)
    s1 = np.dot(sx,wx)
    
    return s1

#-------------- Define_Grid
def GRIDB(s,m):
    M=2**m		# number of grid points in each direction. we choose M as a power of 2 in order to speed up fft.
    h=s*2./(M-1)   	# step size
    x=np.linspace((-2**(m-1))*h,(2**(m-1))*h,M-1)	 # grid points in each direction
    X,Y,Z=np.meshgrid(x,x,x,indexing='ij',sparse=True)	 # evaluation cells. we attach a cell to each grid poin.
    return h,M,X,Y,Z

def GRIDJ(s,m):
	M=2**m		# number of grid points in each direction. we choose M as a power of 2 in order to speed up fft.
	h=s*2./(M-1)   	# step size
	x=np.linspace((-2**(m-1))*h,(2**(m-1))*h,M-1)+h/2	 # grid points in each direction
	X,Y,Z=np.meshgrid(x,x,x,indexing='ij',sparse=True)	 # evaluation cells. we attach a cell to each grid poin.
	return h,M,X,Y,Z


#-------------- Define J field
def J_field(r, X, Y, Z):
    ''' Define analytical smooth current density field '''
           
    ## Define a mask
    mask=((X**2+Y**2+Z**2)<r**2)
    ## Define J field
    R=np.sqrt(X**2+Y**2+Z**2)   
    Jx=((2*Y**3*(R - 1))/R**3 - (2*Y**3)/R**2 - (8*Y*(R - 1))/R + (2*Z*(R - 1))/R + (2*X**2*Z)/R**2 - (2*Y*Z**2)/R**2 + (2*X*Y*Z)/R**2 - (2*X**2*Z*(R - 1))/R**3 + (2*Y*Z**2*(R - 1))/R**3 - (2*X*Y*Z*(R - 1))/R**3)*mask
    Jy=((2*Z**3*(R - 1))/R**3 - (2*Z**3)/R**2 + (2*X*(R - 1))/R - (8*Z*(R - 1))/R + (2*X*Y**2)/R**2 - (2*X**2*Z)/R**2 + (2*X*Y*Z)/R**2 - (2*X*Y**2*(R - 1))/R**3 + (2*X**2*Z*(R - 1))/R**3 - (2*X*Y*Z*(R - 1))/R**3)*mask
    Jz=((2*X**3*(R - 1))/R**3 - (2*X**3)/R**2 - (8*X*(R - 1))/R + (2*Y*(R - 1))/R - (2*X*Y**2)/R**2 + (2*Y*Z**2)/R**2 + (2*X*Y*Z)/R**2 + (2*X*Y**2*(R - 1))/R**3 - (2*Y*Z**2*(R - 1))/R**3 - (2*X*Y*Z*(R - 1))/R**3)*mask
 
    return Jx, Jy, Jz

#-------------- 1/3 Simpson rule
    
def Simpson_integrand_aprox(xB, yB, zB, xJ, yJ, zJ, Jx, Jy, Jz):
    
    R3_2 = ((xB-xJ)**2+(yB-yJ)**2+(zB-zJ)**2)**(3/2)
    deltaX = xB-xJ
    deltaY = yB-yJ
    deltaZ = zB-zJ
  
    Bx_int = (deltaZ*Jy-deltaY*Jz)/R3_2
    By_int = (deltaX*Jz-deltaZ*Jx)/R3_2
    Bz_int = (deltaY*Jx-deltaX*Jy)/R3_2
    
    return Bx_int, By_int, Bz_int

#-------------- Initialization
s = 1	             # define compution domain size (-s to s)
r = 1	             # length of mask
mu0 = 1	             # we assumed that mu0=1
mR = [4, 5, 6]	     # M=2^m where M is umber of grid points in each direction.

Erms = np.zeros(len(mR))
TIME = np.zeros(len(mR))

for idx in range(0,len(mR)):
    m = mR[idx]
    h, M, xJ, yJ, zJ = GRIDJ(s,m)
    h, M, xB, yB, zB = GRIDB(s,m)
    Jx, Jy, Jz = J_field(r, xJ, yJ, zJ)
    ax = ay = az = xJ[0, 0, 0]
    bx = by = bz = xJ[M-2, 0, 0]
    nx = ny = nz = M-1
    
    time_start = time . time ()
    Ix = np.empty([np.size(xB),np.size(yB),np.size(zB)])
    Iy = np.empty([np.size(xB),np.size(yB),np.size(zB)])
    Iz = np.empty([np.size(xB),np.size(yB),np.size(zB)])
    for i in range(0,np.size(xB)):
        for j in range(0,np.size(yB)):
            for k in range(0,np.size(zB)):
                Bx_int, By_int, Bz_int = Simpson_integrand_aprox(xB[i,0,0], yB[0,j,0], zB[0,0,k], xJ, yJ, zJ,Jx, Jy, Jz)
                Ix[i,j,k] = (mu0/(4*np.pi))*qSimpson3D(Bx_int,ax,bx,nx,ay,by,ny,az,bz,nz)
                Iy[i,j,k] = (mu0/(4*np.pi))*qSimpson3D(By_int,ax,bx,nx,ay,by,ny,az,bz,nz)
                Iz[i,j,k] = (mu0/(4*np.pi))*qSimpson3D(Bz_int,ax,bx,nx,ay,by,ny,az,bz,nz)
                
    TIME [idx] = time . time ()-time_start  
      
    X = xB; Y = yB; Z = zB;   
    mask = ((X**2+Y**2+Z**2)<r**2) # sphere mask
    Ix = Ix*mask
    Iy = Iy*mask
    Iz = Iz*mask
    
    Ix = np.nan_to_num(Ix)
    Iy = np.nan_to_num(Iy)
    Iz = np.nan_to_num(Iz)

    
    M_2 = M/2-1
    Ix[int(M_2),:,:] = np.zeros(M-1)
    Iy[:,int(M_2),:] = np.zeros(M-1)
    Iz[:,:,int(M_2)]  =np.zeros(M-1)
    
    #-------------- Analytical_magnetic_flux_density
    R=np.sqrt(X**2+Y**2+Z**2)
    Bxa =((2*X*Y*(R - 1))/R - (2*Z**2*(R - 1))/R - (R - 1)**2)*mask
    Bya=((2*Y*Z*(R - 1))/R - (2*X**2*(R - 1))/R - (R - 1)**2)*mask
    Bza=((2*X*Z*(R - 1))/R - (2*Y**2*(R - 1))/R - (R - 1)**2)*mask
    
    Ix[np.where(abs(Bxa)<1e-4)] = 0
    Iy[np.where(abs(Bya)<1e-4)] = 0
    Iz[np.where(abs(Bza)<1e-4)] = 0
    
    
    Bxa = np.nan_to_num(Bxa)
    Bya = np.nan_to_num(Bya)
    Bza = np.nan_to_num(Bza)
    
    #-------------- L2-norm_error_calculation
    Erms[idx] = 100*np.sqrt(np.sum((Bxa-Ix)**2)+np.sum((Bya-Iy)**2)+np.sum((Bza-Iz)**2))/np.sqrt(np.sum(Bxa**2)+np.sum(Bya**2)+np.sum(Bza**2))

print('Erms=',Erms)
print('elapsed time=',TIME)
