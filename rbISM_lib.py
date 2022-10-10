import numpy as np
import matplotlib.pyplot as plt
import scipy as s
from scipy import signal
import pyISM as cPs
import ISM_processing.APR_lib as apr
import ISM_processing.FRC_lib as frc
import pandas as pd
import math
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable





#I input dataset
#Nz number of axial planes
#Nx number of images pixels
#N number of SPAD panels
#itera number of iteration 
#fingerprints must be passed --> index panels first argument  
#and index axial planes second argument
             
def stopindex_pr_based(I,Nz,Nx,N,iteras,fingerprints_matrix):
    tr=np.empty([Nx,Nx,iteras])
    kl=np.empty([Nx,Nx,iteras])
    pr=np.empty([Nx,Nx,iteras])
    one=np.ones([N**2])
    logyz=np.zeros(one.shape)
    stopindex=np.empty([Nx,Nx])
    for xs in range (Nx):
        print(xs)
        for ys in range (Nx):
            Ii = I[xs,ys,:]
            if np.sum(Ii)>0:
                a=np.ones(Nz)
                k=0
                dd=np.zeros([Nz,N**2])
    
                Ht1=np.matmul(np.transpose(fingerprints_matrix),one)
                while k<iteras:
                    Ha = np.dot(fingerprints_matrix,a)
                    z=np.divide(one,Ha)
                    yz = Ii * z
                    update=np.divide(np.matmul(np.transpose(fingerprints_matrix),yz),Ht1)
                    a = a * update
    
                    logyz[yz>0]=np.log(yz[yz>0])
                    kl[xs,ys,k]=np.matmul(Ii,logyz)
                   
                    
                    dy= np.matmul( np.diag( np.divide(a,Ht1) ) , np.matmul(np.transpose(fingerprints_matrix),np.diag(z) ) )
                    dx = np.diag(update) - np.matmul( dy, np.matmul( np.diag(yz), fingerprints_matrix) )
                    dd = np.matmul(dx,dd)-dy
                    D = np.multiply(np.diag(np.matmul(fingerprints_matrix,dd)),yz)
                    tr[xs,ys,k]=np.sum(D)
                    pr[xs,ys,k]=tr[xs,ys,k]+kl[xs,ys,k]
                    stopindex[xs,ys]=np.argmin(pr[xs,ys,1:])
                   
        return stopindex




def rb_ISM_pr_regu(I,Nz,Nx,N,iteras,fingerprints_matrix,stopindex,regularization='off'):
    
    if regularization=='on':
        tr=np.empty([Nx,Nx,iteras])
        kl=np.empty([Nx,Nx,iteras])
        pr=np.empty([Nx,Nx,iteras])
        one=np.ones([N**2])
        logyz=np.zeros(one.shape)
        ricos=np.empty([Nz,Nx,Nx])
        
        for xs in range (Nx):
            print(xs)
            for ys in range (Nx):
                Ii = I[xs,ys,:]
                if np.sum(Ii)>0:
                    a=np.ones(Nz)
                    k=0
                    dd=np.zeros([Nz,N**2])
        
                    Ht1=np.matmul(np.transpose(fingerprints_matrix),one)
                    while k<stopindex[xs,ys]:
                        Ha = np.dot(fingerprints_matrix,a)
                        z=np.divide(one,Ha)
                        yz = Ii * z
                        update=np.divide(np.matmul(np.transpose(fingerprints_matrix),yz),Ht1)
                        a = a * update
        
                        logyz[yz>0]=np.log(yz[yz>0])
                        kl[xs,ys,k]=np.matmul(Ii,logyz)
                       
                        
                        dy= np.matmul( np.diag( np.divide(a,Ht1) ) , np.matmul(np.transpose(fingerprints_matrix),np.diag(z) ) )
                        dx = np.diag(update) - np.matmul( dy, np.matmul( np.diag(yz), fingerprints_matrix) )
                        dd = np.matmul(dx,dd)-dy
                        D = np.multiply(np.diag(np.matmul(fingerprints_matrix,dd)),yz)
                        tr[xs,ys,k]=np.sum(D)
                        pr[xs,ys,k]=tr[xs,ys,k]+kl[xs,ys,k]
                        stopindex[xs,ys]=np.argmin(pr[xs,ys,1:])
                        k+=1
                        
                    ricos[:,xs,ys]=a
                   
        return stopindex, pr,ricos
    
    
    if regularization=='off':
      
        one=np.ones([N**2])
        ricos=np.empty([Nz,Nx,Nx])
        for xs in range (Nx):
            print(xs)
            for ys in range (Nx):
                Ii = I[xs,ys,:]
                if np.sum(Ii)>0:
                    a=np.ones(Nz)
                    k=0
        
                    Ht1=np.matmul(np.transpose(fingerprints_matrix),one)
                    while k<iteras:
                        Ha = np.dot(fingerprints_matrix,a)
                        z=np.divide(one,Ha)
                        yz = Ii * z
                        update=np.divide(np.matmul(np.transpose(fingerprints_matrix),yz),Ht1)
                        a = a * update
        
                        k+=1
                        
                    ricos[:,xs,ys]=a
                   
        return 0,0,ricos



def rb_ISM_pr_regu_tempo(I,Nz,Nx,N,iteras,fingerprints_matrix,stopindex,regularization='off'):
    
    if regularization=='on':
        tr=np.empty([Nx,Nx,iteras])
        kl=np.empty([Nx,Nx,iteras])
        pr=np.empty([Nx,Nx,iteras])
        one=np.ones([N**2])
        logyz=np.zeros(one.shape)
        ricos=np.empty([Nz,Nx,Nx])
        
        for xs in range (Nx):
            print(xs)
            for ys in range (Nx):
                Ii = I[xs,ys,:]
                if np.sum(Ii)>0:
                    a=np.ones(Nz)
                    k=0
                    dd=np.zeros([Nz,N**2])
        
                    # Ht1=np.matmul(np.transpose(fingerprints_matrix),one)
                    while k<stopindex[xs,ys]:
                        Ha = np.dot(fingerprints_matrix,a)
                        z=np.divide(one,Ha)
                        yz = Ii * z
                        update=np.matmul(np.transpose(fingerprints_matrix),yz)
                        a = a * update
        
                        logyz[yz>0]=np.log(yz[yz>0])
                        kl[xs,ys,k]=np.matmul(Ii,logyz)
                       
                       
                        k+=1
                        
                    ricos[:,xs,ys]=a
                   
        return stopindex, pr,ricos
    
    
    if regularization=='off':
      
        one=np.ones([N**2])
        ricos=np.empty([Nz,Nx,Nx])
        for xs in range (Nx):
            print(xs)
            for ys in range (Nx):
                Ii = I[xs,ys,:]
                if np.sum(Ii)>0:
                    a=np.ones(Nz)
                    k=0
        
                    # Ht1=np.matmul(np.transpose(fingerprints_matrix),one)
                    while k<iteras:
                        Ha = np.dot(fingerprints_matrix,a)
                        z=np.divide(one,Ha)
                        yz = Ii * z
                        update=np.matmul(np.transpose(fingerprints_matrix),yz)
                        a = a * update
        
                        k+=1
                        
                    ricos[:,xs,ys]=a
                   
        return 0,0,ricos