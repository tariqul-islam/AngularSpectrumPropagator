#v2.1
#01/25/2022

import numpy as np
import torch #V 1.7.0 REQUIRED
from torch.fft import fft2 as fft2
from torch.fft import ifft2 as ifft2
from torch.fft import fftshift as fftshift
from torch.fft import ifftshift as ifftshift

from scipy.special import eval_genlaguerre

class angular_spectrum:

    def __init__(self, dx=0, dy=0, refractive_index=0, wavelength=0, sizex=0, sizey=0, device=0):
        self.dx = dx
        self.dy = dy
        self.refractive_index = refractive_index
        self.wavelength = wavelength
        self.sizex = sizex
        self.sizey = sizey
        self.device = device
        self.d2r = np.pi/180
        
        self.k0 = 2 * np.pi / wavelength
        
        self.Xnp, self.Ynp = self.meshgridNumpy(dx,dy,sizex,sizey)
        
        if dx!=0:
            print('Size of Field: ', sizex*dx*10**3, 'x', sizey*dy*10**3, 'mm^2')
        
    def I(self, U):
        return U.abs().cpu().detach().numpy()**2
        
    def abs(self, U):
        return U.abs().cpu().detach().numpy()
    
    def angle(self, U):
        return U.angle().cpu().detach().numpy()
        
    def multiply(self,U0,H):
        return U0*H
        
    def meshgridTorch(self,dx,dy,sizex,sizey,device):
        sizex_t = torch.as_tensor(sizex)
        sizey_t = torch.as_tensor(sizey)

        x = torch.arange(-torch.floor(sizex_t/2),sizex_t/2)
        y = torch.arange(-torch.floor(sizey_t/2),sizey_t/2)

        Dx = x * dx
        Dy = y * dy

        X,Y = torch.meshgrid(Dx,Dy)
        
        #Have to be changed when torch updates their repo 
        #the following two lines makes the current code numpy compatible
        #Torch 1.9
        X = torch.transpose(X, 0, 1).to(device)
        Y = torch.transpose(Y, 0, 1).to(device)
    
        return X,Y
        
    def meshgridNumpy(self, dx,dy,sizex,sizey,device=None):
        x = np.arange(-np.floor(sizex/2),sizex/2)
        y = np.arange(-np.floor(sizey/2),sizey/2)
        
        Dx = x * dx
        Dy = y * dy

        X,Y = np.meshgrid(Dx,Dy)
        
        return X,Y
    
    def planeWaveH(self, theta, phi, dx, dy, sizex, sizey, wavelength, device):
        '''
        sizex_t = torch.as_tensor(sizex).to(device)
        sizey_t = torch.as_tensor(sizey).to(device)

        x = torch.arange(-torch.floor(sizex_t/2),sizex_t/2)
        y = torch.arange(-torch.floor(sizey_t/2),sizey_t/2)

        Dx = x * dx
        Dy = y * dy

        X,Y = torch.meshgrid(Dx,Dy)
        
        #Have to be changed when torc updates their repo
        #the following two lines makes the current code numpy compatible
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        
        #print(X.size())
        '''
        
        X,Y = self.meshgridTorch(dx, dy, sizex, sizey,device)

        d2r = np.pi/180

        k0 = 2*np.pi/wavelength

        kz = torch.cos(theta*d2r)
        kx = torch.sin(theta*d2r) * torch.cos(phi*d2r)
        ky = torch.sin(theta*d2r) * torch.sin(phi*d2r)
        
        Z = kx*X
        Z2 = ky*Y
        
        H = torch.exp(-1j*k0*(kx*X+ky*Y))
        
        return H
        
    def planeWaveZ(self, theta, phi):
    
        theta_t = torch.as_tensor(theta, dtype=torch.float).to(self.device)
        phi_t = torch.as_tensor(phi, dtype=torch.float).to(self.device)
        
        H = self.planeWaveH(theta = theta_t, phi=phi_t, 
                        dx=self.dx, dy=self.dy, sizex=self.sizex, sizey=self.sizey, 
                        wavelength=self.wavelength, device=self.device)
        
        return H
        
    def sphericalWaveH(self, zero_loc,  dx, dy, sizex, sizey, wavelength, device):
        X,Y = self.meshgridTorch(dx, dy, sizex, sizey,device)
        
        x,y,z = zero_loc
        
        k0 = 2*np.pi/wavelength
        
        R = torch.sqrt((X-x)**2+(Y-y)**2+z**2)
        
        U = 1/R * torch.exp(-1j*k0*R)
        
        return U
        
    def sphericalWaveZ(self, zero_loc):
        U = self.sphericalWaveH(zero_loc=zero_loc,  
                                dx=self.dx, dy=self.dy, sizex=self.sizex, sizey=self.sizey, 
                                wavelength=self.wavelength, device=self.device)
        return U
        
        
    def rectApertureH(self, rect_s, dx,dy,sizex,sizey, device):
        '''
        x = np.arange(-np.floor(sizex/2),sizex/2)
        y = np.arange(-np.floor(sizey/2),sizey/2)
        
        Dx = x * dx
        Dy = y * dy

        X,Y = np.meshgrid(Dx,Dy)
        '''
        X,Y = self.meshgridNumpy(dx,dy,sizex,sizey,device)
        
        H = (np.abs(X) <= rect_s/2) * (np.abs(Y) <=rect_s/2)
        
        H = torch.as_tensor(H,dtype=torch.cfloat).to(device)
        
        return H
        
        
    def circularApertureH(self, radius, dx, dy, sizex, sizey, device):
        X,Y = self.meshgridNumpy(dx,dy,sizex,sizey,device)
        H = 1.0* ( X**2+Y**2 <= radius**2)
        H = torch.as_tensor(H, dtype=torch.cfloat).to(device)
        return H
        
    def halfPlaneApertureH(self, dx, dy, sizex, sizey, device, a_type='xp'):
    
        X,Y = self.meshgridNumpy(dx,dy,sizex,sizey,device)
        
        if a_type == 'xp':
            H = 1.0 * (X>=0)
        elif a_type == 'xn':
            H = 1.0 * (X<=0)
        elif a_type == 'yp':
            H = 1.0 * (Y>=0)
        elif a_type == 'yn':
            H = 1.0 * (Y<=0)
        else:
            print('Invalid Aperture Type')
        
        H = torch.as_tensor(H,dtype=torch.cfloat).to(device)
        
        return H
    
    def halfPlaneHilbertApertureH(self, dx, dy, sizex, sizey, device, a_type='xp'):
    
        X,Y = self.meshgridNumpy(dx,dy,sizex,sizey,device)
        
        if a_type == 'xp':
            H = 1.0 * (X>=0)
            H[X==0] = 0.5
        elif a_type == 'xn':
            H = 1.0 * (X<=0)
            H[X==0] = 0.5
        elif a_type == 'yp':
            H = 1.0 * (Y>=0)
            H[Y==0] = 0.5
        elif a_type == 'yn':
            H = 1.0 * (Y<=0)
            H[Y==0]=0.5
        else:
            print('Invalid Aperture Type')
        
        H = torch.as_tensor(H,dtype=torch.cfloat).to(device)
        
        return H
        
#    def 
    def phaseTransitionH(self, nR, dz, wavelength, device):
        H = torch.exp(+1j*2*np.pi/wavelength*nR*dz)
        
        #H = torch.as_tensor(H,dtype=torch.cfloat).to(device)
        
        return H
        
    def phaseTransitionZ(self, U0, nR, dz):
        
        H = self.phaseTransitionH(nR, dz, self.wavelength, self.device)
        
        U = U0 * H
        
        return U, H
   

    def computeH(self, dx, dy, refractive_index, wavelength, sizex, sizey, dz, device):
        
        k0 = 2*np.pi / wavelength
        
        fx = 1/dx
        fy = 1/dy 
        Fx = (np.arange(0,sizex)-np.floor(sizex/2))*fx/sizex
        Fy = (np.arange(0,sizey)-np.floor(sizex/2))*fx/sizex
        
        #Fx = Fx[:sizex]
        #Fy = Fy[:sizey]
        
        #print(Fx.shape, Fy.shape)
        #print(Fx)
        
        FX,FY = np.meshgrid(Fx,Fy)
        kxy = 4 * np.pi * (FX**2+FY**2)
        Factor = (k0*refractive_index)**2 - 4 * np.pi**2 * (FX**2+FY**2)
        Factor = Factor + 1j * 0
        #window = 1.0 * (Factor>=0)
        Factor = Factor #* window
        
        Hf = np.exp(1j * dz * np.sqrt(Factor)) #* window #Angular SPectrum Filter
        
        Hf = torch.as_tensor(Hf,dtype=torch.cfloat).to(device)
        
        #print('Hf size: ', Hf.size())
        
        return Hf
        
    def computeInverseH(self, dx, dy, refractive_index, wavelength, sizex, sizey, dz, device):
        #computes filter for angular spectrum propagation
        #Formulation from Ersoy - Fourier Optics
        
        #x = np.arange(-sizex/2,sizex/2)
        #y = np.arange(-sizey/2,sizey/2)
        #Dx = x * dx
        #Dy = y * dy
        
        k0 = 2*np.pi / wavelength
        
        fx = 1/dx
        fy = 1/dy 
        Fx = (np.arange(0,sizex)-np.floor(sizex/2))*fx/sizex
        Fy = (np.arange(0,sizey)-np.floor(sizex/2))*fx/sizex
        
        #print(Fx.shape, Fy.shape)
        #print(Fx)
        
        FX,FY = np.meshgrid(Fx,Fy)
        kxy = 4 * np.pi * (FX**2+FY**2)
        Factor = (k0*refractive_index)**2 - 4 * np.pi**2 * (FX**2+FY**2)
        Factor = Factor + 1j * 0
        #window = 1.0 * (Factor>=0)
        Factor = Factor #* window
        
        Hf = np.exp(-1j * dz * np.sqrt(Factor)) #* window #Angular SPectrum Filter
        
        Hf = torch.as_tensor(Hf,dtype=torch.cfloat).to(device)
        
        return Hf
        

    def propagateH(self, U0, H):
        #Computes angular spectrum propagatin from H
        
        A = fftshift(fft2(U0))
        A2 = A*H
        U = ifft2(fftshift(A2))
        
        return U

    def propagateZ(self, U0,dz):
        Hf = self.computeH(dx=self.dx, dy=self.dy, 
                      refractive_index=self.refractive_index, 
                      wavelength=self.wavelength, 
                      sizex=self.sizex, sizey=self.sizey, 
                      dz=dz, device=self.device)
                      
        U = self.propagateH(U0,Hf)
        return U, Hf

    def thinLensH(self, focus_length, wavelength, dx, dy, sizex, sizey, device, rect_s=None):
        '''
        x = np.arange(-np.floor(sizex/2),sizex/2)
        y = np.arange(-np.floor(sizey/2),sizey/2)
        
        Dx = x * dx
        Dy = y * dy
        
        X,Y = np.meshgrid(Dx,Dy)
        '''
        X,Y = self.meshgridNumpy(dx,dy,sizex,sizey,device)
        
        H = np.exp(-1j * np.pi / ( wavelength * focus_length ) * (X**2 + Y**2) )
        
        
        H = torch.as_tensor(H,dtype=torch.cfloat).to(device)
        
        if rect_s is not None:
            UR = self.rectApertureH(rect_s, dx, dy, sizex, sizey, device)
            #rectApertureH(self, rect_s, dx,dy,sizex,sizey, device):
            H = H*UR
        
        print('Thin lens size: ', H.size())
        
        return H
        
    def thinLensInverseH(self, focus_length, wavelength, dx, dy, sizex, sizey, device):
        x = np.arange(-np.floor(sizex/2),sizex/2)
        y = np.arange(-np.floor(sizey/2),sizey/2)
        
        Dx = x * dx
        Dy = y * dy
        
        X,Y = np.meshgrid(Dx,Dy)
        
        H = np.exp(+1j * np.pi / ( wavelength * focus_length ) * (X**2 + Y**2) )
        
        H = torch.as_tensor(H,dtype=torch.cfloat).to(device)
        
        return H
    
    def propagateThinLensH(self, U0, H):
        
        #print('Size of tensors: ', U0.size(), H.size())
        
        U = U0 * H
    
        return U
        
    def thinLensZ(self, U0, focus_length, rect_s=None):
        
        H = self.thinLensH(focus_length=focus_length, wavelength=self.wavelength, 
                      dx=self.dx, dy=self.dy, 
                      sizex=self.sizex, sizey=self.sizey, 
                      device=self.device, rect_s=rect_s)
        U = self.propagateThinLensH(U0, H)
        
        return U, H
           
    def shackHartmannArrayH(self, array_size, rect_x, rect_y, focus_length, wavelength, dx, dy, sizex,sizey, device):
        nx = array_size[0]
        ny = array_size[1]
        
        size_x = int(rect_x/nx/dx) #floor to nearest integer
        size_y = int(rect_y/ny/dy)
        
        mH = self.thinLensH(focus_length=focus_length, wavelength=wavelength,
                        dx=dx, dy=dy, sizex=size_x, sizey=size_y, device=device)
        
        #print('Size of each lenselet in the array: ',  size_x*dx*10**3, 'x', size_y*dy*10**3, 'mm^2')
        
        nsx = int(size_x*nx)
        nsy = int(size_y*ny)
        
        #print('rect: ', rect_x, rect_y)
        #print('size:', size_x, size_y)
        #print('ns:', nsx, nsy)
        
        H0 = torch.zeros((nsx,nsy),dtype=torch.cfloat)
        for i in range(nx):
            for j in range(ny):
                #print(i,j)
                H0[size_x*i:size_x*(i+1), size_y*j:size_y*(j+1)] = mH
        
        H1 = torch.zeros((sizex,sizey),dtype=torch.cfloat).to(device)
        sx = int((sizex-nsx)/2)
        sy = int((sizey-nsy)/2)
        H1[sx:sx+nsx,sy:sy+nsy] = H0
        
        params = np.array([ [size_x, size_y], [sx, sx+nsx], [sy, sy+nsy], [nx, ny] ])
            
        H = H1 #torch.as_tensor(H1, dtype=torch.cfloat).to(device)

        return H, params
        
    #def propagateShackHartmannArrayH(self, U0, H):
    # 
    #    U = U0 * H
    #    
    #    return U
        
    def propagateShackHartmannArrayZ(self, U0, array_size, rect_x, rect_y, focus_length):
        
        H, params = self.shackHartmannArrayH(array_size=array_size, rect_x = rect_x, rect_y=rect_y,
                                     focus_length=focus_length, 
                                     wavelength=self.wavelength, 
                                     dx=self.dx, dy=self.dy, 
                                     sizex=self.sizex, sizey=self.sizey, device=self.device)
        #print(H.size(), params)                             
        U = self.multiply(U0, H)
        #U = None
        
        return U, H, params
        
    def lensletZ(self, U0, array_size, rect_x, rect_y, focus_length):
        
        H, params = self.shackHartmannArrayH(array_size=array_size, rect_x = rect_x, rect_y=rect_y,
                                     focus_length=focus_length, 
                                     wavelength=self.wavelength, 
                                     dx=self.dx, dy=self.dy, 
                                     sizex=self.sizex, sizey=self.sizey, device=self.device)
        #print(H.size(), params)                             
        U = self.multiply(U0, H)
        #U = None
        
        return U, H, params
        
    def GaussLaguerreMode(self, X, Y, wavelength, 
                        w0, A, m, n, z, normalize=True):
    
        
        #X,Y = self.meshgridNumpy(dx,dy,sizex,sizey,device)
        
        am = np.abs(m)
        
        z0 = np.pi * w0**2/wavelength
        z1 = z/z0
        w = w0 * np.sqrt(1+z1**2)
        print('Gaussian Spot Size: ', w*10**3, 'mm and Rayleigh Length: ', z0*10**3, 'mm') 
        rho = np.sqrt(X**2+Y**2)
        rho1 = rho/w
        psi = np.arctan(z1)
        phi = np.arctan2(Y,X)
        G = w0/w * np.exp(-rho1**2) * np.exp(1j*rho1**2*z1) * np.exp(-1j*psi)
        R_nm = (np.sqrt(2) * rho1)**am * eval_genlaguerre((n-am)/2,am,2*rho1**2)
        Phi_m = np.exp(1j*m*phi)
        Z_n = np.exp(-1j*n*psi)
        
        U = A * G * R_nm * Phi_m * Z_n
        
        if normalize:
            U_max = np.max(np.abs(U))
            U = U/U_max
        
        return U
        
    def GaussLaugerreWaveH(self, wavelength, dx, dy, sizex, sizey, w0, A, device):
        size_x = sizex
        size_y = sizey
        
        X,Y = self.meshgridNumpy(dx,dy,size_x,size_y,device)
        z = 0.0
        
        H = self.GaussLaguerreMode(X,Y,wavelength, w0, A, 0,0,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 1,1,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 3,5,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 5,9,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 7,13,z)+ \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 9,17,z)
        H = torch.as_tensor(H, dtype=torch.cfloat).to(device)
        
        return H
        
    def GaussLaugerreWaveZ(self, w0, A):
        H = self.GaussLaugerreWaveH(wavelength=self.wavelength, dx=self.dx, dy=self.dy, 
                               sizex=self.sizex, sizey=self.sizey, 
                               w0=w0, A=A, device=self.device)
                               
        return H
        
        
    def GaussLaugerreArrayH(self, array_size, rect_x, rect_y, wavelength, dx, dy, sizex, sizey, w0, A, device):
        nx = array_size[0]
        ny = array_size[1]
        
        size_x = int(rect_x/nx/dx) #floor to nearest integer
        size_y = int(rect_y/ny/dy)
        
        X,Y = self.meshgridNumpy(dx,dy,size_x,size_y,device)
        z = 0.0
        
        mH = self.GaussLaguerreMode(X,Y,wavelength, w0, A, 0,0,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 1,1,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 3,5,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 5,9,z) + \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 7,13,z)+ \
             self.GaussLaguerreMode(X,Y,wavelength, w0, A, 9,17,z)
        mH = torch.as_tensor(mH, dtype=torch.cfloat).to(device)
        #print('Size of each lenselet in the array: ',  size_x*dx*10**3, 'x', size_y*dy*10**3, 'mm^2')
        
        nsx = int(size_x*nx)
        nsy = int(size_y*ny)
        
        #print('rect: ', rect_x, rect_y)
        #print('size:', size_x, size_y)
        #print('ns:', nsx, nsy)
        
        H0 = torch.zeros((nsx,nsy),dtype=torch.cfloat)
        for i in range(nx):
            for j in range(ny):
                #print(i,j)
                H0[size_x*i:size_x*(i+1), size_y*j:size_y*(j+1)] = mH
        
        H1 = torch.zeros((sizex,sizey),dtype=torch.cfloat).to(device)
        sx = int((sizex-nsx)/2)
        sy = int((sizey-nsy)/2)
        H1[sx:sx+nsx,sy:sy+nsy] = H0
        
        params = np.array([ [size_x, size_y], [sx, sx+nsx], [sy, sy+nsy], [nx, ny] ])
            
        H = H1 #torch.as_tensor(H1, dtype=torch.cfloat).to(device)

        return H, params
        
    def GaussLaguerreArrayZ(self, U0, array_size, rect_x, rect_y, w0, A):
    
        H, params = self.GaussLaugerreArrayH(array_size=array_size, rect_x=rect_x, rect_y=rect_y, 
                                wavelength=self.wavelength, 
                                dx=self.dx, dy=self.dy, sizex=self.sizex, sizey=self.sizey, 
                                w0=w0, A=A, device=self.device)
                                
        U = self.multiply(U0,H)
        
        return U, H, params
