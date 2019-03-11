from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')


import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import scipy
from scipy import asarray as ar,exp

import matplotlib.image as mpimg


plt.rcParams.update({'font.size': 22})



################################################## BEST  RUN SO FAR ###################################################

#Window 0.1 -> 9.0 on both (not sure but one of them may have actually been on 2-9.

f1o= np.loadtxt("nov14ov/Nov-14-18, 5_04 PM, Scan1, 0_0 Deg.dat",skiprows=1,unpack=True)
f2o= np.loadtxt("nov14ov/Nov-14-18, 5_17 PM, Scan1, 0_0 Deg.dat", skiprows=1,unpack=True)
f3o= np.loadtxt("nov14ov/Nov-14-18, 5_30 PM, Scan1, -14_4 Deg.dat", skiprows=1,unpack=True)
f4o= np.loadtxt("nov14ov/Nov-14-18, 5_42 PM, Scan1, -28_8 Deg.dat", skiprows=1,unpack=True)
f5o= np.loadtxt("nov14ov/Nov-14-18, 5_54 PM, Scan1, -43_2 Deg.dat",skiprows=1,unpack=True)
f6o= np.loadtxt("nov14ov/Nov-14-18, 6_07 PM, Scan1, -57_6 Deg.dat",skiprows=1,unpack=True)
f7o= np.loadtxt("nov14ov/Nov-14-18, 6_19 PM, Scan1, -72_0 Deg.dat", skiprows=1,unpack=True)
f8o= np.loadtxt("nov14ov/Nov-14-18, 6_32 PM, Scan1, -86_4 Deg.dat", skiprows=1,unpack=True)
f9o= np.loadtxt("nov14ov/Nov-14-18, 6_44 PM, Scan1, -100_8 Deg.dat",skiprows=1,unpack=True)
f10o= np.loadtxt("nov14ov/Nov-14-18, 6_56 PM, Scan1, -115_2 Deg.dat",skiprows=1,unpack=True)
f11o= np.loadtxt("nov14ov/Nov-14-18, 7_09 PM, Scan1, -129_6 Deg.dat",skiprows=1,unpack=True)
f12o= np.loadtxt("nov14ov/Nov-14-18, 7_21 PM, Scan1, -144_0 Deg.dat",skiprows=1,unpack=True)
f13o= np.loadtxt("nov14ov/Nov-14-18, 7_34 PM, Scan1, -158_4 Deg.dat",skiprows=1,unpack=True)

f14o= np.loadtxt("nov14ov/Nov-14-18, 7_46 PM, Scan1, -172_8 Deg.dat",skiprows=1,unpack=True)
f15o= np.loadtxt("nov14ov/Nov-14-18, 7_58 PM, Scan1, -189_0 Deg.dat",skiprows=1,unpack=True)
f16o= np.loadtxt("nov14ov/Nov-14-18, 8_11 PM, Scan1, -201_6 Deg.dat",skiprows=1,unpack=True)
f17o= np.loadtxt("nov14ov/Nov-14-18, 8_23 PM, Scan1, -217_8 Deg.dat",skiprows=1,unpack=True)
f18o= np.loadtxt("nov14ov/Nov-14-18, 8_36 PM, Scan1, -230_4 Deg.dat",skiprows=1,unpack=True)
f19o= np.loadtxt("nov14ov/Nov-14-18, 8_48 PM, Scan1, -244_8 Deg.dat",skiprows=1,unpack=True)
f20o= np.loadtxt("nov14ov/Nov-14-18, 9_00 PM, Scan1, -259_2 Deg.dat",skiprows=1,unpack=True)
f21o= np.loadtxt("nov14ov/Nov-14-18, 9_13 PM, Scan1, -275_4 Deg.dat",skiprows=1,unpack=True)
f22o= np.loadtxt("nov14ov/Nov-14-18, 9_25 PM, Scan1, -288_0 Deg.dat",skiprows=1,unpack=True)
f23o= np.loadtxt("nov14ov/Nov-14-18, 9_37 PM, Scan1, -302_4 Deg.dat",skiprows=1,unpack=True)
f24o= np.loadtxt("nov14ov/Nov-14-18, 9_50 PM, Scan1, -316_8 Deg.dat",skiprows=1,unpack=True)
f25o= np.loadtxt("nov14ov/Nov-14-18, 10_02 PM, Scan1, -331_2 Deg.dat",skiprows=1,unpack=True)
f26o= np.loadtxt("nov14ov/Nov-14-18, 10_15 PM, Scan1, -345_6 Deg.dat",skiprows=1,unpack=True)
f27o= np.loadtxt("nov14ov/Nov-14-18, 10_27 PM, Scan1, -361_8 Deg.dat",skiprows=1,unpack=True)


f2dist = np.column_stack((f2o[0,:],f3o[0,:],f4o[0,:],f5o[0,:],f6o[0,:],f7o[0,:],f8o[0,:],f9o[0,:],f10o[0,:],f11o[0,:],f12o[0,:],
f13o[0,:],f14o[0,:],f15o[0,:],f16o[0,:],f17o[0,:],f18o[0,:],f19o[0,:],f20o[0,:],f21o[0,:],f22o[0,:],f23o[0,:],f24o[0,:],f25o[0,:],f26o[0,:],f27o[0,:]))
f2pix = np.column_stack((f2o[1,:],f3o[1,:],f4o[1,:],f5o[1,:],f6o[1,:],f7o[1,:],f8o[1,:],f9o[1,:],f10o[1,:],f11o[1,:],f12o[1,:],
f13o[1,:],f14o[1,:],f15o[1,:],f16o[1,:],f17o[1,:],f18o[1,:],f19o[1,:],f20o[1,:],f21o[1,:],f22o[1,:],f23o[1,:],f24o[1,:],f25o[1,:],f26o[1,:],f27o[1,:]))


y2_list = f2dist
x2_list = np.linspace(0.,360.,27.)
z2_list = f2pix

theta32 = np.linspace(0.,360.,z2_list.shape[1])



#N = int(len(z_list)**.5)
#z = z_list.reshape(N, N)
#plt.imshow(z_list.T, extent=(np.amin(y_list), np.amax(y_list),np.amin(x_list), np.amax(x_list)), aspect = 'auto')
print("sinogram")
plt.imshow(z2_list, extent=(np.amin(x2_list), np.amax(x2_list),np.amin(y2_list), np.amax(y2_list)), aspect = 'auto')
plt.ylabel("Distance moved (mm)")
plt.xlabel("Angle (degrees)")
acol = plt.colorbar()
#acol.ax.set_title('This is a title')
acol.set_label(' # of counts obtained')

#plt.imshow(z, extent=(np.amin(x_list), np.amax(x_list), np.amin(y_list), np.amax(y_list)), norm=LogNorm(), aspect = 'auto')
#plt.imshow(airadon4)
plt.show()

plt.imshow(iradon(z2_list,theta32,circle=True).T)
#plt.colorbar()
plt.xlabel("Position (to mm scale)")
plt.ylabel("Position (to mm scale)")


plt.show()
print("Reconstructed image")
plt.imshow(iradon(z2_list,theta32,circle=True).T, extent=(np.amin(y2_list), np.amax(y2_list),np.amin(x2_list), np.amax(x2_list)),aspect='auto')
plt.xlabel("Distance")
plt.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
plt.show()
 



############################ BEST RUN(?) BUT ONLY FOR SINGLE SOURCE :(


f1o= np.loadtxt("409py/0d.dat",skiprows=1,unpack=True)
f2o= np.loadtxt("409py/10_8d.dat", skiprows=1,unpack=True)
f3o= np.loadtxt("409py/21_6d.dat", skiprows=1,unpack=True)
f4o= np.loadtxt("409py/32_4d.dat", skiprows=1,unpack=True)
f5o= np.loadtxt("409py/43_2d.dat",skiprows=1,unpack=True)
f6o= np.loadtxt("409py/54.dat", skiprows=1,unpack=True)
f7o= np.loadtxt("409py/64_8.dat", skiprows=1,unpack=True)
f8o= np.loadtxt("409py/75_6.dat", skiprows=1,unpack=True)
f9o= np.loadtxt("409py/86_4.dat",skiprows=1,unpack=True)
f10o= np.loadtxt("409py/97_2.dat", skiprows=1,unpack=True)
f11o= np.loadtxt("409py/108.dat", skiprows=1,unpack=True)
f12o= np.loadtxt("409py/118_8.dat", skiprows=1,unpack=True)
f13o= np.loadtxt("409py/129_6.dat",skiprows=1,unpack=True)
f14o= np.loadtxt("409py/140_4.dat", skiprows=1,unpack=True)
f15o= np.loadtxt("409py/151_2.dat", skiprows=1,unpack=True)
f16o= np.loadtxt("409py/162.dat", skiprows=1,unpack=True)
f17o= np.loadtxt("409py/172_8.dat",skiprows=1,unpack=True)
f18o= np.loadtxt("409py/183_6.dat", skiprows=1,unpack=True)
f19o= np.loadtxt("409py/194_4.dat", skiprows=1,unpack=True)
f20o= np.loadtxt("409py/205_2.dat", skiprows=1,unpack=True)
f21o= np.loadtxt("409py/216.dat", skiprows=1,unpack=True)
f22o= np.loadtxt("409py/226_8.dat",skiprows=1,unpack=True)
f23o= np.loadtxt("409py/237_6.dat", skiprows=1,unpack=True)
f24o= np.loadtxt("409py/248_4.dat", skiprows=1,unpack=True)
f25o= np.loadtxt("409py/259_2.dat", skiprows=1,unpack=True)
f26o= np.loadtxt("409py/270.dat",skiprows=1,unpack=True)
f27o= np.loadtxt("409py/280_8.dat", skiprows=1,unpack=True)
f28o= np.loadtxt("409py/291_6.dat", skiprows=1,unpack=True)
f29o= np.loadtxt("409py/302_4.dat", skiprows=1,unpack=True)
f30o= np.loadtxt("409py/313_2.dat",skiprows=1,unpack=True)
f31o= np.loadtxt("409py/324.dat", skiprows=1,unpack=True)
f32o= np.loadtxt("409py/336_6.dat", skiprows=1,unpack=True)
f33o= np.loadtxt("409py/345_6.dat", skiprows=1,unpack=True)
f34o= np.loadtxt("409py/356_4.dat",skiprows=1,unpack=True)
f35o= np.loadtxt("409py/361_8.dat", skiprows=1,unpack=True)


f1dist2 = np.column_stack((f1o[0,:],f2o[0,:],f3o[0,:],f4o[0,:],f5o[0,:],f6o[0,:],f7o[0,:],f8o[0,:],f9o[0,:],f10o[0,:],
                          f11o[0,:],f12o[0,:],f13o[0,:],f14o[0,:],f15o[0,:],f16o[0,:],f17o[0,:]
                          ,f18o[0,:],f19o[0,:],f20o[0,:],f21o[0,:],f22o[0,:],f23o[0,:],f24o[0,:],f25o[0,:]
                          ,f26o[0,:],f27o[0,:],f28o[0,:],f29o[0,:],f30o[0,:],f31o[0,:],f32o[0,:],f33o[0,:]
                          ,f34o[0,:],f35o[0,:]))
f1pix2 = np.column_stack((f1o[1,:],f2o[1,:],f3o[1,:],f4o[1,:],f5o[1,:],f6o[1,:],f7o[1,:],f8o[1,:],f9o[1,:],
                         f10o[1,:],
                          f11o[1,:],f12o[1,:],f13o[1,:],f14o[1,:],f15o[1,:],f16o[1,:],f17o[1,:]
                          ,f18o[1,:],f19o[1,:],f20o[1,:],f21o[1,:],f22o[1,:],f23o[1,:],f24o[1,:],f25o[1,:]
                          ,f26o[1,:],f27o[1,:],f28o[1,:],f29o[1,:],f30o[1,:],f31o[1,:],f32o[1,:],f33o[1,:]
                          ,f34o[1,:],f35o[1,:]))


 
theta33 = np.linspace(0.,360.,35.)
airadon4 = iradon(f1pix2, theta33, circle=True)
plt.imshow(airadon4)
plt.xlabel("Position (to mm scale)")
plt.ylabel("Position (to mm scale)")
plt.show()

y_list = f1dist2
x_list = np.linspace(0.,360.,35.)
z_list = f1pix2


print("sinogram")
plt.imshow(z_list, extent=(np.amin(x_list), np.amax(x_list),np.amin(y_list), np.amax(y_list)), aspect = 'auto')
plt.colorbar()
plt.ylabel("Distance moved (mm)")
plt.xlabel("Angle (degrees)")


plt.show()

print("Reconstructed image")
plt.imshow(iradon(f1pix2,theta33,circle=True), extent=(np.amin(y_list), np.amax(y_list),np.amin(x_list), np.amax(x_list)),aspect='auto')
plt.xlabel("Distance moved (mm)")
plt.ylabel("Projection Angle")
plt.colorbar()
plt.show() 




######86.4 and 97.2 degs --- CURVE FITS!
faa2 = np.loadtxt("nov14ov/Nov-14-18, 6_44 PM, Scan1, -100_8 Deg.dat",skiprows=1)
#plt.plot(faa2[:,0], faa2[:,1])
#plt.xlim(8,14)
#plt.show()

faa = np.loadtxt("nov14ov/Nov-14-18, 10_27 PM, Scan1, -361_8 Deg.dat",skiprows=1)
#plt.plot(faa[:,0],faa[:,1])
#plt.xlim(3.5, 12)
#plt.show()



#x2 = faa[:,0]
#y2 = faa[:,1]

x2 = faa[:,0]
y2 = faa[:,1]


n = len(x2)                          #the number of data
#mean = sum(x2*y2)/(sum(x2)+sum(y2))               #note this correction
#sigma = np.sqrt(sum(y2*(x2-mean)**2)/(sum(x2)+sum(y2)))   #note this correction

mean = 7.449062
sigma = 1.64062

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
    
popt,pcov = curve_fit(gaus,x2,y2,p0=[174.574,mean,sigma])

plt.plot(x2,y2,'b+:',label='data')
plt.plot(x2,gaus(x2,*popt),'ro:',label='fit')
plt.legend()
plt.title('Fig. 3 - Fit for Time Constant')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()

plt.plot(x2,y2,'b+:',label='data')
plt.plot(x2,174.5474*np.exp(-(x2 - 7.449062)**2/(2*1.643062**2)),'ro:',label='fit')
plt.errorbar(x2,y2,yerr=np.sqrt(y2),xerr=0.05, linestyle='-')
plt.legend()
plt.xlabel('Distance (mm)')
plt.ylabel('Counts')
plt.show()

aio = np.loadtxt('foo.txt',delimiter=',')
x2=aio[:,0]
y2=aio[:,1]

mean2 = 7.449062
sigma2 = 1.64062

    
popt,pcov = curve_fit(gaus,x2,y2,p0=[200,mean2,sigma2])

plt.plot(x2,y2,'b+:',label='data')
plt.plot(x2,gaus(x2,*popt),'ro:',label='fit')
plt.errorbar(x2,y2,yerr=np.sqrt(y2),xerr=0.05, linestyle='-')
plt.legend()
plt.xlabel('Distance (mm)')
plt.ylabel('Counts')
plt.show()






def gaussian(x, mu, sig):
    return (1/(sig * np.sqrt(2*np.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

sig = 1.822
#sig = 0.015
#mu = 0.33
mu = 7.5
xdata = faa[:,0]
ydata2 = faa[:,1]
#plt.xlim(0,1.5)
ydata = gaussian(xdata, mu, sig)
popt2, pcov2 = curve_fit(gaussian,xdata,ydata2)
#plt.plot(xdata - 0.012, .323 + gaussian(xdata, *popt)/48)
plt.plot(xdata, gaussian(xdata, *popt2), label='Gaussian')
plt.plot(xdata,ydata2, label='Data')

plt.legend()
#plt.title("Zero Field Transition Plot")
plt.show()


a34t = np.array([6.2, 7.5, 10, 12.5])
b34t = np.array([17, 20, 25, 28])
plt.plot(a34t, b34t)
plt.errorbar(a34t,b34t, yerr=(b34t)**0.25,xerr=0.005)
plt.xlabel("Width of slit (mm)")
plt.ylabel("Peak Count")
plt.show()



