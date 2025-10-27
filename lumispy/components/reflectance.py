# -*- coding: utf-8 -*-
# Copyright 2019-2025 The LumiSpy developers
#
# This file is part of LumiSpy.
#
# LumiSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the license, or
# (at your option) any later version.
#
# LumiSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LumiSpy. If not, see <https://www.gnu.org/licenses/#GPL>.


from hyperspy.component import Component
from lumispy.utils.axes import compare_extent
import numpy as np

class Reflectance(Component):
    r""" Calculate the reflectance (Fresnel equation) at the interface between 
    2 optical media, with n1 and n2 complex refractive indexes (energy dependent). 
    
    .. math::
        
        n_1\sin(\theta_1) &= n_2\sin(\theta_2) \\
        R_s(\theta) &= \left|\frac{n_1\cos(\theta_1)-n_2\cos(\theta_2)}
                        {n_1\cos(\theta_1)+n_2\cos(\theta_2)}\right|^2 \\
        R_p(\theta) &= \left|\frac{n_1\cos(\theta_2)-n_2\cos(\theta_1)}
                        {n_1\cos(\theta_2)+n_2\cos(\theta_1)}\right|^2 \\
        R_{unpol} &= \frac{R_s+R_p}{2}
    
    Parameters
    ----------
    theta : float
        Incidence angle in degree (with respect to the normal to n1/n2 interface)
        Default is 0.
    n1 : dict
        Dictionary for n1 index. Please use `get_nk` function to format.
    n2 : dict
        Dictionary for n2 index. Please use `get_nk` function to format.
    pol : string
        Polarisation s, p or unpolarised. Default is None, unpolarised light
    """
    
    def __init__(self, theta=0, n1=None, n2=None, pol=None):
        # Acthung to wavlength unit in index 
        # Define the parameters
        Component.__init__(self, ('theta',))
        self.name = 'Fresnel Reflectance component'
        self.theta.value = theta
        self.theta.units = 'Degree'
        self.theta.free = False
    
        self._n1 = n1
        self._n2 = n2
        self._pol = pol
        
        # Boundaries
        self.theta.bmin = 0.
        self.theta.bmax = 90.
        

    def function(self, x):
        self.check_compatibilty()
        t = np.radians(self.theta.value)
        n1, n2 = self._n1.function, self._n2.function
        if self._pol == 's':
            num = n1(x)*np.cos(t)-np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2)
            den = n1(x)*np.cos(t)+np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2)
            R=np.abs(num/den)**2
        if self._pol == 'p':
            num = n1(x)*np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2) - n2(x)**2*np.cos(t)
            den = n1(x)*np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2) + n2(x)**2*np.cos(t)
            R=np.abs(num/den)**2
        if self._pol == None:
            num_s = n1(x)*np.cos(t)-np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2)
            den_s = n1(x)*np.cos(t)+np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2)
            R_s =  np.abs(num_s/den_s)**2
            num_p = n1(x)*np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2) - n2(x)**2*np.cos(t)**2
            den_p = n1(x)*np.sqrt(n2(x)**2-n1(x)**2*np.sin(t)**2) + n2(x)**2*np.cos(t)**2
            R_p =  np.abs(num_p/den_p)**2
            R=(R_s+R_p)/2
        return R
        
    def check_compatibilty(self):
        if self._axes_manager:
            n1 = self._n1
            n2 = self._n2
            n1_units = n1['x_units']
            n2_units = n2['x_units']
            s_axis = self._axes_manager[-1]
            s_axis_extent = s_axis.axes_manager.signal_extent
            x_min, x_max = s_axis_extent
            if n1_units.lower() != s_axis.units.lower():
                raise Exception(f'n1 refractive index in {n1_units} while'
                                f'signal axis unit is {s_axis.units}.')
            if n2_units.lower() != s_axis.units.lower():
                raise Exception(f'n2 refractive index in {n2_units} while'
                                f'signal axis unit is {s_axis.units}.')
            if not compare_extent(n1['x_extent'], s_axis_extent):
                raise Exception(f'n1 not available on the signal {s_axis.name}'
                                f' range {x_min:.2f}-{x_max:.2f} {s_axis.units}')
            if not compare_extent(n2['x_extent'], s_axis_extent):
                raise Exception(f'n2 not available on the signal {s_axis.name}'
                                f' range {x_min:.2f}-{x_max:.2f} {s_axis.units}')
        else:
            pass