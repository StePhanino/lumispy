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


from refractiveindex import RefractiveIndexMaterial
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

DICT_X = {'eV' : 'Energy', 
          'um' : 'Wavelength'}


class RefractiveIndex():
    r"""
    It takes 3 possible inputs:
        
        - custom refractive index array :
            x (np.arry), x_units(string), n (complex)(np.array) 
        - constant value on the entire spectrum:
            n (float) 
        - reference to refractiveindex :
            shelf=str, book=str, page=str
    
    It returns a dictionary with x_min, x_max, x_units and the 
    interpolation function n(x).
    
    Supported x_units: um and eV.

    Parameters
    ----------
    output : str
        x axis in 'nm' on 'eV'. The default is 'eV'.
    **kwargs :
        It accepts en (in eV) or wl (in um) and complex n (n+jk) 
        or a reference from refractive index database (shelf, book, page)
    
    Examples
    --------
    
    >>> n1 = RefractiveIndex(n=1)
    >>> en=np.linspace(0.5,2,100)
    >>> n2_array = (np.random.uniform(0.9, 1.1, 100) + 
                    1.j * np.random.uniform(-1, 1, 100))
    >>> n2 = RefractiveIndex(x=en, x_units='eV', n=n2_array)
    >>> n3 = RefractiveIndex(shelf='main', book='GaAs', page='Papatryfonos')

    """
    
    def __init__(self, output='eV', **kwargs):
        self.n = None
        self.x = None
        self.units = output
        self.function = None
        self.extent = None
        
        if output not in ['nm','eV']:
            raise ValueError('Output possibilities are nm or eV')
        
        if  'x' in kwargs and 'x_units' in kwargs and 'n' in kwargs:
            n = kwargs.get('n')
            x_units = kwargs.get('x_units')
            x = kwargs.get('x')
            if x_units in ['um','eV']:
                if x_units == output:
                    x_out = x
                elif x_units == 'um':
                    if output == 'nm':
                        x_out = x*1000
                    elif output == 'eV':
                        x_out = np.sort(1.23984/x)
                        n = n[::-1]
                else:
                    x_out = np.sort(1239.84/x)
                    n = n[::-1]
                f_n = CubicSpline(x_out,n,extrapolate=False)
            else:
                raise ValueError('Only um and eV supported as x axis units.')
                
        elif 'shelf' and 'book' and 'page' in kwargs:
            
            shelf = kwargs.get('shelf')
            book = kwargs.get('book')
            page = kwargs.get('page')
            refr_object = RefractiveIndexMaterial(shelf=shelf, 
                                                book=book, 
                                                page=page)
            if output == 'eV':
                x_out = np.sort(1.23984/refr_object.material.originalData['wavelength (um)'])
                n = refr_object.material.originalData['n'][::-1]
            elif output == 'nm':
                x_out = 1000*refr_object.material.originalData['wavelength (um)']
                n = refr_object.material.originalData['n']
            
            f_n = CubicSpline(x_out,n,extrapolate=False)
            extent = (x_out[0], x_out[-1])
            
        elif 'n' in kwargs and len(kwargs)==1:
            n = complex(kwargs.get('n'))
            x_out = np.linspace(0,10,100)
            f_n = np.vectorize(lambda x : n)
            extent = None
            
        self.x = x_out
        self.n = n
        self.function = f_n
        self.units = output
        self.extent = extent
            
    def plot(self):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            
            ax.plot(self.x, np.real(self.n),
                    color='steelblue',label='Real part')
            xlabel = f"{DICT_X[self.units]} ({self.units})"
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Real part')
            if not np.all(np.imag(self.n)==0):
                axr = ax.twinx()
                axr.plot(self.x, np.imag(self.n), 
                         color='maroon', label='Imaginary part')
                axr.set_ylabel('Imaginary part')
            ax.set_title('Refractive index dispersion')
            ax.legend(loc='upper left')
            