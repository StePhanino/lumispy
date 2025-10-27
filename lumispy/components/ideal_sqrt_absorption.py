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

import numpy as np
from scipy.special import erfc, erfi
from lumispy.components import UrbachTail
from hyperspy.component import Component

class IdealSqrtAbsorption(Component):
    r'''Ideal Sqrt absorption with or without Urbach tail convolution.
        
        - Without Urbach tail:
            
            .. math::
            
                \alpha_{ideal}(E) = 
                \begin{cases}
                0 & \text{if} \quad E \le E_g \\
                a_0 \sqrt{\frac{E-E_g}{E_0-E_g}} & \text{if} \quad E>E_g
                \end{cases}
            
        - With Urbach tail defined here :py:class:`urbach_tail.UrbachTail`:
            
            .. math::
            
                \DeclareMathOperator\erf{erf}
                \DeclareMathOperator\erfi{erfi}
                \alpha_0(E) =
                \begin{cases}
                Ce^{\frac{E-E_g}{g}} & \text{if} \quad E \le E_g \\
                a_0\sqrt{\frac{E-E_g}{E_0-E_g}} 
                + C \left[e^\frac{E-E_g}{g}\erf{\sqrt{\frac{E-E_g}{g}}} 
                          - e^\frac{E_g-E}{g}\erfi{\sqrt{\frac{E-E_g}{g}}}\right]
                & \text{if} \quad E>E_g
                \end{cases}
                
            where
            
            - :math:`C=\tfrac{a_0}{4}\sqrt{\frac{\pi g}{E_0-E_g}}`, 
            - :math:`\erf(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2} \, dt` 
              is the `error function`,
            - :math:`\erfi(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{t^2} \, dt` 
              is the `imaginary error function` [1]_ .
            
        ============== ================================= ========================
        Variable        Parameter                         Units                  
        ============== ================================= ========================
        :math:`E_g`     Band gap                          eV                     
        :math:`E_0`     Ref energy level                  eV                     
        :math:`a_0`     Abs  coefficient at :math:`E_0`   :math:`\text{cm}^{-1}` 
        g               Urbach tail width                 eV
        ============== ================================= ========================
    
        Parameters
        ----------
        Eg : float
            Band gap energy. The default is 1.42 (GaAs).
        a0 : float
            Absorption coefficient at E0. The default is 14800 (cm-1).
            Be careful with units.
        E0 : float
            Reference energy for a0. The default is 1.6 (GaAs).
        g : float 
            Width of the Urbach tail (eV).
        include_tail : boolean
            If True it takes into account the Urbach tail (analytical solution).
            If False, it's the ideal sqrt absorption.
    
        References
        -------
        .. [1] `Wikipedia Error Function 
                <https://en.wikipedia.org/wiki/Error_function>`_

    '''
    def __init__(self, Eg=1.4, a0=14800, E0=1.6, g=0.015, include_tail=False):
        Component.__init__(self, ('Eg', 'a0', 'E0', 'g'))
        if include_tail:
            self.name = 'Ideal sqrt-shaped absorption coefficient with Urbach tail'
        else:
            self.name = 'Ideal sqrt-shaped absorption coefficient'
        self.Eg.value = Eg
        self.Eg.units = 'eV'
        self.Eg.bmin = 0
        self.a0.value = a0
        self.a0.units = 'cm-1'
        self.a0.bmin = 0
        self.E0.value = E0
        self.E0.units = 'eV'
        self.E0.bmin = 0
        self.g.value = g
        self.g.bmin = 0
        self._include_tail = include_tail
        self.tail_type = UrbachTail().name
        
    def convolution_tail(self, x):
        '''
        Analytical convolution with an Urbach tail
        '''
        Eg = self.Eg.value
        a0 = self.a0.value
        E0 = self.E0.value
        g = self.g.value
        #This definition is important!
        y = x-Eg
        
        _c1 = 0.25*a0*np.sqrt(g*np.pi/(E0-Eg))

        def _fpos(y,a0,Eg,E0,g):
            return (a0*np.sqrt(y/(E0-Eg)) 
                    +_c1*(np.exp(y/g)*erfc(np.sqrt(y/g))
                          -np.exp(-y/g)*erfi(np.sqrt(y/g)))
                    )
        
        def _fneg(y,g):
            return _c1*np.exp(y/g)
            
        '''    
        _fpos = lambda y : (a0*np.sqrt(y/(E0-Eg)) 
                            +_c1*(np.exp(y/g)*erfc(np.sqrt(y/g))
                                  -np.exp(-y/g)*erfi(np.sqrt(y/g)))
                            )
        _fneg = lambda y : _c1*np.exp(y/g)
        '''
        _f = np.piecewise(y, [y>=0, y<0], [_fpos, _fneg])
        return _f
        
    def function(self,x):
        Eg = self.Eg.value
        a0 = self.a0.value
        E0 = self.E0.value
        if self._include_tail:
            _f = self.convolution_tail(x)
        else:
            #In this case g parameter is still defined but not used!
            _f = np.piecewise(x, [x<=Eg, x>Eg], 
                              [lambda x : 0, 
                               lambda x : a0*np.sqrt((x-Eg)/(E0-Eg))])
        return _f