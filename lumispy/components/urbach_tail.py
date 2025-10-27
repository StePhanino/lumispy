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

from hyperspy.components1d import Expression

class UrbachTail(Expression):
    r"""Urbach tail in the form
    
        .. math:: 
        
            \tau(E) = \tfrac{1}{2g} e^{-\left|\tfrac{E}{g}\right|}
        
        
        ============== ============== ========
         Variable        Parameter      Units  
        ============== ============== ========
         :math:`g`       Tail width     eV     
        ============== ============== ========
         
        Parameters
        ----------
        g : float
            Tail width in eV. The default is 0.015 (GaAs).
     """

    def __init__(self, 
                 g=0.015,
                 module=["numpy","scipy"],
                 **kwargs):
        
        super().__init__(
            name='Urbach tail component',
            g=g,
            expression="0.5/g*exp(-abs(x/g))",
            module=module,
            autodoc=False,
            **kwargs
            
        )
        # Units
        self.g.units = 'eV'
        
        #Boundaries
        self.g.bmin = 0
        self.g.bmax = None