from hyperspy.component import Component
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat_fromstr, ufloat
from scipy.constants import physical_constants
c_const = physical_constants['speed of light in vacuum'][0] #m/s
h_const = physical_constants['Planck constant in eV/Hz'][0] #eV/Hz
k_const = physical_constants['Boltzmann constant in eV/K'][0] #eV/K

class GeneralizedPlanck(Component):
    r'''A component for the Normalised (to 1) Generalised Planck law. 
        We mainly follow [Chen2021]_ and references therein.
        
        .. math::
            \Phi(E) &= A(E) \times \frac{E^2}{4\pi^2\hbar^3c_0^2} \times 
                \frac{1}{e^{\frac{E - \Delta E_f}{k_BT}}-1} \\
            A(E) &= \left[1-R(E)\right]\times\left[1-e^{-\alpha(E)d}\right]
        
        where :math:`E = \hbar\omega`, :math:`\Delta E_f = E_f^c - E_f^v` and 
        :math:`R(E)` is the reflectance. d is the absorption depth 
        (simple assumption of an homogeneous slab).
        
        The absorption coefficient :math:`\alpha(E)` can be written as follow:
        
        .. math::
            \alpha(E) &= \alpha_0(E)\times(f_v-f_c) \\
            f_v &= \frac{1}{e^\frac{\epsilon_h-E_f^v}{k_BT}} = 
                \frac{1}{e^\frac{p(E-E_g)}{k_BT}} \\
            f_c &=  \frac{1}{e^\frac{\epsilon_e-E_f^c}{k_BT}} = 
                \frac{1}{e^\frac{(1-p)(E-E_g)}{k_BT}} \\
        
        where the :math:`\alpha_0` takes into account the Urbach tail via 
        this convolution integral:
            
        .. math::
            \alpha_0(E) = \frac{1}{2g} \int_{-\infty}^{E-E_g} 
                \alpha_{ideal}(E-\epsilon) 
                \underbrace{e^{-\left|\tfrac{\epsilon}{g}\right|}}_\text{Urbach tail} 
                \, d\epsilon
                
        where :math:`\alpha_{ideal}` is the ideal absorption coefficient. 
        For GaAs we can use [Casey1975]_ : 
            
        .. math::
            \alpha_{ideal}(E) = 
            \begin{cases}
            0 & \text{if} \quad E \le E_g \\
            a_0 \sqrt{\frac{E-E_g}{E_0-E_g}} & \text{if} \quad E>E_g
            \end{cases}
            
        ============== ========================= ========
        Variable        Parameter                 Units  
        ============== ========================= ========
        :math:`E_g`     Band gap                  eV     
        :math:`g`       Urbach tail width         eV     
        :math:`p`       Energy re-distribution    None   
        :math:`T`       Carrier temperature       K      
        :math:`d`       Absorption depth          nm     
        :math:`E_f^c`   Electrons Fermi level     eV     
        :math:`E_f^v`   Holes Fermi level         eV     
        ============== ========================= ========
        
        Parameters
        ----------
        Eg : float
            Band gap at the temperature T. The default is 1.42 (GaAs).
        g : float
            The Urbach tail width. The default is 0.015.
        p : float
            Redistribution parameter for photon energy. 
            The default is 0.8 for GaAs (given by the difference in holes and 
                                         electrons effective masses).
            
            .. math::
                
                \epsilon_h - E_V &:= p(E-E_g) \\
                E_C - \epsilon_c &:= (1-p)(E-E_g)
                
        ideal_absorption : IdealSqrtAbsorption component
            The ideal absorption used in the Generalised Planck Law.
        tail : UrbachTail component
            The Urbach tail component for convolution.
        reflectance : Reflectance component, optional
            The reflectance profile.
        analytical : bool
            In case of a sqrt shaped ideal absorption coefficient, if True, 
            it uses an explicit expression of the convolution integral.
            If False, it calculates the convolution with 
            :py:func:`scipy.integrate.quad_vec`.
        **kwargs : 
            All the parameters of the `ideal_absorption`, `reflectance` and 
            `tail` components.
    
        Returns
        -------
        None.
        
        References
        ----------
        .. [Chen2021] Chen et al., *Quantitative Assessment of Carrier Density by 
                      Cathodoluminescence. I. GaAs thin films and modeling*,
                      Phys. Rev. Applied **15**, 024006 (2021).
                      `DOI:10.1103/PhysRevApplied.15.024006 
                      <https://doi.org/10.1103/PhysRevApplied.15.024006>`_
                      
        .. [Casey1975] Casey et al., *Concentration dependence of the absorption 
                       coefficient for n− and p−type GaAs between 1.3 and 1.6 eV*,
                       J. Appl. Phys. **46**, 250 (1975).
                       `DOI:10.1063/1.321330
                       <https://doi.org/10.1063/1.321330>`_
                       
    '''
    
    def __init__(self, Eg=1.42, g=0.015, p=0.8, T=300, d=150, Efv=0, Efc=0.1,
                 ideal_abs_coeff=None,
                 tail=None, 
                 reflectance=None,
                 analytical=False,
                 **kwargs):
        
        self._reflectance = reflectance
        self._tail = tail
        self._ideal_abs_coeff = ideal_abs_coeff
        self._aux_component_list = [self._reflectance,
                                    self._ideal_abs_coeff,
                                    self._tail]
        self._kwargs = kwargs
        self._default_params = ['Eg', 'g', 'p', 'T', 'd', 'Efv', 'Efc']
                
        Component.__init__(self, self._default_params)
        self.name = "GeneralizedPlanck"
        self.Eg.value = Eg
        self.Eg.units = 'eV'
        self.Eg.bmin = 0
        self.g.value = g
        self.g.units = 'eV'
        self.g.bmin = 0
        self.p.value = p
        self.p.units = ''
        self.p.bmin = 0
        self.p.bmax = 1
        self.T.value = T
        self.T.units = 'K'
        self.T.bmin = 0
        self.d.value = d
        self.d.units = 'nm'
        self.d.bmin = 0
        self.Efc.value = Efc
        self.Efc.units = 'eV'
        self.Efv.value = Efv
        self.Efv.units = 'eV'
        self._analytical = analytical
        self.update_component()
        
    def update_component(self):
        param_list = []
        for component in self._aux_component_list:
            if hasattr(component, '_analytical'):
                setattr(component, '_analytical', self._analytical)
            if hasattr(component, '_tail'):
                setattr(component, '_tail', self._tail)
            if self.model:
                '''
                TODO: Update to deal with models
                Here we want to look when the component is appended to a model
                in order to append all other (tail, ref, id_abs) to the model too.
                Maybe use events???
                '''
                self.model.append(component)
                component.active = False
            for param in component.parameters:
                #update reflectance, tail and absorption with parameters in kwargs
                if param.name in self._kwargs:
                    value = self._kwargs[param.name]
                    setattr(getattr(component, param.name), 'value', value)
                    setattr(getattr(component, param.name), 'units', param.units)
                #For parameters in genp initialization, let's twin the component parameter
                #to the genp one.
                if param.name in self._default_params:
                    param.twin = getattr(self, param.name)
                    #setattr(p, 'value' , getattr(self, p.value))
                #Otherwise set new genp attributes from each component 
                else:
                    self.init_parameters([param.name])
                    setattr(getattr(self, param.name), 'twin', param)
                    setattr(getattr(self, param.name), 'units', param.units)
                    param_list.append(param.name)
                    param.free = False
                    
    def planck(self, x, mu=None):
        Efv = self.Efv.value
        Efc = self.Efc.value
        Eg = self.Eg.value
        T = self.T.value
        if mu is None:
            m = Efc + Eg - Efv #This is taken from stephane code
            '''
            Efc reference is the Ec : positive above Ec
            Efv reference is the Ev : positive below Ev
            '''
        else:
            m = mu
        #const = 2*np.pi / h_const**3 / c_const**2
        #_f = const*x**2 / (np.divide(np.exp(x-m), k_const*T)-1)
        #since working on normalized spectra...it's useless to put a constant.
        _f = x**2 / (np.exp((x-m)/(k_const*T))-1)
        return _f
    
    def fermi_distribution_vc(self, x, band='c'):
        Efv = self.Efv.value
        Efc = self.Efc.value
        Eg = self.Eg.value
        T = self.T.value
        p = self.p.value
        
        if band == 'v':
            return 1 / (np.exp(np.divide((p-1)*(x-Eg)-Efv, k_const*T)) +1)
        elif band == 'c':
            return  1 / (np.exp(np.divide(p*(x-Eg)-Efc, k_const*T)) +1)
        else:
            raise ValueError('Only c and v values allowed for band option')

    def abs_coeff_tail_occupation(self, x):
        occupation = (self.fermi_distribution_vc(x, band='v') 
                      - self.fermi_distribution_vc(x, band='c')
                      )
        _f = self._ideal_abs_coeff.function(x)*occupation
        return _f
    
    def absorption(self, x):
        '''TODO - generalize to other absorption type. 
        Here we assume a homogeneously excited slab.
        '''
    
        d = self.d.value/1e7 #a0 in abs_ideal in cm-1 and d in nm
        _f = ((1-self._reflectance.function(x)) 
              * (1 - np.exp(-self.abs_coeff_tail_occupation(x)*d))
             )
        return _f 
    
    def function(self, x):
        _pa = self.absorption(x) * self.planck(x)
        #normalisation by default
        _f = _pa/np.max(_pa)
        return _f
    
    def BandGapNarrowing(self, a=1.424, b='1.83(0.18)e-8' ):
        '''
        

        Parameters
        ----------
        a : TYPE, optional
            DESCRIPTION. The default is 1.424.
        b : TYPE, optional
            DESCRIPTION. The default is '1.83(0.18)e-8'.

        Returns
        -------
        p : TYPE
            DESCRIPTION.

        '''
        Eg = ufloat(self.Eg.value, self.Eg.std)
        b=ufloat_fromstr(b)
        p = (abs(Eg-a)/b)**3
        print(f'Estimated hole density = {p:.3u} 1/cm\N{SUPERSCRIPT THREE}')
        return p
    
    def plotComponents(self):
        from matplotlib.ticker import AutoMinorLocator
        if self.model:
            fig, (axt, axb) = plt.subplots(figsize=(6,6), 
                                           nrows=2, ncols=1,
                                           height_ratios=(0.4,1))
            axbr = axb.twinx()
            x = self.model.axis.axis
            xc = x[x.size//2]
            ycbl = self.abs_coeff_tail_occupation(xc)
            ycbr = self.absorption(xc)
            axb.annotate("",xy=(np.min(x), ycbl), xycoords='data',
                         xytext=(xc*0.99, ycbl), textcoords='data',
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3",
                                         color='dodgerblue',
                                         ls='--')
                         )
            axbr.annotate("",xy=(np.max(x), ycbr), xycoords='data',
                         xytext=(xc*1.01, ycbr), textcoords='data',
                         arrowprops=dict(arrowstyle="->", 
                                         connectionstyle="arc3",
                                         color='orangered',
                                         ls='--')
                         )
            df = (self.fermi_distribution_vc(x, band='v') 
                  -self.fermi_distribution_vc(x, band='c')
                  )
            axt.plot(x, df, label='f$_v$ - f$_c$')
            axt.plot(x, self.fermi_distribution_vc(x, band='v'), 
                     label='f$_v$')
            axt.plot(x, self.fermi_distribution_vc(x, band='c'), 
                     label='f$_c$')
            axt.legend(bbox_to_anchor=(0,0.95,1,.05), ncols=3, frameon=False,
                       mode='expand', loc='lower left')
            axt.set_ylim(-0.01, 1.05)
            axt.set_ylabel('Occupation \n probability')
            axt.set_xlabel('Energy (eV)')
            axt.xaxis.set_minor_locator(AutoMinorLocator())
            axt.grid(visible=True, which='major', axis='x')
            axt.grid(visible=True, which='minor', axis='x', c='0.85')
            text_t = ('E$_{fc}$ = '+ f'{self.Efc.value:.4f} eV \n'
                      'E$_{fv}$ = '+ f'{self.Efv.value:.4f} eV \n'
                      f'T = {self.T.value:.0f} K'
                      )
            axt.text(0.7, 0.3, text_t, ha='left', va='center', size=9,
                     transform=axt.transAxes)
            
            axb.plot(x, self.abs_coeff_tail_occupation(x), c='dodgerblue')
            axb.set_ylabel(r'$\alpha = \alpha_0(f_v-f_c)$ [cm$^{-1}$]')
            axb.set_xlabel('Energy (eV)')
            axb.xaxis.set_minor_locator(AutoMinorLocator())
            axb.grid(visible=True, which='major', axis='x')
            axb.grid(visible=True, which='minor', axis='x', c='0.85')
            axbr.plot(x, self.absorption(x), c='orangered')
            axbr.set_ylabel(r'A = (1-R)(1-$e^{-\alpha d}$)')
            text_b = ('E$_g$ = '+ f'{self.Eg.value:.4f} eV \n'
                      '$\gamma$ = '+ f'{self.g.value:.4f} eV \n'
                      f'd = {self.d.value:.0f} nm'
                      )
            axb.text(0.7, 0.2, text_b, ha='left', va='center', size=9,
                     transform=axb.transAxes)
            fig.tight_layout()
            
#TODO Add functions to plot abs (all types in one graph) and occupation 

    
#     def plotComponents(self, x=None, d=0.075,
#                        savefig_path=None, savefig_name=None):
#         from matplotlib.ticker import AutoMinorLocator
#         from matplotlib.gridspec import GridSpec
#         from scipy.interpolate import interp1d
#         from scipy.optimize import brentq
#         from hyperspy.drawing.utils import plot_spectra
#         from pathlib import Path
        
#         if self.model:
#             x = self.model.axis.axis
#             mask = self.model._channel_switches
#             model_x = x[mask]
#             model_y = self.model.as_signal().data[mask]
#             org_signal = self.model.signal
#         else:
#             raise NotImplementedError('Implemented only with a model associated')
        
#         label = ['Original data', 'Fit']
#         rows = []
#         cols = ['value (std)', 'free']
#         texts = []
#         colors = []
#         for param in self.parameters:
#             print(param.name)
#             rows.append(f'{param.name}  ({param.units})')
#             value = str(param.value)[:6] + ' (' + str(param.std)[:6] + ')'
#             if param.free:
#                 bmin = f'{param.bmin:.4f}' if param.bmin else None
#                 bmax = f'{param.bmax:.4f}' if param.bmax else None
#                 lim = [bmin, bmax]
#                 colors.append('azure')
#             else:
#                 colors.append('white')
#                 lim = param.free
#             texts.append([value,lim])
        
#         max_position = x[np.argmax(model_y)]
#         y_label = ('Normalised Intensity' if org_signal.metadata.Signal.scaled 
#                    else org_signal.metadata.Signal.quantity)
#         #model interpolation for root-findings
#         _f = interp1d(model_x, model_y-0.5)
#         intervals = [model_x[:np.argmax(model_y)],
#                      model_x[np.argmax(model_y):]
#                      ]
#         mins = []
#         #Find roots of the function for the fwhm
#         for interval in intervals:
#             zeros = interval[np.argmin(np.abs(_f(interval)))]
#             d1 = [zeros*(1 - d), zeros*(1+d)]
#             mins.append(brentq(_f, *d1))
# #TODO -> create a method of GerenalizedPlanck to calculate fwhm 
#         self.fwhm = abs(mins[0]-mins[1])
        
#         fig = plt.figure(figsize=(12,9))
#         gs = GridSpec(3, 3, figure=fig)
#         axtl = fig.add_subplot(gs[:,0])
#         axtm = fig.add_subplot(gs[0,1])
#         axbm = fig.add_subplot(gs[1:,1])
#         axbmr = axbm.twinx()
#         axr = fig.add_subplot(gs[:,2])
    
#         #Plot original data + fit
        
#         plot_spectra([org_signal, self.model.as_signal()], 
#                              linestyle=['-', '--'], 
#                              color=['black', 'darkorange'],
#                              fig=fig,
#                              ax=axtl)
#         axtl.legend(label, loc='center left', bbox_to_anchor=(0.01,0.95), 
#                   ncol=1, frameon=False)
#         axtl.set_xlabel('Energy (eV)')
#         axtl.set_xlim(max_position-5*self.fwhm, max_position+5*self.fwhm)
#         axtl.set_ylim(1e-3, 1.1)
#         axtl.set_ylabel(y_label)
#         axtl.set_yscale('log')
#         axtl.axvline(x=max_position, ls=':', c='0.7', lw=1)
#         axtl.axvline(x=self.Eg.value, ls=':', c='dodgerblue', lw=1)
#         axtl.annotate('', xy=(mins[0],0.5), xytext=(mins[1], 0.5),
#                       arrowprops=dict(arrowstyle='<->',
#                                    color='black',
#                                    ls=':'))
#         axtl.annotate(f'FWHM = {self.fwhm:.4f} eV',
#                       xy=(mins[1], 0.5), xycoords='data',
#                       xytext=(20,20), textcoords='offset points',
#                       size=10, va='bottom', ha='left',
#                       backgroundcolor='w',
#                       arrowprops=dict(arrowstyle="-|>",
#                                       connectionstyle="arc3,rad=-0.2",
#                                       fc="w"))
#         axtl.annotate( 'E$_{peak}$ =' +  f'{max_position:.4f} eV',
#                       xy=(max_position, 1e-1), xycoords='data',
#                       xytext=(50,30), textcoords='offset points',
#                       size=10, va='bottom', ha='left',
#                       backgroundcolor='w',
#                       arrowprops=dict(arrowstyle="-|>",
#                                       connectionstyle="arc3,rad=-0.2",
#                                       fc="w"))
#         axtl.annotate(f'E$_g$ = {self.Eg.value:.4f} eV',
#                       xy=(self.Eg.value, 2e-2), xycoords='data',
#                       xytext=(-50,10), textcoords='offset points',
#                       size=10, va='bottom', ha='right',
#                       backgroundcolor='w',
#                       arrowprops=dict(arrowstyle="-|>",
#                                       connectionstyle="arc3,rad=-0.2",
#                                       fc="w"))
        
#         #Plot table with all parameters
#         axr.set_xticks([])
#         axr.set_yticks([])
#         for spine in axr.spines:
#             axr.spines[spine].set_visible(False)

#         axr.table(cellText=texts,
#                            cellLoc='center',
#                            rowLabels=rows,
#                            colLabels=cols,
#                            loc='center left',
#                            rowColours=colors,
#                            fontsize=14,
#                            colWidths=[0.4,0.5]
#                           )
#         title = org_signal.metadata.General.title
#         fig.suptitle(title)
        
#         #Plot occupation probability, absorption coefficient and absorption
#         xc = x[x.size//2]
#         ycbl = self.abs_coeff_tail_occupation(xc)
#         ycbr = self.absorption(xc)
#         axbm.annotate("",xy=(np.min(x), ycbl), xycoords='data',
#                      xytext=(xc*0.99, ycbl), textcoords='data',
#                      arrowprops=dict(arrowstyle="->", 
#                                      connectionstyle="arc3",
#                                      color='dodgerblue',
#                                      ls='--')
#                      )
#         axbmr.annotate("",xy=(np.max(x), ycbr), xycoords='data',
#                      xytext=(xc*1.01, ycbr), textcoords='data',
#                      arrowprops=dict(arrowstyle="->", 
#                                      connectionstyle="arc3",
#                                      color='orangered',
#                                      ls='--')
#                      )
#         df = (self.fermi_distribution_vc(x, band='v') 
#               -self.fermi_distribution_vc(x, band='c')
#               )
#         axtm.plot(x, df, label='f$_v$ - f$_c$')
#         axtm.plot(x, self.fermi_distribution_vc(x, band='v'), 
#                  label='f$_v$')
#         axtm.plot(x, self.fermi_distribution_vc(x, band='c'), 
#                  label='f$_c$')
#         axtm.legend(bbox_to_anchor=(0,0.95,1,.05), ncols=3, frameon=False,
#                    mode='expand', loc='lower left')
#         axtm.set_ylim(-0.01, 1.05)
#         axtm.set_ylabel('Occupation \n probability')
#         axtm.set_xlabel('Energy (eV)')
#         axtm.xaxis.set_minor_locator(AutoMinorLocator())
#         axtm.grid(visible=True, which='major', axis='x')
#         axtm.grid(visible=True, which='minor', axis='x', c='0.85')
#         text_t = ('E$_{fc}$ = '+ f'{self.Efc.value:.4f} eV \n'
#                   'E$_{fv}$ = '+ f'{self.Efv.value:.4f} eV \n'
#                   f'T = {self.T.value:.0f} K'
#                   )
#         axtm.text(0.7, 0.3, text_t, ha='left', va='center', size=9,
#                  transform=axtm.transAxes)
        
#         axbm.plot(x, self.abs_coeff_tail_occupation(x), c='dodgerblue')
#         axbm.set_ylabel(r'$\alpha = \alpha_0(f_v-f_c)$ [cm$^{-1}$]')
#         axbm.set_xlabel('Energy (eV)')
#         axbm.xaxis.set_minor_locator(AutoMinorLocator())
#         axbm.grid(visible=True, which='major', axis='x')
#         axbm.grid(visible=True, which='minor', axis='x', c='0.85')
#         axbmr.plot(x, self.absorption(x), c='orangered')
#         axbmr.set_ylabel(r'A = (1-R)(1-$e^{-\alpha d}$)')
#         text_b = ('E$_g$ = '+ f'{self.Eg.value:.4f} eV \n'
#                   '$\gamma$ = '+ f'{self.g.value:.4f} eV \n'
#                   f'd = {self.d.value:.0f} nm'
#                   )
#         axbm.text(0.7, 0.2, text_b, ha='left', va='center', size=9,
#                  transform=axbm.transAxes)
        
#         fig.tight_layout()
#         if  savefig_path:
#             if not savefig_name:
#                 savefig_name = title + '_fit.svg'
#             else:
#                 if Path(savefig_name).suffix != '.svg':
#                     raise ValueError('Only exporting in SVG format: please check filename')
                
#             fig.savefig(savefig_path / savefig_name, format='svg')