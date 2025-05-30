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

"""
Signal class for luminescence spectral data (1D).
-------------------------------------------------
"""

import numpy as np
from warnings import warn

from hyperspy.signals import Signal1D
from hyperspy._signals.lazy import LazySignal
from traits.api import Undefined

from lumispy.signals.common_luminescence import CommonLumi
from lumispy import to_array, savetxt
from lumispy.utils import solve_grating_equation
from lumispy.utils.axes import GRATING_EQUATION_DOCSTRING_PARAMETERS
from lumispy.utils.signals import com
from lumispy.utils.io import (
    SAVETXT_DOCSTRING,
    SAVETXT_PARAMETERS,
    TOARRAY_DOCSTRING,
    TOARRAY_PARAMETERS,
)


class LumiSpectrum(Signal1D, CommonLumi):
    """**General 1D luminescence signal class.**"""

    _signal_dimension = 1
    _signal_type = "Luminescence"

    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_background_from_file(self, background=None, inplace=False, **kwargs):
        """Subtract the background to the signal in all navigation axes. If no
        background file is passed as argument, the `remove_background()` from
        HyperSpy is called with the GUI.

        Parameters
        ----------
        background : array shape (2, n) or Signal1D
            An array containing the background x-axis and the intensity values
            [[xs],[ys]] or a Signal1D object. If the x-axis values do not match
            the signal_axes, then interpolation is done before subtraction. If
            only the intensity values are provided, [ys], the functions assumes
            no interpolation needed.
        inplace : boolean
            If False, it returns a new object with the transformation. If True,
            the original object is transformed, returning no object.

        Returns
        -------
        signal : LumiSpectrum
            A background subtracted signal.

        Notes
        -----
        This function does not work with non-uniform axes.
        """
        warn(
            "The use of `remove_background_from_file` is deprecated and will "
            "be removed in LumiSpy 1.0. Please use `remove_background_signal` "
            "from the Signal1D class.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self.metadata.Signal, "background_subtracted"):
            if self.metadata.Signal.background_subtracted is True:
                raise RecursionError(
                    "You have already removed background once. If you need to "
                    "remove it again, set the "
                    "s.metadata.Signal.background_subtracted to False."
                )
        elif background is None:  # pragma: no cover
            warn(
                "Using the Hyperspy specfic `remove_background` function. "
                "Use `s.remove_background()` instead.",
                category=SyntaxWarning,
            )
            self.remove_background(**kwargs)
        else:
            signal_x = self.axes_manager.signal_axes[0].axis

            if hasattr(background, "axes_manager"):  # Check if Hyperspy-like object
                x = background.axes_manager.signal_axes[0].axis
                y = background.data
                background = [x, y]

            background_xy = background

            if np.shape(background_xy)[0] == 1:
                bkg_x = signal_x
                bkg_y = background_xy[0]
            elif np.shape(background_xy)[0] == 2:
                try:
                    bkg_x = background_xy[0]
                    bkg_y = background_xy[1]
                except IndexError:  # pragma: no cover
                    raise AttributeError(
                        "Please provide a background file containing both the x and y axis."
                    )
            else:
                raise AttributeError(
                    "Please, provide a background of shape (2, n) or (n,)"
                )

            if not np.array_equal(bkg_x, signal_x):
                # Interpolation needed
                bkg_y = np.interp(signal_x, bkg_x, bkg_y)

            if not inplace:
                self_subtracted = self.map(
                    lambda s, bkg: s - bkg, bkg=bkg_y, inplace=False
                )
                self_subtracted.metadata.set_item("Signal.background_subtracted", True)
                self_subtracted.metadata.set_item("Signal.background", bkg_y)
                return self_subtracted
            else:
                self.metadata.set_item("Signal.background_subtracted", True)
                self.metadata.set_item("Signal.background", bkg_y)
                return self.map(lambda s, bkg: s - bkg, bkg=bkg_y, inplace=True)

    SAVETXT_EXAMPLE = """
    Examples
    --------
    >>> import lumispy as lum
    >>> import numpy as np
    ...
    >>> # Spectrum:
    >>> S = lum.signals.LumiSpectrum(np.arange(5))
    >>> S.savetxt('spectrum.txt')
    0.00000	0.00000
    1.00000	1.00000
    2.00000	2.00000
    3.00000	3.00000
    4.00000	4.00000
    ...
    >>> # Linescan:
    >>> L = lum.signals.LumiSpectrum(np.arange(25).reshape((5,5)))
    >>> L.savetxt('linescan.txt')
    0.00000	0.00000	1.00000	2.00000	3.00000	4.00000
    0.00000	0.00000	5.00000	10.00000	15.00000	20.00000
    1.00000	1.00000	6.00000	11.00000	16.00000	21.00000
    2.00000	2.00000	7.00000	12.00000	17.00000	22.00000
    3.00000	3.00000	8.00000	13.00000	18.00000	23.00000
    4.00000	4.00000	9.00000	14.00000	19.00000	24.00000
    """

    def savetxt(
        self,
        filename,
        fmt="%.5f",
        delimiter="\t",
        axes=True,
        transpose=False,
        **kwargs,
    ):
        """Writes luminescence spectrum object to simple text file.
        %s
        %s
        %s
        """
        savetxt(self, filename, fmt, delimiter, axes, transpose, **kwargs)

    savetxt.__doc__ %= (
        SAVETXT_DOCSTRING,
        SAVETXT_PARAMETERS,
        SAVETXT_EXAMPLE,
    )

    TOARRAY_EXAMPLE = """
    Notes
    -----
    The output of this function can be used to convert a signal object to a
    pandas dataframe, e.g. using `df = pd.Dataframe(S.to_array())`.

    Examples
    --------
    >>> import lumispy as lum
    >>> import numpy as np
    ...
    >>> # Spectrum:
    >>> S = lum.signals.LumiSpectrum(np.arange(5))
    >>> S.to_array()
    array([[0., 0.],
          [1., 1.],
          [2., 2.],
          [3., 3.],
          [4., 4.]])
    ...
    >>> # Linescan:
    >>> L = lum.signals.LumiSpectrum(np.arange(25).reshape((5,5)))
    >>> L.to_array()
    array([[ 0.,  0.,  1.,  2.,  3.,  4.],
          [ 0.,  0.,  1.,  2.,  3.,  4.],
          [ 1.,  5.,  6.,  7.,  8.,  9.],
          [ 2., 10., 11., 12., 13., 14.],
          [ 3., 15., 16., 17., 18., 19.],
          [ 4., 20., 21., 22., 23., 24.]])
    """

    def to_array(self, axes=True, transpose=False):
        """Returns luminescence spectrum object as numpy array (optionally
        including the axes).
            %s
            %s
        %s
        """
        return to_array(self, axes, transpose)

    to_array.__doc__ %= (
        TOARRAY_DOCSTRING,
        TOARRAY_PARAMETERS,
        TOARRAY_EXAMPLE,
    )

    def px_to_nm_grating_solver(
        self,
        gamma_deg,
        deviation_angle_deg,
        focal_length_mm,
        ccd_width_mm,
        grating_central_wavelength_nm,
        grating_density_gr_mm,
        inplace=False,
    ):
        """Converts signal axis of 1D signal (in pixels) to wavelength, solving
        the grating equation. See `lumispy.axes.solve_grating_equation` for
        more details.

        Parameters
        ----------
        %s
            inplace : bool
                If False, it returns a new object with the transformation. If
                True, the original object is transformed, returning no object.

        Returns
        -------
        signal : LumiSpectrum
            A signal with calibrated wavelength units.

        Examples
        --------
        >>> s = LumiSpectrum(np.ones(20),))
        >>> s.px_to_nm_grating_solver(*params, inplace=True)
        >>> s.axes_manager.signal_axes[0].units == 'nm'
        """

        nm_axis = solve_grating_equation(
            self.axes_manager.signal_axes[0],
            gamma_deg,
            deviation_angle_deg,
            focal_length_mm,
            ccd_width_mm,
            grating_central_wavelength_nm,
            grating_density_gr_mm,
        )

        if inplace:
            s = self
        else:
            s = self.deepcopy()

        s.axes_manager.set_axis(
            nm_axis,
            self.axes_manager.signal_axes[0].index_in_axes_manager,
        )

        return s

    px_to_nm_grating_solver.__doc__ %= GRATING_EQUATION_DOCSTRING_PARAMETERS.replace(
        "\n", "\n\t"
    )

    def centroid(self, signal_range=None, **kwargs):
        """
        Finds the centroid (center of mass) of a peak in the spectrum from
        the wavelength (or pixel number) and the intensity at each pixel
        value. It basically represents a "weighted average" of the peak.

        Notes
        -----
        This function only works for a single peak. If you have multiple
        peaks, slice the signal beforehand or use the signal_range parameter.

        TODO: Implement this function for multiple peaks (with the npeaks
        parameter) by finding the top 2 peaks from mean spectrum and then
        returning a signal with 2 com.

        Parameters
        ----------
        signal_range : tuple of ints or floats, optional
            A tuple representing the indices of the signal axis (start index,
            end index) where the peak is located. If the tuple contains int,
            it slices on index. If the tuple contains float, it slices on
            signal units (default HyperSpy s.inav[:] functionality).

        kwargs : dictionary
            For the scipy.interpolate.interp1d function.

        Returns
        -------
        signal : Signal2D, BaseSignal
            Signal object containing the center of mass for every pixel. Depending
            on the dimensionality the type is Signal2D or a BaseSignal (for single
            spectrum).
        """
        if signal_range:
            if type(signal_range) != tuple:
                raise TypeError(
                    "The `signal_range` parameter must be a tuple of length 2."
                )
            if len(signal_range) != 2:
                raise ValueError(
                    f"The `signal_range` parameter must be a tuple of length 2. "
                    "You passed a tuple of length {len(signal_range)}."
                )

            s = self.isig[signal_range[0] : signal_range[1]]

        else:
            s = self

        signal_axis = s.axes_manager.signal_axes[0]
        center_of_mass = s.map(com, signal_axis=signal_axis, inplace=False)

        # Transfer axes metadata to title
        center_of_mass.metadata.General.title = f"Centroid map"
        if signal_axis.name not in (Undefined, ""):
            center_of_mass.metadata.General.title += f" of {signal_axis.name}"
        if signal_axis.units not in (Undefined, ""):
            center_of_mass.metadata.General.title += f" ({signal_axis.units})"
        if s.metadata.General.title not in (Undefined, ""):
            center_of_mass.metadata.General.title += f" for {s.metadata.General.title}"
        if center_of_mass.axes_manager.navigation_size > 0:
            center_of_mass = center_of_mass.transpose()
        return center_of_mass


class LazyLumiSpectrum(LazySignal, LumiSpectrum):
    """**General lazy 1D luminescence signal class.**"""

    _lazy = True
