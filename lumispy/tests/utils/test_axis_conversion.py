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

from numpy import arange, ones
from numpy.testing import assert_allclose
from pytest import raises, mark, warns

from hyperspy.axes import DataAxis, UniformDataAxis
from lumispy.signals import LumiSpectrum, LumiTransientSpectrum
from lumispy.utils.axes import (
    nm2eV,
    eV2nm,
    axis2eV,
    data2eV,
    var2eV,
    nm2invcm,
    invcm2nm,
    axis2invcm,
    data2invcm,
    var2invcm,
    solve_grating_equation,
)


def test_nm2eV():
    wl = arange(300, 400, 90)
    en = nm2eV(wl)
    assert_allclose(en[0], 4.13160202)
    assert_allclose(en[-1], 3.17818160)


def test_eV2nm():
    en = arange(1, 2, 0.8)
    wl = eV2nm(en)
    assert_allclose(wl[0], 1239.50284)
    assert_allclose(wl[-1], 688.611116)


def test_axis2eV():
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    axis2 = DataAxis(axis=arange(0.2, 0.400, 0.01), units="µm")
    axis3 = DataAxis(axis=arange(1, 2, 0.1), units="eV")
    evaxis, factor = axis2eV(axis)
    evaxis2, factor2 = axis2eV(axis2)
    with raises(AttributeError, match="Signal unit is already eV."):
        axis2eV(axis3)
    assert factor == 1e6
    assert factor2 == 1e3
    assert evaxis.name == "Energy"
    assert evaxis.units == "eV"
    assert not evaxis.navigate
    assert evaxis2.units == "eV"
    assert evaxis2.size == 20
    assert_allclose(evaxis.axis[0], evaxis2.axis[0])
    assert_allclose(evaxis.axis[-1], evaxis2.axis[-1])
    assert_allclose(evaxis.axis[0], 3.1781816)


def test_data2eV():
    data = 100 * ones(20)
    ax0 = DataAxis(axis=arange(200, 400, 10), units="nm")
    evaxis, factor = axis2eV(ax0)
    evdata = data2eV(data, factor, evaxis.axis, ax0)
    assert_allclose(evdata[0], 12.271168)
    ax0 = DataAxis(axis=arange(0.2, 0.4, 0.01), units="µm")
    evaxis, factor = axis2eV(ax0)
    evdata = data2eV(data, factor, evaxis.axis, ax0)
    assert_allclose(evdata[0], 12.271168e-3)


def test_var2eV():
    data = 100 * ones(20)
    ax0 = DataAxis(axis=arange(200, 400, 10), units="nm")
    evaxis, factor = axis2eV(ax0)
    evvar = var2eV(data, factor, evaxis.axis, ax0)
    assert_allclose(evvar[0], 1.5058156)


@mark.parametrize(("jacobian"), (True, False))
@mark.parametrize(("variance"), (True, False, "constant"))
def test_to_eV(jacobian, variance):
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    data = ones(20)
    data[0] += 1
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    if variance:
        if variance == "constant":
            S1.set_noise_variance(1.0)
        else:
            S1.estimate_poissonian_noise_variance()
    S2 = S1.to_eV(inplace=False, jacobian=jacobian)
    S1.axes_manager[0].units = "µm"
    S1.axes_manager[0].axis = axis.axis / 1000
    if jacobian:
        S1.data *= 1000
    S1.to_eV(jacobian=jacobian)
    assert S1.axes_manager[0].units == "eV"
    assert S2.axes_manager[0].name == "Energy"
    assert S2.axes_manager[0].size == 20
    assert S1.axes_manager[0].axis[0] == S2.axes_manager[0].axis[0]
    assert_allclose(S1.data, S2.data, 5e-4)
    if not jacobian:
        S1.data *= 10  # check that deepcopy worked
        assert S2.data[-1] == 2
        assert S1.data[-1] == 20
    else:
        S1.data *= 10
        assert_allclose(S2.data[-1], 0.06454526)
    nav = UniformDataAxis(size=4)
    # navigation dimension 1
    L1 = LumiSpectrum(
        ones((4, 20)), axes=[nav.get_axis_dictionary(), axis.get_axis_dictionary()]
    )
    if variance:
        if variance == "constant":
            L1.set_noise_variance(1.0)
        else:
            L1.estimate_poissonian_noise_variance()
    L2 = L1.to_eV(inplace=False, jacobian=jacobian)
    L1.to_eV(jacobian=jacobian)
    assert L1.axes_manager.signal_axes[0].units == "eV"
    assert L2.axes_manager.signal_axes[0].name == "Energy"
    assert L2.axes_manager.signal_axes[0].size == 20
    assert (
        L1.axes_manager.signal_axes[0].axis[0] == L2.axes_manager.signal_axes[0].axis[0]
    )
    assert_allclose(L1.data, L2.data, 5e-4)
    # navigation dimension 2
    M1 = LumiSpectrum(
        ones((4, 4, 20)),
        axes=[
            nav.get_axis_dictionary(),
            nav.get_axis_dictionary(),
            axis.get_axis_dictionary(),
        ],
    )
    if variance:
        if variance == "constant":
            M1.set_noise_variance(1.0)
        else:
            M1.estimate_poissonian_noise_variance()
    M2 = M1.to_eV(inplace=False, jacobian=jacobian)
    M1.to_eV(jacobian=jacobian)
    assert M1.axes_manager.signal_axes[0].units == "eV"
    assert M2.axes_manager.signal_axes[0].name == "Energy"
    assert M2.axes_manager.signal_axes[0].size == 20
    assert (
        M1.axes_manager.signal_axes[0].axis[0] == M2.axes_manager.signal_axes[0].axis[0]
    )
    assert_allclose(M1.data, M2.data, 5e-4)
    if variance:
        if variance != "constant":
            assert (
                S1.metadata.Signal.Noise_properties.variance.axes_manager[-1].axis[0]
                == S1.axes_manager[-1].axis[0]
            )
            assert (
                S2.metadata.Signal.Noise_properties.variance.axes_manager[-1].axis[0]
                == S2.axes_manager[-1].axis[0]
            )
        assert (
            S1.metadata.Signal.Noise_properties.variance
            == S2.metadata.Signal.Noise_properties.variance
        )
        assert (
            L1.metadata.Signal.Noise_properties.variance
            == L2.metadata.Signal.Noise_properties.variance
        )
        assert (
            M1.metadata.Signal.Noise_properties.variance
            == M2.metadata.Signal.Noise_properties.variance
        )
    else:
        assert S1.metadata.has_item("Signal.Noise_properties.variance") == False


def test_eV_slicing():
    S = LumiSpectrum(arange(100), axes=[{"axis": arange(100) + 300}])
    S.to_eV(inplace=True)
    S.isig[3.251:4.052]


def test_to_eV_2D():
    axis1 = UniformDataAxis(size=10, offset=200, scale=10, units="nm")
    axis2 = UniformDataAxis(size=8, offset=0, scale=0.1, units="ns")
    data = arange(80).reshape(8, 10)
    S1 = LumiTransientSpectrum(data, axes=[axis2, axis1])
    S2 = S1.to_eV(inplace=True, jacobian=False)
    assert S1.axes_manager[0].units == "eV"
    assert S1.axes_manager[0].axis[-1] == nm2eV(axis1.axis[0])
    assert S1.data[0, 0] == data[0, -1]


@mark.parametrize(("jacobian"), (True, False))
def test_reset_variance_linear_model_eV(jacobian):
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    data = ones(20)
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    S1.metadata.set_item("Signal.Noise_properties.Variance_linear_model.gain_factor", 2)
    S1.metadata.set_item("Signal.Noise_properties.Variance_linear_model.gain_offset", 1)
    S1.metadata.set_item(
        "Signal.Noise_properties.Variance_linear_model.correlation_factor", 2
    )
    S1.estimate_poissonian_noise_variance()
    S2 = S1.to_eV(inplace=False, jacobian=jacobian)
    if jacobian:
        with warns(UserWarning, match="Following"):
            S1.to_eV(inplace=True, jacobian=jacobian)
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.gain_factor == 1
        )
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.gain_offset == 0
        )
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.correlation_factor
            == 1
        )
        assert (
            S2.metadata.has_item("Signal.Noise_properties.Variance_linear_model")
            == False
        )
    else:
        S1.to_invcm(inplace=True, jacobian=jacobian)
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.gain_factor == 2
        )
        assert (
            S2.metadata.Signal.Noise_properties.Variance_linear_model.gain_factor == 2
        )


def test_nm2invcm():
    wl = arange(300, 410, 100)
    invcm = nm2invcm(wl)
    assert_allclose(invcm[0], 33333.3333)
    assert_allclose(invcm[-1], 25000)


def test_invcm2nm():
    invcm = arange(10000, 20000, 6000)
    wl = invcm2nm(invcm)
    assert_allclose(wl[0], 1000)
    assert_allclose(wl[-1], 625)


def test_axis2invcm():
    axis = UniformDataAxis(size=21, offset=200, scale=10)
    axis2 = DataAxis(axis=arange(0.2, 0.410, 0.01), units="µm")
    axis3 = DataAxis(axis=arange(1, 2, 0.1), units=r"cm$^{-1}$")
    invcmaxis, factor = axis2invcm(axis)
    invcmaxis2, factor2 = axis2invcm(axis2)
    with raises(AttributeError, match="Signal unit is already"):
        axis2invcm(axis3)
    assert factor == 1e7
    assert factor2 == 1e4
    assert invcmaxis.name == "Wavenumber"
    assert invcmaxis.units == r"cm$^{-1}$"
    assert not invcmaxis.navigate
    assert invcmaxis2.units == r"cm$^{-1}$"
    assert invcmaxis2.size == 21
    assert_allclose(invcmaxis.axis[0], invcmaxis2.axis[0])
    assert_allclose(invcmaxis.axis[-1], invcmaxis2.axis[-1])
    assert_allclose(invcmaxis.axis[0], 25000)


def test_data2invcm():
    data = 100 * ones(20)
    factor = 1e7
    ax0 = arange(200, 400, 10)
    invcmaxis = nm2invcm(ax0)
    invcmdata = data2invcm(data, factor, invcmaxis)
    assert_allclose(invcmdata[-1], 1.521)


def test_var2invcm():
    data = 100 * ones(20)
    factor = 1e7
    ax0 = arange(200, 400, 10)
    invcmaxis = nm2invcm(ax0)
    invcmdata = var2invcm(data, factor, invcmaxis)
    assert_allclose(invcmdata[-1], 0.02313441)


@mark.parametrize(("jacobian"), (True, False))
@mark.parametrize(("variance"), (True, False, "constant"))
def test_to_invcm(jacobian, variance):
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    data = ones(20)
    data[0] += 1
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    if variance:
        if variance == "constant":
            S1.set_noise_variance(1.0)
        else:
            S1.estimate_poissonian_noise_variance()
    S2 = S1.to_invcm(inplace=False, jacobian=jacobian)
    S1.axes_manager[0].units = "µm"
    S1.axes_manager[0].axis = axis.axis / 1000
    if jacobian:
        S1.data *= 1000
    S1.to_invcm(jacobian=jacobian)
    assert S1.axes_manager[0].units == r"cm$^{-1}$"
    assert S2.axes_manager[0].name == "Wavenumber"
    assert S2.axes_manager[0].size == 20
    assert S1.axes_manager[0].axis[0] == S2.axes_manager[0].axis[0]
    assert_allclose(S1.data, S2.data, 5e-4)
    if not jacobian:
        S1.data *= 10  # check that deepcopy worked
        assert S2.data[-1] == 2
        assert S1.data[-1] == 20
    else:
        S1.data *= 10
        assert S2.data[-1] == 0.008
    nav = UniformDataAxis(size=4)
    # navigation dimension 1
    L1 = LumiSpectrum(
        ones((4, 20)), axes=[nav.get_axis_dictionary(), axis.get_axis_dictionary()]
    )
    if variance:
        if variance == "constant":
            L1.set_noise_variance(1.0)
        else:
            L1.estimate_poissonian_noise_variance()
    L2 = L1.to_invcm(inplace=False, jacobian=jacobian)
    L1.to_invcm(jacobian=jacobian)
    assert L1.axes_manager.signal_axes[0].units == r"cm$^{-1}$"
    assert L2.axes_manager.signal_axes[0].name == "Wavenumber"
    assert L2.axes_manager.signal_axes[0].size == 20
    assert (
        L1.axes_manager.signal_axes[0].axis[0] == L2.axes_manager.signal_axes[0].axis[0]
    )
    assert_allclose(L1.data, L2.data, 5e-4)
    # navigation dimension 2
    M1 = LumiSpectrum(
        ones((4, 4, 20)),
        axes=[
            nav.get_axis_dictionary(),
            nav.get_axis_dictionary(),
            axis.get_axis_dictionary(),
        ],
    )
    if variance:
        if variance == "constant":
            M1.set_noise_variance(1.0)
        else:
            M1.estimate_poissonian_noise_variance()
    M2 = M1.to_invcm(inplace=False, jacobian=jacobian)
    M1.to_invcm(jacobian=jacobian)
    assert M1.axes_manager.signal_axes[0].units == r"cm$^{-1}$"
    assert M2.axes_manager.signal_axes[0].name == "Wavenumber"
    assert M2.axes_manager.signal_axes[0].size == 20
    assert (
        M1.axes_manager.signal_axes[0].axis[0] == M2.axes_manager.signal_axes[0].axis[0]
    )
    assert_allclose(M1.data, M2.data, 5e-4)
    if variance:
        if variance != "constant":
            assert (
                S1.metadata.Signal.Noise_properties.variance.axes_manager[-1].axis[0]
                == S1.axes_manager[-1].axis[0]
            )
            assert (
                S2.metadata.Signal.Noise_properties.variance.axes_manager[-1].axis[0]
                == S2.axes_manager[-1].axis[0]
            )
        assert (
            S1.metadata.Signal.Noise_properties.variance
            == S2.metadata.Signal.Noise_properties.variance
        )
        assert (
            L1.metadata.Signal.Noise_properties.variance
            == L2.metadata.Signal.Noise_properties.variance
        )
        assert (
            M1.metadata.Signal.Noise_properties.variance
            == M2.metadata.Signal.Noise_properties.variance
        )
    else:
        assert S1.metadata.has_item("Signal.Noise_properties.variance") == False


def test_invcm_slicing():
    S = LumiSpectrum(arange(100), axes=[{"axis": arange(100) + 300}])
    S.to_invcm(inplace=True)
    S.isig[26000.0:30000.0]


@mark.parametrize(("jacobian"), (True, False))
def test_reset_variance_linear_model_invcm(jacobian):
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    data = ones(20)
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    S1.metadata.set_item("Signal.Noise_properties.Variance_linear_model.gain_factor", 2)
    S1.metadata.set_item("Signal.Noise_properties.Variance_linear_model.gain_offset", 1)
    S1.metadata.set_item(
        "Signal.Noise_properties.Variance_linear_model.correlation_factor", 2
    )
    S1.estimate_poissonian_noise_variance()
    S2 = S1.to_invcm(inplace=False, jacobian=jacobian)
    if jacobian:
        with warns(UserWarning, match="Following"):
            S1.to_invcm(inplace=True, jacobian=jacobian)
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.gain_factor == 1
        )
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.gain_offset == 0
        )
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.correlation_factor
            == 1
        )
        assert (
            S2.metadata.has_item("Signal.Noise_properties.Variance_linear_model")
            == False
        )
    else:
        S1.to_invcm(inplace=True, jacobian=jacobian)
        assert (
            S1.metadata.Signal.Noise_properties.Variance_linear_model.gain_factor == 2
        )
        assert (
            S2.metadata.Signal.Noise_properties.Variance_linear_model.gain_factor == 2
        )


@mark.parametrize(("jacobian"), (True, False))
@mark.parametrize(("variance"), (True, False, "constant"))
def test_to_invcm_relative(jacobian, variance):
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    data = ones(20)
    data[0] += 1
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    if variance:
        if variance == "constant":
            S1.set_noise_variance(1.0)
        else:
            S1.estimate_poissonian_noise_variance()
    S2 = S1.to_invcm_relative(laser=244, inplace=False, jacobian=jacobian)
    S1.axes_manager[0].units = "µm"
    S1.axes_manager[0].axis = axis.axis / 1000
    if jacobian:
        S1.data *= 1000
    S1.to_invcm_relative(laser=0.244, jacobian=jacobian)
    assert S1.axes_manager[0].units == r"cm$^{-1}$"
    assert S2.axes_manager[0].name == "Raman Shift"
    assert S2.axes_manager[0].size == 20
    assert S1.axes_manager[0].axis[0] == S2.axes_manager[0].axis[0]
    assert_allclose(S1.data, S2.data, 5e-4)
    nav = UniformDataAxis(size=4)
    # navigation dimension 1
    L1 = LumiSpectrum(
        ones((4, 20)), axes=[nav.get_axis_dictionary(), axis.get_axis_dictionary()]
    )
    if variance:
        if variance == "constant":
            L1.set_noise_variance(1.0)
        else:
            L1.estimate_poissonian_noise_variance()
    L2 = L1.to_invcm_relative(laser=244, inplace=False, jacobian=jacobian)
    L1.to_invcm_relative(laser=244, jacobian=jacobian)
    assert L1.axes_manager.signal_axes[0].units == r"cm$^{-1}$"
    assert L2.axes_manager.signal_axes[0].name == "Raman Shift"
    assert L2.axes_manager.signal_axes[0].size == 20
    assert (
        L1.axes_manager.signal_axes[0].axis[0] == L2.axes_manager.signal_axes[0].axis[0]
    )
    assert_allclose(L1.data, L2.data, 5e-4)
    # navigation dimension 2
    M1 = LumiSpectrum(
        ones((4, 4, 20)),
        axes=[
            nav.get_axis_dictionary(),
            nav.get_axis_dictionary(),
            axis.get_axis_dictionary(),
        ],
    )
    if variance:
        if variance == "constant":
            M1.set_noise_variance(1.0)
        else:
            M1.estimate_poissonian_noise_variance()
    M2 = M1.to_invcm_relative(laser=244, inplace=False, jacobian=jacobian)
    M1.to_invcm_relative(laser=244, jacobian=jacobian)
    assert M1.axes_manager.signal_axes[0].units == r"cm$^{-1}$"
    assert M2.axes_manager.signal_axes[0].name == "Raman Shift"
    assert M2.axes_manager.signal_axes[0].size == 20
    assert (
        M1.axes_manager.signal_axes[0].axis[0] == M2.axes_manager.signal_axes[0].axis[0]
    )
    assert_allclose(M1.data, M2.data, 5e-4)
    if variance:
        if variance != "constant":
            assert (
                S1.metadata.Signal.Noise_properties.variance.axes_manager[-1].axis[0]
                == S1.axes_manager[-1].axis[0]
            )
            assert (
                S2.metadata.Signal.Noise_properties.variance.axes_manager[-1].axis[0]
                == S2.axes_manager[-1].axis[0]
            )
        assert (
            S1.metadata.Signal.Noise_properties.variance
            == S2.metadata.Signal.Noise_properties.variance
        )
        assert (
            L1.metadata.Signal.Noise_properties.variance
            == L2.metadata.Signal.Noise_properties.variance
        )
        assert (
            M1.metadata.Signal.Noise_properties.variance
            == M2.metadata.Signal.Noise_properties.variance
        )
    else:
        assert S1.metadata.has_item("Signal.Noise_properties.variance") == False


@mark.parametrize(("jacobian"), (True, False))
def test_to_raman_shift(jacobian):
    axis = UniformDataAxis(size=20, offset=200, scale=10)
    data = ones(20)
    data[0] += 1
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    S2 = S1.to_raman_shift(laser=244, inplace=False, jacobian=jacobian)
    S1.axes_manager[0].units = "µm"
    S1.axes_manager[0].axis = axis.axis / 1000
    if jacobian:
        S1.data *= 1000
    S1.to_raman_shift(laser=0.244, jacobian=jacobian)
    assert S1.axes_manager[0].units == r"cm$^{-1}$"
    assert S2.axes_manager[0].name == "Raman Shift"
    assert S2.axes_manager[0].size == 20
    assert S1.axes_manager[0].axis[0] == S2.axes_manager[0].axis[0]
    assert_allclose(S1.data, S2.data, 5e-4)
    if not jacobian:
        S1.data *= 10  # check that deepcopy worked
        assert S2.data[0] == 2
        assert S1.data[0] == 20
    else:
        S1.data *= 10
        assert S2.data[0] == 0.008


def test_to_raman_shift_laser():
    axis = UniformDataAxis(size=20, offset=200, scale=10, units="nm")
    data = ones(20)
    S1 = LumiSpectrum(data, axes=(axis.get_axis_dictionary(),))
    with raises(AttributeError, match="Laser wavelength"):
        S1.to_raman_shift()
    with raises(AttributeError, match="Laser wavelength units"):
        S1.to_raman_shift(laser=0.244)
    S1.metadata.set_item("Acquisition_instrument.Laser.wavelength", 244)
    S2 = S1.to_raman_shift(inplace=False)
    S1.axes_manager[0].units = "µm"
    S1.axes_manager[0].axis = axis.axis / 1000
    with raises(AttributeError, match="Laser wavelength units"):
        S1.to_raman_shift(laser=244)
    S1.metadata.set_item("Acquisition_instrument.Laser.wavelength", 0.244)
    S1.to_raman_shift()
    assert S1.axes_manager[0].units == r"cm$^{-1}$"
    assert S2.axes_manager[0].name == "Raman Shift"
    assert S2.axes_manager[0].size == 20
    assert S1.axes_manager[0].axis[0] == S2.axes_manager[0].axis[0]
    assert_allclose(S1.data, S2.data, 5e-4)


def test_ramanshift_slicing():
    S = LumiSpectrum(arange(100), axes=[{"axis": arange(100) + 300}])
    S.to_raman_shift(inplace=True, laser=295)
    S.isig[3000.0:8000.0]


def test_solve_grating_equation():
    with warns(SyntaxWarning, match="(not in pixel units)"):
        axis = UniformDataAxis(size=20, offset=200, scale=10, units="nm")
        solve_grating_equation(axis, 3, -20, 300, 25, 600, 150)

    axis1 = UniformDataAxis(
        size=10,
        offset=200,
        scale=10,
    )
    axis2 = UniformDataAxis(size=10, offset=200, scale=10, units="px")

    nm_axis1 = solve_grating_equation(axis1, 3, -20, 300, 25, 600, 150)
    with warns(UserWarning, match="(range exceeds)"):
        nm_axis2 = solve_grating_equation(axis2, 1, 1, 1, 1, 1, 1)

    assert nm_axis1.name == "Wavelength"
    assert nm_axis1.units == "nm"
    assert nm_axis2.name == "Wavelength"
    assert nm_axis2.units == "nm"

    assert_allclose(nm_axis1.axis[0], 368.614, atol=0.1)
    assert_allclose(nm_axis1.axis[-1], 768.249, atol=0.1)
    assert_allclose(nm_axis2.axis[0], 411559.839, atol=0.1)
    assert_allclose(nm_axis2.axis[-1], 321785.967, atol=0.1)
