# -*- coding: utf-8 -*-
"""Workaround for a hung Windows WMI service.

On this machine the WMI service intermittently stops answering, which makes
platform.win32_ver() (called during numpy.testing/scipy imports) block forever.
Setting platform._wmi = None forces the documented OSError fallback path
(winreg / environment variables). Import this module BEFORE numpy/scipy/torch.
"""

import platform

platform._wmi = None
