# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

try:
    from ._version import __version__
except ImportError:
    # Fallback for when the package is not installed or setuptools_scm hasn't run.
    __version__ = "0.0.0+unknown"

__all__: list[str] = []
