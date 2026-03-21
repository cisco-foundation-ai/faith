# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.algo.hash import dict_sha256
from faith._internal.iter.transform import IsoMapping
from faith._types.record.sample import SampleRecord


class HashModelDataTransform(IsoMapping[SampleRecord]):
    """Transform that computes and stores the model_data_hash on each record's metadata."""

    def _map_fn(self, element: SampleRecord) -> SampleRecord:
        element.metadata.model_data_hash = dict_sha256(element.model_data.to_dict())
        return element
