# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.iter.common import GetAttrTransform
from faith.record_pipelines.hashing import HashModelDataTransform
from tests.benchmark.categories.fake_record_maker import make_fake_record


def test_hash_model_data_transform() -> None:
    assert not list([] >> HashModelDataTransform())

    assert list(
        [make_fake_record()]
        >> HashModelDataTransform()
        >> GetAttrTransform("metadata")
        >> GetAttrTransform("model_data_hash")
    ) == [
        "ee20aaa7dcb57841a6e08461c782f2ff5b62a5d5fd5ae119f274119dfcb30030",
    ]
