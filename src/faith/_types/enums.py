# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Case-insensitive enum types."""

from enum import Enum, StrEnum


class CIEnum(Enum):
    """Enum base that adds case-insensitive name-based lookup and lowercase __str__."""

    @classmethod
    def _missing_(cls, value: object) -> "CIEnum | None":
        if isinstance(value, str):
            for member in cls:
                if member.name.casefold() == value.casefold():
                    return member
        return None

    def __str__(self) -> str:
        return self.name.lower()


class CIStrEnum(StrEnum):
    """A StrEnum that supports case-insensitive value lookup."""

    @classmethod
    def _missing_(cls, value: object) -> "CIStrEnum | None":
        if isinstance(value, str):
            for member in cls:
                if member.value.casefold() == value.casefold():
                    return member
        return None
