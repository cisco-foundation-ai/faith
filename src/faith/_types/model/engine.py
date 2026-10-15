# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""ModelEngine enum and EngineParams configuration."""

import re
from dataclasses import dataclass, field
from enum import auto
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIEnum


class ModelEngine(CIEnum):
    """Enum representing different model engine backends."""

    OPENAI = auto()
    OPENROUTER = auto()
    VLLM = auto()
    SAGEMAKER = auto()


REDACTED = "<redacted>"

# Credential words wherever they appear: none has an innocuous use in a kwarg name.
_SECRET_WORDS = frozenset({"apikey", "credential", "credentials", "password", "secret"})
# Credential words only as the last word, since `token` is also how a backend spells
# a generation bound: `hf_token` is a credential, `max_tokens` and `stop_token_ids`
# are not.
_SECRET_TAILS = frozenset({"key", "token"})


def is_secret_name(name: str) -> bool:
    """Whether `name` names a credential, matching whole words rather than substrings."""
    words = re.split(r"[^a-z0-9]+", name.lower())
    return bool(_SECRET_WORDS & set(words)) or words[-1] in _SECRET_TAILS


def _redact_secrets(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replace the value of every credential-named kwarg with a sentinel.

    The key is kept so the record still shows a credential was configured. Matching
    is by name, so a credential named outside `_SECRET_WORDS`/`_SECRET_TAILS` is
    still written in the clear.
    """
    return {
        key: REDACTED if is_secret_name(key) else value for key, value in kwargs.items()
    }


@dataclass(frozen=True)
class EngineParams(DataClassJsonMixin):
    """Parameters for the model backend engine."""

    engine_type: ModelEngine = field(metadata=config(decoder=ModelEngine, encoder=str))
    num_gpus: int = 1
    context_length: int = 3500
    kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata=config(exclude=lambda x: not x, encoder=_redact_secrets),
    )
