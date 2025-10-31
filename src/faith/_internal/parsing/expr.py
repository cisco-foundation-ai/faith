# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides a wrapper `evaluate_expr` around simpleeval to evaluate expressions safely.

The wrapper `evaluate_expr` allows for the evaluation of expressions with a
limited set of builtin allowed functions and types, and has an adjustable maximum
comprehension length to allow for evaluations on large datasets.
"""

from typing import Any

import simpleeval

_ALLOWED_CALLABLES = {
    "abs": abs,
    "aiter": aiter,
    "all": all,
    "anext": anext,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def evaluate_expr(
    expr: str,
    names: dict[str, Any] | None = None,
    max_comprehension_length: int = simpleeval.MAX_COMPREHENSION_LENGTH,
) -> Any:
    """Evaluate an expression with a limited set of allowed functions and types.

    Args:
        expr (str): The expression to evaluate.
        names (dict[str, Any]): A dictionary of variable names and their values
            to be used in the expression.
        max_comprehension_length (int): The maximum length of comprehensions
            allowed in the expression.

    Returns:
        Any: The result of the evaluated expression.
    """
    original_comp_len = simpleeval.MAX_COMPREHENSION_LENGTH
    try:
        simpleeval.MAX_COMPREHENSION_LENGTH = max_comprehension_length
        expr_evaluator = simpleeval.EvalWithCompoundTypes(
            names=names or {}, functions=_ALLOWED_CALLABLES
        )
        return expr_evaluator.eval(expr)
    finally:
        simpleeval.MAX_COMPREHENSION_LENGTH = original_comp_len
