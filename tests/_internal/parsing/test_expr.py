# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.parsing.expr import evaluate_expr


def _foo(x: int) -> int:
    """Example function for testing."""
    return x + 3


def test_evaluate_expr() -> None:
    # Test basic arithmetic.
    assert evaluate_expr("1 + 2") == 3
    assert evaluate_expr("5 - 3") == 2
    assert evaluate_expr("4 * 2") == 8
    assert evaluate_expr("8 / 2") == 4.0

    # Test string concatenation.
    assert evaluate_expr("'Hello' + ' ' + 'World'") == "Hello World"

    # Test list operations.
    assert evaluate_expr("[1, 2, 3] + [4, 5]") == [1, 2, 3, 4, 5]
    assert evaluate_expr("[1, 2, 3] * 2") == [1, 2, 3, 1, 2, 3]

    # Test dictionary access.
    assert evaluate_expr("{'a': 1, 'b': 2}['a']") == 1

    # Test function calls.
    assert evaluate_expr("max(1, 2, 3)") == 3
    assert evaluate_expr("min(10, -5, 0)") == -5

    # Test custom functions.
    assert evaluate_expr("bar(7)", functions={"bar": _foo}) == 10

    # Test with custom names.
    assert evaluate_expr("x + 5", {"x": 10, "y": "unused"}) == 15

    # Test with complex expressions.
    assert evaluate_expr("sum([x for x in range(500)])") == 124750
    assert (
        evaluate_expr(
            "sum([x for x in range(100_000) if x % 7 == 2])",
            max_comprehension_length=100_000,
        )
        == 714292857
    )
