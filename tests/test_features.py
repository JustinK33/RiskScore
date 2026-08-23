"""Tests for the canonical column registry.

The registry is a data structure, so most of what can go wrong is a
contradiction between two of its entries rather than a bug in a function. Those
checks run at import time; the tests here prove the import-time checks actually
reject what they claim to.
"""

from __future__ import annotations

import pytest

from risk_score.features import (
    COLUMN_REGISTRY,
    ENGINEERED_BY_NAME,
    ENGINEERED_FEATURES,
    MIN_ALIAS_LENGTH,
    ColumnSpec,
    FeatureTier,
    ParseKind,
    _validate_registry,
    alias_lookup,
    alias_priority,
    column_spec,
    columns_in_tier,
    columns_to_read,
    model_feature_columns,
    required_columns,
)


def test_every_alias_resolves_to_exactly_one_canonical_column() -> None:
    lookup = alias_lookup()
    for spec in COLUMN_REGISTRY.values():
        for key in spec.match_keys():
            assert lookup[key.strip().lower()] == spec.name


def test_b03_short_aliases_are_rejected() -> None:
    """`"n"` was an alias for `loan_status`, so any column named 'n' - and the
    real extract has unnamed index columns that read as such - silently became
    the target. The floor makes that unrepresentable."""
    assert "n" not in alias_lookup()

    with pytest.raises(ValueError, match="shorter than"):
        _validate_registry(
            [
                ColumnSpec(
                    name="loan_status",
                    tier=FeatureTier.TARGET_SOURCE,
                    parse=ParseKind.CATEGORY,
                    aliases=("n",),
                )
            ]
        )


def test_short_canonical_names_are_allowed() -> None:
    """The floor applies to guesses, not to a column's literal source name."""
    assert len("id") < MIN_ALIAS_LENGTH
    assert column_spec("id").name == "id"


def test_registry_rejects_an_alias_claimed_by_two_columns() -> None:
    with pytest.raises(ValueError, match="claimed by both"):
        _validate_registry(
            [
                ColumnSpec(
                    name="annual_inc",
                    tier=FeatureTier.BORROWER,
                    parse=ParseKind.NUMERIC,
                    aliases=("income",),
                ),
                ColumnSpec(
                    name="dti",
                    tier=FeatureTier.BORROWER,
                    parse=ParseKind.NUMERIC,
                    aliases=("income",),
                ),
            ]
        )


def test_registry_rejects_a_duplicate_canonical_name() -> None:
    with pytest.raises(ValueError, match="Duplicate canonical column"):
        _validate_registry(
            [
                ColumnSpec(name="dti", tier=FeatureTier.BORROWER, parse=ParseKind.NUMERIC),
                ColumnSpec(name="dti", tier=FeatureTier.BORROWER, parse=ParseKind.PERCENT),
            ]
        )


def test_lender_priced_columns_are_excluded_by_default() -> None:
    lender_priced = set(columns_in_tier(FeatureTier.LENDER_PRICED))
    assert lender_priced == {"int_rate", "grade", "sub_grade", "installment"}

    default_features = set(model_feature_columns())
    assert not (default_features & lender_priced)
    assert (
        set(model_feature_columns(include_lender_priced=True)) == default_features | lender_priced
    )


def test_post_origination_columns_are_never_features() -> None:
    """No opt-in flag reaches this tier: these are known only after funding."""
    post = set(columns_in_tier(FeatureTier.POST_ORIGINATION))
    assert "total_pymnt" in post
    for include in (False, True):
        assert not (set(model_feature_columns(include_lender_priced=include)) & post)


def test_b01_date_columns_are_never_model_features() -> None:
    """`earliest_cr_line` has 655 distinct values in the real extract. Routing
    columns by runtime dtype sent it to OneHotEncoder; declaring the feature list
    keeps it out by construction."""
    for include in (False, True):
        features = model_feature_columns(include_lender_priced=include)
        for name in features:
            assert COLUMN_REGISTRY[name].parse is not ParseKind.MONTH_DATE


def test_p01_read_projection_is_far_smaller_than_the_raw_extract() -> None:
    """The real extract has 145 columns; reading all of them is what makes a
    1.8M-row run get killed."""
    read = columns_to_read()
    assert len(read) < 40
    # Identifiers and free text are the bulk of what is skipped.
    assert "url" not in read
    assert "desc" not in read
    # Lender-priced columns are read even though they are not modelled, because
    # the leakage-cost comparison fits both variants from a single read.
    assert "int_rate" in read
    assert set(model_feature_columns(include_lender_priced=True)) <= set(read)


def test_read_projection_covers_everything_required() -> None:
    assert set(required_columns()) <= set(columns_to_read())


def test_required_columns_are_the_ones_the_pipeline_cannot_impute() -> None:
    assert required_columns() == ("loan_status", "issue_d", "loan_amnt", "term", "annual_inc")


def test_alias_priority_puts_the_canonical_name_first() -> None:
    """This ordering is what settles the duplicate-column contest in schema.py."""
    priority = alias_priority("loan_amnt")
    assert priority[0] == "loan_amnt"
    assert priority.index("loan_amnt") < priority.index("funded_amnt")


def test_config_aliases_extend_the_lookup() -> None:
    lookup = alias_lookup({"annual_inc": ["yearly_pay"]})
    assert lookup["yearly_pay"] == "annual_inc"
    # A bare string is accepted as a single alias, not iterated per character.
    assert alias_lookup({"annual_inc": "yearly_pay"})["yearly_pay"] == "annual_inc"


def test_config_aliases_are_held_to_the_same_standard() -> None:
    with pytest.raises(KeyError, match="unknown canonical column"):
        alias_lookup({"not_a_column": ["whatever"]})
    with pytest.raises(ValueError, match="shorter than"):
        alias_lookup({"annual_inc": ["ai"]})
    with pytest.raises(ValueError, match="already claimed"):
        alias_lookup({"annual_inc": ["dti"]})


def test_column_spec_error_names_the_near_misses() -> None:
    """An error a reader can act on without opening the registry."""
    with pytest.raises(KeyError, match="fico_range_low"):
        column_spec("fico_range")


def test_engineered_features_only_require_things_that_exist() -> None:
    known = set(COLUMN_REGISTRY) | set(ENGINEERED_BY_NAME)
    for feature in ENGINEERED_FEATURES:
        assert set(feature.requires) <= known, feature.name
        # A consumed column must be a raw registry column: consuming another
        # derived feature would mean build order silently decides the result.
        assert set(feature.consumes) <= set(COLUMN_REGISTRY), feature.name


def test_engineered_features_are_declared_in_build_order() -> None:
    built: set[str] = set()
    for feature in ENGINEERED_FEATURES:
        derived_inputs = set(feature.requires) & set(ENGINEERED_BY_NAME)
        assert derived_inputs <= built, f"{feature.name} needs {derived_inputs - built} first"
        built.add(feature.name)


def test_consumed_columns_are_not_also_offered_as_raw_features() -> None:
    """A column that a derived feature replaces must not survive alongside it -
    that duplication is audit bug B01."""
    consumed = {name for feature in ENGINEERED_FEATURES for name in feature.consumes}
    engineered_names = set(ENGINEERED_BY_NAME)
    for name in consumed:
        assert name not in engineered_names


def test_numeric_parse_kinds_are_classified_as_numeric() -> None:
    assert column_spec("int_rate").is_numeric  # '13.56%' is a number, not a category
    assert column_spec("term").is_numeric  # ' 36 months' likewise
    assert column_spec("emp_length").is_numeric  # '10+ years' likewise
    assert not column_spec("purpose").is_numeric
    assert not column_spec("issue_d").is_numeric
