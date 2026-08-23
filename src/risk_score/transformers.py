"""The declared feature contract, and the two transformers that enforce it.

Why this module exists
----------------------
``build_preprocessor`` used to decide which transformer each column went to by
inspecting its *runtime dtype*:

.. code-block:: python

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns

Anything pandas had not managed to parse was therefore, by definition,
categorical. ``earliest_cr_line`` is a date string with 655 distinct values in
the real extract; ``int_rate`` and ``revol_util`` arrive as percent strings in
the ``loans_full_schema`` extract with roughly 600 and 1100 distinct values. All
three went to ``OneHotEncoder``, and with ``sparse_output=False`` that is about
2400 dense float columns over 1.8M rows - roughly 35 GB. The pipeline was not
slow on the real dataset, it was impossible to run, and the monotone rate
information was destroyed in the process (audit B01, B22, B23).

The fix is not a better dtype check. It is to remove the inference entirely:

* :class:`FeatureSpec` **declares** the numeric list and the categorical list.
  The preprocessor is built from those lists and never calls ``select_dtypes``.
* :class:`CanonicalizeFrame` casts every input column to its declared
  :class:`~risk_score.features.ParseKind` and reindexes to the declared inputs,
  so a column cannot arrive as ``object`` and be silently reinterpreted.
* :class:`EngineerFeatures` drops the raw sources its outputs replace and
  reindexes to exactly :attr:`FeatureSpec.model_features`.

After that, a string column reaching ``OneHotEncoder`` unintentionally is not a
bug that has been fixed - it is a state the code cannot represent.

Why they are transformers rather than functions
-----------------------------------------------
Both steps live *inside* the ``sklearn`` ``Pipeline``. Training and ``POST
/predict`` then run the same objects in the same order, from one pickle, so they
cannot drift apart. Every function called from here is stateless - no means, no
medians, no category vocabularies - so putting them ahead of the
``ColumnTransformer`` introduces no leakage: nothing here learns anything from
the rows it sees.

The cost is honest and worth stating: a pickled bundle is bound to this module
path forever. :data:`SPEC_VERSION` and the never-move rule in
``docs/decisions/`` are how that is managed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from risk_score.feature_engineering import build_feature_matrix, parse_declared_columns
from risk_score.features import (
    COLUMN_REGISTRY,
    ENGINEERED_BY_NAME,
    EngineeredFeature,
    ParseKind,
    column_spec,
    model_feature_columns,
    resolvable_engineered_features,
)
from risk_score.schema import DEFAULT_DATE_FORMATS, normalize_column_names, parse_month_column

#: Bumped whenever the meaning of a :class:`FeatureSpec` field changes, so a
#: bundle written by an older version is rejected at load rather than producing
#: plausible scores from a misread contract.
SPEC_VERSION = 1

#: Missing categorical values become their own level rather than being imputed to
#: the mode. In credit data missingness is informative - an applicant who did not
#: state ``emp_length`` is not an applicant with the most common ``emp_length`` -
#: and a real level is also what makes the reason codes readable.
MISSING_CATEGORY = "__missing__"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The frozen contract between the fitted model and everything that feeds it.

    Built once at training time from the columns the extract actually has, then
    persisted in the bundle. Serving reads it back rather than re-deriving it,
    which is what stops a ``/predict`` request from being preprocessed
    differently from the rows the model was fitted on.
    """

    raw_inputs: tuple[str, ...]
    """Canonical columns :class:`CanonicalizeFrame` keeps, in registry order."""

    required_raw_inputs: tuple[str, ...]
    """Inputs whose absence raises instead of being filled with NA."""

    engineered: tuple[str, ...]
    """Derived feature names, in build order."""

    numeric_features: tuple[str, ...]
    """Columns the preprocessor imputes and scales. Declared, never inferred."""

    categorical_features: tuple[str, ...]
    """Columns the preprocessor one-hot encodes. Declared, never inferred."""

    include_lender_priced: bool = False
    """Whether the ``LENDER_PRICED`` tier was admitted. Travels with the spec so a
    metrics file can say on its own whether the lender's own price was an input."""

    date_formats: tuple[str, ...] = DEFAULT_DATE_FORMATS
    """Pinned at fit time: the served row is parsed by the formats that trained it."""

    spec_version: int = SPEC_VERSION

    def __post_init__(self) -> None:
        overlap = set(self.numeric_features) & set(self.categorical_features)
        if overlap:
            # A column in both lists is transformed twice and appears twice in
            # the design matrix, which quietly doubles its weight.
            raise ValueError(f"Columns declared both numeric and categorical: {sorted(overlap)}.")
        if not self.numeric_features and not self.categorical_features:
            raise ValueError(
                "FeatureSpec declares no model features. The extract supplied none "
                f"of {list(model_feature_columns(include_lender_priced=True))}."
            )
        unknown_required = set(self.required_raw_inputs) - set(self.raw_inputs)
        if unknown_required:
            raise ValueError(f"required_raw_inputs not in raw_inputs: {sorted(unknown_required)}.")
        unknown_engineered = [name for name in self.engineered if name not in ENGINEERED_BY_NAME]
        if unknown_engineered:
            raise ValueError(f"Unknown engineered features: {unknown_engineered}.")

    @property
    def model_features(self) -> tuple[str, ...]:
        """Every column the model sees, numeric first. This is the frame layout
        :class:`EngineerFeatures` produces and the order the preprocessor expects."""
        return (*self.numeric_features, *self.categorical_features)

    @property
    def engineered_specs(self) -> tuple[EngineeredFeature, ...]:
        """The declared features as registry objects, in build order."""
        return tuple(ENGINEERED_BY_NAME[name] for name in self.engineered)

    def parse_kinds(self) -> dict[str, ParseKind]:
        """Declared parse rule per raw input."""
        return {name: column_spec(name).parse for name in self.raw_inputs}

    def summary(self) -> str:
        """One-line summary for logs and the run manifest."""
        return (
            f"features={len(self.model_features)} "
            f"(numeric={len(self.numeric_features)} categorical={len(self.categorical_features)}) "
            f"engineered={len(self.engineered)} inputs={len(self.raw_inputs)} "
            f"lender_priced={'on' if self.include_lender_priced else 'off'}"
        )


def build_feature_spec(
    available: Iterable[str],
    *,
    include_lender_priced: bool = False,
    date_formats: tuple[str, ...] = DEFAULT_DATE_FORMATS,
) -> FeatureSpec:
    """Derive the contract from the columns one extract actually supplies.

    The spec has to be data-dependent, and only at this one point. ``revol_util``
    exists in one real extract and not the other, and ``fico_range_low`` /
    ``fico_range_high`` are absent from *both* - so a hard-coded feature list
    either references columns that do not exist or omits ones that do.
    ``configs/feature_config.yaml`` listed ``fico_range_low`` as a model feature
    and was loaded by nothing, which is how that went unnoticed.

    Everything downstream then treats the result as fixed.
    """
    present = {str(name) for name in available}

    # 1. Raw candidates: modelable tiers, minus whatever this extract lacks.
    raw_candidates = [
        name
        for name in model_feature_columns(include_lender_priced=include_lender_priced)
        if name in present
    ]

    # 2. Which derived features this extract can support. Walked by the shared
    #    resolver so the spec and build_feature_matrix cannot disagree.
    buildable, _blocked = resolvable_engineered_features(present)

    # 3. Inputs: the raw candidates, plus every source a buildable feature reads -
    #    including sources that are not features themselves (`earliest_cr_line`
    #    and `issue_d` exist only to produce credit_history_months).
    needed: set[str] = set(raw_candidates)
    for feature in buildable:
        needed.update(name for name in feature.requires if name in present)
        for group in feature.requires_any:
            if all(name in present for name in group):
                needed.update(group)
                break
    raw_inputs = tuple(name for name in COLUMN_REGISTRY if name in needed)

    # 4. Model features. Raw candidates lose anything a derived feature replaces.
    consumed = {name for feature in buildable for name in feature.consumes}
    numeric: list[str] = []
    categorical: list[str] = []
    for name in raw_candidates:
        if name in consumed:
            continue
        (numeric if column_spec(name).is_numeric else categorical).append(name)
    for feature in buildable:
        (numeric if feature.numeric else categorical).append(feature.name)

    return FeatureSpec(
        raw_inputs=raw_inputs,
        # An input the registry marks required cannot be NA-filled: a run with no
        # `annual_inc` column at all is a misconfiguration, not missing data.
        required_raw_inputs=tuple(name for name in raw_inputs if COLUMN_REGISTRY[name].required),
        engineered=tuple(feature.name for feature in buildable),
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        include_lender_priced=include_lender_priced,
        date_formats=tuple(date_formats),
    )


class _SpecTransformer(BaseEstimator, TransformerMixin):
    """Shared plumbing: hold the spec, learn nothing, insist on a DataFrame."""

    def __init__(self, spec: FeatureSpec) -> None:
        # Stored verbatim under the constructor's own parameter name so
        # `sklearn.clone` and `get_params` work without a custom implementation.
        self.spec = spec

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Nothing is learned. Present so the step composes inside a ``Pipeline``.

        That is the property which makes it safe to run these steps ahead of the
        train/validation split: a transformer with no fitted state cannot carry
        information from one partition into another.

        ``fitted_`` exists only because ``check_is_fitted`` decides an estimator
        is unfitted when it has no trailing-underscore attribute, and a
        ``Pipeline`` ending in such a step therefore reports itself unfitted
        after ``fit``. Nothing reads it.
        """
        self._require_frame(X)
        self.fitted_ = True
        return self

    @staticmethod
    def _require_frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame with named columns, got {type(X).__name__}. "
                f"These steps route columns by name; a bare array has none."
            )
        return X


class CanonicalizeFrame(_SpecTransformer):
    """Raw extract or request payload -> the declared inputs, correctly typed.

    Four things, in this order:

    1. Resolve source names to canonical ones by declared alias priority.
    2. Refuse to continue if a required input is absent, naming it.
    3. Reindex to :attr:`FeatureSpec.raw_inputs` - which drops unknown columns,
       creates absent optional ones as all-NA, and fixes the column order.
    4. Cast each column to its declared ``ParseKind``.

    Step 3 before step 4 is deliberate: parsing 145 columns to throw 115 away is
    most of the work for none of the result.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._require_frame(X)
        frame, _report = normalize_column_names(frame)

        missing = [name for name in self.spec.required_raw_inputs if name not in frame.columns]
        if missing:
            raise KeyError(
                f"Required input columns are absent: {missing}. "
                f"Present after alias resolution: {sorted(frame.columns)[:20]}."
            )

        # reindex rather than `frame[cols]`: it tolerates absent optional columns
        # by creating them as NA, which `[]` would raise on.
        frame = frame.reindex(columns=list(self.spec.raw_inputs))

        # Dates first, with one whole-column format from the pinned list. An
        # all-NA column - an absent optional one, just created above - comes back
        # as NaT rather than raising.
        dates = {
            name: parse_month_column(frame[name], formats=self.spec.date_formats, column_name=name)[
                0
            ]
            for name, kind in self.spec.parse_kinds().items()
            if kind is ParseKind.MONTH_DATE
        }
        frame = frame.assign(**dates) if dates else frame
        # Everything else by declared kind, through the one function that owns it.
        return parse_declared_columns(frame, columns=self.spec.raw_inputs)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Declared inputs. Lets ``set_output(transform="pandas")`` work."""
        return list(self.spec.raw_inputs)


class EngineerFeatures(_SpecTransformer):
    """The declared inputs -> exactly :attr:`FeatureSpec.model_features`.

    ``require_all=True`` and the pinned feature list are the point. At training
    time the extract is known, so a feature that cannot be built is a
    misconfiguration rather than something to skip; at serving time the list is
    whatever the model was fitted on, so a request carrying an extra column
    cannot produce a feature the model has never seen.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = build_feature_matrix(
            self._require_frame(X), require_all=True, features=self.spec.engineered_specs
        )

        missing = [name for name in self.spec.model_features if name not in frame.columns]
        if missing:
            raise KeyError(
                f"Declared model features are absent after engineering: {missing}. "
                f"Built: {sorted(frame.columns)}."
            )
        # Positional selection, so the ColumnTransformer's declared indices and
        # this frame's layout are the same object of truth.
        frame = frame.loc[:, list(self.spec.model_features)]

        # Missing becomes a level, not an imputed mode - see MISSING_CATEGORY.
        # `.astype(object)` first because fico_band is a Categorical, and filling
        # a Categorical with an unlisted value raises.
        filled = {
            name: frame[name]
            .astype("object")
            .where(frame[name].notna(), MISSING_CATEGORY)
            .astype(str)
            for name in self.spec.categorical_features
        }
        return frame.assign(**filled) if filled else frame

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """The model features, in the order this transformer emits them."""
        return list(self.spec.model_features)
