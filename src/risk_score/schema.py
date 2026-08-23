"""Schema normalization: raw extract column names and formats -> canonical ones.

This is the first thing that touches the data, so a mistake here is invisible
everywhere downstream. Two bugs lived in this module:

**Duplicate columns (audit B02).** Alias resolution used a "first source column
wins" rule over ``loans.columns``, and skipped any canonical name already
claimed. Given a frame with both ``funded_amnt`` and ``loan_amnt`` - which the
standard extract has - ``funded_amnt`` came first, got renamed to ``loan_amnt``,
and then the genuine ``loan_amnt`` was skipped and kept its name. The result was
two columns both called ``loan_amnt``, so ``loans["loan_amnt"]`` returned a
DataFrame and the run died several modules later inside a numeric coercion.
Resolution is now by declared alias *priority*: the canonical name always beats
an alias, losers are dropped, and every drop is reported.

**Per-row date inference (audit B28).** ``pd.to_datetime(..., format="mixed")``
infers a format per row. On a column of ``03/04/2016`` values it can read some
rows as 3 April and others as 4 March, producing a date column that is wrong for
an arbitrary subset of rows with no error and no warning - and dates here decide
which partition a loan lands in. Parsing now tries an ordered list of
*unambiguous* whole-column formats. Ambiguous input raises and names the
formats tried, so the caller states their intent instead of getting a coin flip.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from risk_score.features import COLUMN_REGISTRY, ParseKind, alias_lookup, required_columns

#: Whole-column date formats, tried in order. Every entry is unambiguous: no
#: format here can parse the same string as another with a different meaning.
#: `%m/%d/%Y` is deliberately absent - it is indistinguishable from `%d/%m/%Y`
#: for the first twelve days of a month, and guessing silently is how you get a
#: date column that is wrong for a third of its rows.
DEFAULT_DATE_FORMATS: tuple[str, ...] = (
    "%b-%Y",  # Aug-2003 - the standard Lending Club encoding
    "%Y-%m-%d",  # ISO, produced by re-exports
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m",  # month precision, no day
    "%d-%b-%Y",  # 01-Aug-2003
    "%b-%y",  # Aug-03 - older extracts
    "%B %Y",  # August 2003
)

#: A candidate format must parse at least this share of non-null values to be
#: accepted. Set just below 1.0 rather than at it because real extracts carry a
#: handful of genuinely corrupt cells, and aborting a 1.8M-row run over four bad
#: dates is worse than quarantining them - which the caller does, with a count.
MIN_DATE_PARSE_RATE = 0.98


@dataclass(frozen=True)
class SchemaReport:
    """What normalization did, so a run can be audited without rerunning it.

    Every field here ends up in the run manifest. A pipeline that silently drops
    a fifth of its columns looks identical to one that drops none, which is why
    these are counted rather than logged and forgotten.
    """

    renamed: dict[str, str] = field(default_factory=dict)
    """Source column name -> canonical name."""

    dropped_duplicate_sources: dict[str, str] = field(default_factory=dict)
    """Source column -> the canonical name it lost the contest for."""

    unknown_columns: tuple[str, ...] = ()
    """Columns absent from the registry. Not an error, but worth seeing."""

    missing_required: tuple[str, ...] = ()
    """Required canonical columns that no source column supplied."""

    date_formats_used: dict[str, str] = field(default_factory=dict)
    """Date column -> the format that won, for reproducibility."""

    unparseable_dates: dict[str, int] = field(default_factory=dict)
    """Date column -> count of values that became NaT."""

    def summary(self) -> str:
        """One-line summary for logs."""
        return (
            f"renamed={len(self.renamed)} "
            f"dropped_duplicates={len(self.dropped_duplicate_sources)} "
            f"unknown={len(self.unknown_columns)} "
            f"unparseable_dates={sum(self.unparseable_dates.values())}"
        )


def resolve_column_names(
    columns: Iterable[object],
    *,
    extra_aliases: Mapping[str, str | Iterable[str]] | None = None,
) -> tuple[dict[str, str], dict[str, str], tuple[str, ...]]:
    """Decide which source column becomes each canonical column.

    Returns ``(rename_map, dropped, unknown)``.

    The contest is settled by the position of the matched key in the column's
    declared alias list, canonical name first. So ``loan_amnt`` (priority 0)
    beats ``funded_amnt`` (priority 3) regardless of which appears first in the
    frame - the fix for audit B02, whose whole cause was letting frame order
    decide.
    """
    lookup = alias_lookup(extra_aliases)
    # canonical -> (priority, source name)
    best: dict[str, tuple[int, str]] = {}
    losers: dict[str, str] = {}
    unknown: list[str] = []

    for column in columns:
        source = str(column)
        canonical = lookup.get(source.strip().lower())
        if canonical is None:
            unknown.append(source)
            continue
        # Priority is the index of the matched key. Config-supplied aliases are
        # not in match_keys(), so they sort after every declared one, which is
        # the right default: an explicit registry entry outranks a config patch.
        keys = [key.strip().lower() for key in COLUMN_REGISTRY[canonical].match_keys()]
        try:
            priority = keys.index(source.strip().lower())
        except ValueError:
            priority = len(keys)

        incumbent = best.get(canonical)
        if incumbent is None:
            best[canonical] = (priority, source)
        elif priority < incumbent[0]:
            losers[incumbent[1]] = canonical
            best[canonical] = (priority, source)
        else:
            losers[source] = canonical

    rename_map = {source: canonical for canonical, (_, source) in best.items()}
    return rename_map, losers, tuple(unknown)


def normalize_column_names(
    loans: pd.DataFrame,
    *,
    column_aliases: Mapping[str, str | Iterable[str]] | None = None,
    drop_unknown: bool = False,
) -> tuple[pd.DataFrame, SchemaReport]:
    """Rename source columns to canonical names and report what happened.

    ``drop_unknown=True`` keeps only registry columns. That is the pipeline's
    setting: the real extract has 145 columns of which about 30 are used, and
    carrying the rest costs memory for nothing.
    """
    rename_map, losers, unknown = resolve_column_names(loans.columns, extra_aliases=column_aliases)

    to_drop = list(losers)
    if drop_unknown:
        to_drop.extend(unknown)
    # One drop and one rename, in that order. Dropping first means the rename
    # cannot collide with a column that is about to disappear.
    result = loans.drop(columns=to_drop) if to_drop else loans
    result = result.rename(columns=rename_map)

    present = set(result.columns)
    report = SchemaReport(
        # Identity renames are dropped from the report: a manifest listing
        # 'loan_amnt -> loan_amnt' 25 times buries the two that mattered.
        renamed={
            source: canonical for source, canonical in rename_map.items() if source != canonical
        },
        dropped_duplicate_sources=dict(losers),
        unknown_columns=unknown,
        missing_required=tuple(name for name in required_columns() if name not in present),
    )
    return result, report


def parse_month_column(
    series: pd.Series,
    *,
    formats: Sequence[str] = DEFAULT_DATE_FORMATS,
    column_name: str = "date",
) -> tuple[pd.Series, str]:
    """Parse a date column with one explicit whole-column format.

    Returns ``(parsed, format_used)``. Unparseable values become ``NaT`` for the
    caller to quarantine and count.

    Each candidate format is applied to the entire column and scored by how much
    of it parsed. The first format clearing :data:`MIN_DATE_PARSE_RATE` wins. If
    none does, the formats are combined - later ones filling in what earlier ones
    left as ``NaT`` - which handles a genuinely heterogeneous column without ever
    resorting to per-row inference over *ambiguous* formats, since no ambiguous
    format is a candidate.
    """
    text = series.astype("string").str.strip()
    # An empty string is missing data, not a malformed date; converting it here
    # keeps it out of the unparseable count.
    text = text.replace("", pd.NA)
    non_null = int(text.notna().sum())
    if non_null == 0:
        return pd.to_datetime(pd.Series([pd.NaT] * len(series), index=series.index)), "none"

    attempts: list[tuple[float, str, pd.Series]] = []
    for candidate in formats:
        parsed = pd.to_datetime(text, format=candidate, errors="coerce")
        rate = float(parsed.notna().sum()) / non_null
        if rate >= MIN_DATE_PARSE_RATE:
            return parsed, candidate
        attempts.append((rate, candidate, parsed))

    # Nothing matched cleanly. Combine, best-first, so a mixed column still
    # parses; then insist the union clears the floor.
    attempts.sort(key=lambda item: item[0], reverse=True)
    combined = attempts[0][2]
    used = [attempts[0][1]]
    for _, candidate, parsed in attempts[1:]:
        if combined.notna().all():
            break
        filled = combined.fillna(parsed)
        if int(filled.notna().sum()) > int(combined.notna().sum()):
            combined = filled
            used.append(candidate)

    if float(combined.notna().sum()) / non_null < MIN_DATE_PARSE_RATE:
        samples = text.dropna().unique()[:5].tolist()
        raise ValueError(
            f"Could not parse date column `{column_name}` with any known format. "
            f"Tried {list(formats)}. Sample values: {samples}. "
            f"Pass an explicit format via `date_formats=` - note that slash-separated "
            f"dates like '03/04/2016' are ambiguous and are never guessed."
        )
    return combined, "+".join(used)


def normalize_credit_schema(
    loans: pd.DataFrame,
    *,
    column_aliases: Mapping[str, str | Iterable[str]] | None = None,
    date_formats: Sequence[str] = DEFAULT_DATE_FORMATS,
    drop_unknown: bool = False,
) -> tuple[pd.DataFrame, SchemaReport]:
    """Canonicalize names and parse every registered date column.

    Date parsing covers *all* ``MONTH_DATE`` columns, not just ``issue_d``. The
    previous version parsed only the split column, so ``earliest_cr_line``
    reached the preprocessor as a string with 655 distinct values and was
    one-hot encoded (audit B01).
    """
    result, report = normalize_column_names(
        loans, column_aliases=column_aliases, drop_unknown=drop_unknown
    )

    date_columns = [
        name
        for name, spec in COLUMN_REGISTRY.items()
        if spec.parse is ParseKind.MONTH_DATE and name in result.columns
    ]
    if not date_columns:
        return result, report

    # Build every parsed column first, then assign once. Assigning inside the
    # loop would trigger a fragmentation warning and a copy per column under
    # pandas 3 copy-on-write.
    parsed_columns: dict[str, pd.Series] = {}
    for name in date_columns:
        parsed, format_used = parse_month_column(
            result[name], formats=date_formats, column_name=name
        )
        parsed_columns[name] = parsed
        report.date_formats_used[name] = format_used
        report.unparseable_dates[name] = int((result[name].notna() & parsed.isna()).sum())
    return result.assign(**parsed_columns), report
