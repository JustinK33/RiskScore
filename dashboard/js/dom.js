/**
 * The DOM primitives every panel builds from. No fetching, no charts, no state.
 *
 * There is no template library here and no `innerHTML` anywhere in this
 * dashboard. That is one decision with two reasons: a run id, a feature label,
 * and a job's error `detail` all originate off this page - the last one is an
 * exception message from a child process - and `textContent` cannot execute
 * them. The second reason is that a `<table>` built from an HTML string is
 * unreadable at the point where the interesting question is which column formats
 * how, which is exactly what `renderTable` makes explicit.
 *
 * Every table this module builds is a real `<table>` with a `<caption>` and
 * `<th scope>`, because the tables are also the accessible alternative to the
 * canvases: a screen reader gets the numbers, not a description of a picture.
 */

import { MISSING } from "./format.js";

/** `querySelector`, shortened, because this file uses it forty times. */
export function $(selector, root = document) {
  return root.querySelector(selector);
}

/** `querySelectorAll` as a real array. */
export function $$(selector, root = document) {
  return [...root.querySelectorAll(selector)];
}

/**
 * Build an element. `props` sets properties, except `dataset`, `attrs`, and
 * `style`, which are merged.
 *
 * Properties rather than attributes by default (`textContent`, `className`,
 * `disabled`), so a value is never stringified into markup.
 */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined) continue;
    if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "attrs") for (const [name, val] of Object.entries(value)) node.setAttribute(name, val);
    else if (key === "style") Object.assign(node.style, value);
    else node[key] = value;
  }
  for (const child of [children].flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Replace an element's children in one operation. */
export function replaceChildren(node, children) {
  node.replaceChildren(...[children].flat().filter((child) => child !== null && child !== undefined));
}

/** Set an element's text, tolerating a missing element during a partial render. */
export function setText(selectorOrNode, value) {
  const node = typeof selectorOrNode === "string" ? $(selectorOrNode) : selectorOrNode;
  if (!node) return;
  node.textContent = value === null || value === undefined || value === "" ? MISSING : String(value);
}

/**
 * A table from a column spec.
 *
 * `columns` entries are `{key, label, format, align, tone}`. `format` receives
 * the whole row, not just the cell, because a PSI cell's *colour* depends on the
 * band column beside it. `tone` names a CSS custom property, so a highlighted
 * cell follows the theme.
 *
 * An empty `rows` renders the `empty` message inside the table rather than
 * leaving a headed table with no body, which reads as a loading state that never
 * finished.
 */
export function renderTable(columns, rows, { caption = "", empty = "No rows.", classes = "" } = {}) {
  const head = el("thead", {}, [
    el(
      "tr",
      {},
      columns.map((column) =>
        el("th", {
          textContent: column.label,
          // `scope` is what lets a screen reader announce "AUC ROC: 0.683" rather
          // than reading a grid of unattached numbers.
          attrs: { scope: "col" },
          className: column.align === "right" ? "num" : "",
        }),
      ),
    ),
  ]);

  const body = el(
    "tbody",
    {},
    rows.length === 0
      ? [
          el("tr", {}, [
            el("td", {
              textContent: empty,
              className: "table-empty",
              attrs: { colspan: String(columns.length) },
            }),
          ]),
        ]
      : rows.map((row) =>
          el(
            "tr",
            {},
            columns.map((column, index) => {
              const text = column.format ? column.format(row) : row[column.key];
              const tone = column.tone?.(row);
              const props = {
                textContent: text === null || text === undefined || text === "" ? MISSING : String(text),
                className: column.align === "right" ? "num" : "",
                style: tone ? { color: `var(--${tone})` } : null,
              };
              // The first column is the row's header, so a screen reader can say
              // which row a cell belongs to.
              return index === 0
                ? el("th", { ...props, attrs: { scope: "row" } })
                : el("td", props);
            }),
          ),
        ),
  );

  return el("table", { className: `data-table ${classes}`.trim() }, [
    caption ? el("caption", { textContent: caption }) : null,
    head,
    body,
  ]);
}

/**
 * A chart's accessible alternative: a `<details>` holding the same numbers.
 *
 * Collapsed by default and reachable by keyboard. A canvas is opaque to assistive
 * technology no matter what `aria-label` it carries - a label can say "calibration
 * curve" but cannot say that bin 7 predicted 0.31 and observed 0.28 - so the
 * table is the actual alternative and the label is a summary.
 */
export function dataTableDetails(summaryText, table) {
  return el("details", { className: "data-fallback" }, [
    el("summary", { textContent: summaryText }),
    table,
  ]);
}

/**
 * Set or clear a banner.
 *
 * `tone` is one of `info`, `warn`, `danger`, `ok`. Hidden with `hidden` rather
 * than `display: none` so the element is out of the accessibility tree as well as
 * out of the layout.
 */
export function setBanner(node, message, tone = "info") {
  if (!node) return;
  if (!message) {
    node.hidden = true;
    node.textContent = "";
    return;
  }
  node.hidden = false;
  node.dataset.tone = tone;
  node.textContent = message;
}

/**
 * The status pill in the header.
 *
 * The element is an `aria-live="polite"` region in the markup, so a change here
 * is announced without stealing focus - which matters because this is where a
 * failed load is reported, and a sighted user sees it immediately while a screen
 * reader user would otherwise never learn the page had stopped loading.
 */
export function setStatus(node, message, state = "ok") {
  if (!node) return;
  node.textContent = message;
  node.dataset.state = state;
}

/** A labelled key/value pair for the identity strip. */
export function definition(term, value, { mono = false } = {}) {
  return el("div", { className: "def" }, [
    el("dt", { textContent: term }),
    el("dd", { textContent: value ?? MISSING, className: mono ? "mono" : "" }),
  ]);
}
