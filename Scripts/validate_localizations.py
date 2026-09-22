#!/usr/bin/env python3
"""Check translation coverage, plural forms, and format arguments without Xcode."""

from collections import Counter
import json
from pathlib import Path
import re
import sys


def units(node):
    if "stringUnit" in node:
        yield node["stringUnit"]
    for value in node.values():
        if isinstance(value, dict):
            yield from units(value)


def placeholders(text):
    # Ignore positional argument numbers: translations may reorder arguments.
    return Counter(re.sub(r"^%\d+\$", "%", match) for match in re.findall(
        r"%(?:\d+\$)?(?:[-+ #0]*\d*(?:\.\d+)?(?:ll|l|h|z)?[diuoxXfFeEgGaAcCsSp@]|%)", text
    ))


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    count = 0
    for path in sorted((root / "ChatLLM").glob("*.xcstrings")):
        catalog = json.loads(path.read_text())
        for key, entry in catalog["strings"].items():
            if not entry.get("shouldTranslate", True):
                continue
            count += 1
            localizations = entry.get("localizations", {})
            for language in ("en", "de", "es"):
                translation = localizations.get(language, {})
                values = list(units(translation))
                if not values:
                    errors.append(f"{path.name}: {language}: missing {key!r}")
                plural = translation.get("variations", {}).get("plural")
                if plural is not None and not {"one", "other"} <= plural.keys():
                    errors.append(f"{path.name}: {language}: incomplete plurals for {key!r}")
                for unit in values:
                    if unit.get("state") != "translated" or not unit.get("value"):
                        errors.append(f"{path.name}: {language}: unfinished {key!r}")
                    if placeholders(unit.get("value", "")) != placeholders(key):
                        errors.append(f"{path.name}: {language}: mismatched placeholders for {key!r}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {count} entries in English, German, and Spanish.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
