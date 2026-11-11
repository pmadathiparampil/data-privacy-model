import os
import re
import sys
import html
import difflib


def report_name(before_path, after_path):
    # before1234.txt -> "1234" ; after1234.txt -> "1234"
    def suffix(path, prefix):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.lower().startswith(prefix):
            return stem[len(prefix):]
        return ""

    sfx = suffix(before_path, "before") or suffix(after_path, "after")
    return f"diff_report{sfx}.html"


def tokenize(text):
    # words/numbers, whitespace runs, or single punctuation characters
    return re.findall(r"\w+|\s+|[^\w\s]", text)


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main(before_path, after_path, context=5):
    a_text, b_text = read(before_path), read(after_path)

    identical = a_text == b_text
    if identical:
        print("No differences found. The files are identical.")

    a, b = tokenize(a_text), tokenize(b_text)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)

    html_parts = []
    change_count = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old = "".join(a[i1:i2])
        new = "".join(b[j1:j2])

        if tag == "equal":
            html_parts.append(html.escape(new))
            continue

        change_count += 1
        before_ctx = "".join(a[max(0, i1 - context):i1])
        after_ctx = "".join(a[i2:i2 + context])

        label = {"replace": "CHANGED", "delete": "REMOVED", "insert": "ADDED"}[tag]
        print(f"--- Change #{change_count} ({label}) ---")
        print(f"Context : ...{before_ctx!r} [ {old!r} -> {new!r} ] {after_ctx!r}...")
        print()

        if old:
            html_parts.append(f'<del style="background:#ffd7d5">{html.escape(old)}</del>')
        if new:
            html_parts.append(f'<ins style="background:#ccffd8">{html.escape(new)}</ins>')

    if identical:
        header = "<h3>No differences found. The files are identical.</h3>"
    else:
        header = "<h3>Red = removed, Green = added</h3>"

    out_name = report_name(before_path, after_path)
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(
            "<html><body>" + header +
            "<pre style='white-space:pre-wrap;font-family:monospace'>"
            + "".join(html_parts) +
            "</pre></body></html>"
        )

    print(f"Total changes: {change_count}")
    print(f"Highlighted report saved to {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py before.txt after.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
