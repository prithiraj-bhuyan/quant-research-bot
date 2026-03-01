"""
export.py — Export research artifacts as Markdown or CSV.
"""

import csv
import io


# ---------------------------------------------------------------------------
# Evidence Table exports
# ---------------------------------------------------------------------------

def export_evidence_table_csv(data: dict) -> str:
    """Export evidence table as CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Claim", "Evidence Snippet", "Citation", "Confidence", "Notes"])
    for claim in data.get("claims", []):
        writer.writerow([
            claim.get("claim", ""),
            claim.get("evidence_snippet", ""),
            claim.get("citation", ""),
            claim.get("confidence", ""),
            claim.get("notes", ""),
        ])
    return output.getvalue()


def export_evidence_table_markdown(data: dict) -> str:
    """Export evidence table as Markdown."""
    lines = [
        f"# Evidence Table — {data.get('thread_title', 'Research Thread')}",
        "",
        f"**Total Claims:** {data.get('total_claims', 0)} | "
        f"**Sources:** {data.get('total_sources', 0)}",
        "",
        "| # | Claim | Evidence Snippet | Citation | Confidence | Notes |",
        "|---|-------|-----------------|----------|------------|-------|",
    ]
    for i, claim in enumerate(data.get("claims", []), 1):
        c = claim.get("claim", "").replace("|", "\\|")
        e = claim.get("evidence_snippet", "")[:120].replace("|", "\\|").replace("\n", " ")
        cit = claim.get("citation", "").replace("|", "\\|")
        conf = claim.get("confidence", "")
        n = claim.get("notes", "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {c} | {e} | {cit} | {conf} | {n} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bibliography export
# ---------------------------------------------------------------------------

def export_bibliography_markdown(data: dict) -> str:
    """Export annotated bibliography as Markdown."""
    lines = [
        f"# Annotated Bibliography — {data.get('thread_title', 'Research Thread')}",
        "",
        f"**Total Sources:** {data.get('total_sources', 0)}",
        "",
    ]
    for i, src in enumerate(data.get("sources", []), 1):
        lines.append(f"## {i}. {src.get('source', 'Unknown')}")
        lines.append("")
        lines.append(f"**Method:** {src.get('method', 'N/A')}")
        lines.append("")
        lines.append(f"**Limitations:** {src.get('limitations', 'N/A')}")
        lines.append("")
        lines.append(f"**Why it matters:** {src.get('why_it_matters', 'N/A')}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def export_bibliography_csv(data: dict) -> str:
    """Export bibliography as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Source", "Method", "Limitations", "Why It Matters"])
    for src in data.get("sources", []):
        writer.writerow([
            src.get("source", ""),
            src.get("method", ""),
            src.get("limitations", ""),
            src.get("why_it_matters", ""),
        ])
    return output.getvalue()


# ---------------------------------------------------------------------------
# Synthesis Memo export
# ---------------------------------------------------------------------------

def export_synthesis_memo_markdown(data: dict) -> str:
    """Export synthesis memo as Markdown."""
    lines = [
        f"# Synthesis Memo — {data.get('thread_title', 'Research Thread')}",
        "",
        f"**Word Count:** ~{data.get('word_count', 0)}",
        "",
        "---",
        "",
        data.get("memo_text", "No memo content."),
        "",
    ]

    refs = data.get("references", [])
    if refs:
        lines.append("---")
        lines.append("")
        lines.append("## References")
        lines.append("")
        for ref in refs:
            num = ref.get("number", "?")
            src = ref.get("source", "Unknown")
            lines.append(f"[{num}] {src}")
        lines.append("")

    return "\n".join(lines)
