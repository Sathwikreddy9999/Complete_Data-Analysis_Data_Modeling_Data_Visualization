import streamlit as st
import os
import io
import json
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from style_utils import apply_apple_style

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
PARSE_MODEL    = "nvidia/nemoretriever-parse"
CHAT_MODEL     = "nvidia/llama-3.1-nemotron-ultra-253b-v1"   # Most powerful Nemotron available

NVIDIA_BASE    = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS        = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

OUTPUT_DIR = Path("parsed_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_content(message: dict) -> str:
    """
    Safely extract text from an NVIDIA chat message.
    Nemotron-ultra (reasoning model) puts output in 'reasoning_content', not 'content'.
    """
    return (message.get("content") or message.get("reasoning_content") or "").strip()

# ── Helpers: document → images ────────────────────────────────────────────────

def pdf_to_page_images(file_bytes: bytes) -> list[str]:
    """Convert each PDF page to a base64-encoded PNG (low DPI to stay within API limits)."""
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    uris = []
    for page in doc:
        pix = page.get_pixmap(dpi=72)           # lower DPI avoids 400 from NVIDIA
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode()
        uris.append(f"data:image/png;base64,{b64}")
    doc.close()
    return uris


def pdf_extract_text_per_page(file_bytes: bytes) -> list[str]:
    """Fallback: extract raw text from each page using pypdf."""
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return [page.extract_text() or "" for page in reader.pages]


def docx_to_text(file_bytes: bytes) -> str:
    import docx as _docx
    doc = _docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


# ── Core: parse with NVIDIA nemoretriever-parse ───────────────────────────────

def extract_nemo_content(response_json: dict) -> str:
    """
    nvidia/nemoretriever-parse returns data via tool_calls (content is null).
    Extracts and joins all text elements from the tool_call arguments.
    """
    choice = response_json["choices"][0]["message"]
    # Primary: extract from tool_calls
    if choice.get("tool_calls"):
        try:
            raw_args = choice["tool_calls"][0]["function"]["arguments"]
            elements = json.loads(raw_args)  # list of [{bbox, text, type}]
            # elements may be [[{...}]] (nested) or [{...}]
            if elements and isinstance(elements[0], list):
                elements = elements[0]
            texts = [el.get("text", "") for el in elements if el.get("text")]
            return "\n".join(texts)
        except Exception:
            return str(choice["tool_calls"])
    # Fallback: plain content or reasoning_content
    return get_content(choice)


def parse_with_nvidia(image_uris: list[str], filename: str, raw_bytes: bytes) -> dict:
    """
    Send document page images to nvidia/nemoretriever-parse.
    Falls back to pypdf text extraction if the parser fails on a page.
    Returns structured JSON.
    """
    fallback_texts = pdf_extract_text_per_page(raw_bytes)
    all_parsed_text = []

    for idx, uri in enumerate(image_uris):
        payload = {
            "model": PARSE_MODEL,
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": uri}}],
            }],
            "max_tokens": 4096,
        }
        try:
            r = requests.post(NVIDIA_BASE, headers=HEADERS, json=payload, timeout=60)
            if r.status_code == 200:
                content = extract_nemo_content(r.json())
                if content.strip():
                    all_parsed_text.append({"page": idx + 1, "content": content})
                    continue
            # If parser failed or returned empty, use pypdf fallback
            fallback = fallback_texts[idx] if idx < len(fallback_texts) else ""
            all_parsed_text.append({"page": idx + 1, "content": fallback, "source": "pypdf_fallback"})
        except Exception:
            fallback = fallback_texts[idx] if idx < len(fallback_texts) else ""
            all_parsed_text.append({"page": idx + 1, "content": fallback, "source": "pypdf_fallback"})

    # Ask CHAT model to consolidate into a rich JSON structure
    combined_raw = "\n\n".join(
        f"[Page {p['page']}]\n{p.get('content', p.get('error', ''))}"
        for p in all_parsed_text
    )
    structure_prompt = f"""You are a document structuring AI.
The following is raw parsed text from document "{filename}" (extracted page by page):

{combined_raw[:25000]}

Return ONLY a valid JSON object with these keys:
- title (string)
- summary (string, 3-5 sentences)
- sections (list of {{heading, content}})
- tables (list of {{caption, rows}})
- key_entities (list of strings: names, orgs, locations)
- dates (list of strings)
- numbers (list of strings: monetary, statistics, metrics)
- full_text (string, concatenated clean text)

Return ONLY JSON, no markdown fences."""

    payload2 = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You produce structured JSON from document text. Output only valid JSON."},
            {"role": "user", "content": structure_prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    r2 = requests.post(NVIDIA_BASE, headers=HEADERS, json=payload2, timeout=90)
    r2.raise_for_status()
    raw_json = get_content(r2.json()["choices"][0]["message"])
    # Strip fences if present
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    try:
        return json.loads(raw_json)
    except Exception:
        return {"full_text": combined_raw, "parse_status": "raw_text", "pages": all_parsed_text}


def parse_text_doc_with_nvidia(text: str, filename: str) -> dict:
    """For TXT / DOCX: directly ask the LLM to produce the structured JSON."""
    prompt = f"""You are a document structuring AI.
Document name: "{filename}"
Full text:
{text[:25000]}

Return ONLY a valid JSON object with these keys:
- title (string)
- summary (string, 3-5 sentences)
- sections (list of {{heading, content}})
- tables (list of {{caption, rows}})
- key_entities (list of strings)
- dates (list of strings)
- numbers (list of strings)
- full_text (string)

Return ONLY JSON, no markdown fences."""

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You produce structured JSON. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    r = requests.post(NVIDIA_BASE, headers=HEADERS, json=payload, timeout=90)
    r.raise_for_status()
    raw = get_content(r.json()["choices"][0]["message"])
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"full_text": text, "parse_status": "raw_text"}


# ── Core: chat with JSON ───────────────────────────────────────────────────────

def chat_with_document_json(user_question: str, doc_json: dict, doc_name: str) -> str:
    """Answer user question using the structured JSON as context."""
    json_context = json.dumps(doc_json, indent=2)
    if len(json_context) > 20000:
        json_context = json_context[:20000] + "\n...[truncated]"

    system = f"""You are an expert Document Q&A Assistant.
The document "{doc_name}" has been parsed into this structured JSON:

{json_context}

Answer questions using ONLY information from this JSON.
Be precise — quote specific sections, numbers, or dates when relevant.
If the answer isn't in the document, say so clearly."""

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_question},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    r = requests.post(NVIDIA_BASE, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    content = get_content(r.json()["choices"][0]["message"])
    return content or "(No response from model)"


# ── Rich document display ─────────────────────────────────────────────────────

def render_doc_as_cards(doc_json: dict, doc_name: str):
    """Render a parsed JSON document as clean Streamlit components — no icons."""
    import pandas as pd

    title    = doc_json.get("title") or doc_name
    summary  = doc_json.get("summary", "")
    sections = doc_json.get("sections", [])
    tables   = doc_json.get("tables", [])
    entities = doc_json.get("key_entities", [])
    dates    = doc_json.get("dates", [])
    numbers  = doc_json.get("numbers", [])

    st.markdown(f"## {title}")
    if summary:
        st.info(summary)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sections",     len(sections))
    c2.metric("Tables",       len(tables))
    c3.metric("Key Entities", len(entities))
    c4.metric("Dates Found",  len(dates))

    # Key entities as tags
    if entities:
        st.markdown("**Key Entities**")
        st.markdown("  ".join(f"`{e}`" for e in entities))

    # Dates & numbers inline
    if dates:
        st.markdown("**Dates:** " + " · ".join(dates[:10]))
    if numbers:
        st.markdown("**Key Numbers:** " + " · ".join(numbers[:10]))

    # Tables as expandable dropdowns
    if tables:
        with st.expander("Tables", expanded=False):
            for tbl in tables:
                caption = tbl.get("caption", "Table")
                rows = tbl.get("rows", [])
                if rows:
                    try:
                        df = pd.DataFrame(rows)
                        st.markdown(f"**{caption}**")
                        st.dataframe(df, use_container_width=True)
                    except Exception:
                        st.markdown(f"**{caption}:** {rows}")

    # Sections as expandable dropdown
    if sections:
        with st.expander("Document Sections", expanded=False):
            for sec in sections:
                heading = sec.get("heading", "Section")
                content = sec.get("content", "")
                st.markdown(f"**{heading}**")
                st.markdown(content)
                st.divider()

    # Raw JSON as expandable dropdown
    with st.expander("Raw JSON", expanded=False):
        st.json(doc_json)


# ── Comparison helpers ────────────────────────────────────────────────────────

def _to_float(v):
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def _val(item):
    """Extract value safely whether it's a dict or scalar."""
    return item.get("value") if isinstance(item, dict) else item


def _conf(item):
    """Extract confidence safely if it exists."""
    return item.get("confidence") if isinstance(item, dict) else None


def extract_slim_po_from_json(doc_json: dict) -> dict:
    """
    Use the LLM to extract a slim PO schema (vendor, PO#, total, line_items)
    from an already-parsed doc_json. Includes Confidence Intervals (CI).
    """
    full_text = doc_json.get("full_text", "") or json.dumps(doc_json)
    prompt = f"""From the document text below, extract Purchase Order fields.
Return ONLY valid JSON (no markdown fences). For every field, estimate your confidence (0.0 to 1.0):
{{
  "vendor_name":  {{"value": "<string>", "confidence": <float>}},
  "po_number":    {{"value": "<string>", "confidence": <float>}},
  "grand_total":  {{"value": "<numeric string, e.g. '5000.00'>", "confidence": <float>}},
  "line_items": [
    {{"sku": {{"value": "<string>", "confidence": <float>}}, "quantity": {{"value": <number>, "confidence": <float>}}, "unit_price": {{"value": "<numeric string>", "confidence": <float>"}}}}
  ]
}}

DOCUMENT TEXT:
{full_text[:15000]}

Return ONLY valid JSON:"""
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You extract PO fields. Output only valid JSON."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }
    r = requests.post(NVIDIA_BASE, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    raw = get_content(r.json()["choices"][0]["message"])
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def compare_pos(primary: dict, historical: dict) -> list:
    """Compare two slim PO dicts; return list of change dicts."""
    changes = []
    
    p_total_raw = primary.get("grand_total")
    p_total = _to_float(_val(p_total_raw))
    h_total_raw = historical.get("grand_total")
    h_total = _to_float(_val(h_total_raw))
    
    if p_total is not None and h_total is not None and p_total != h_total:
        changes.append({"field": "grand_total",
                         "primary": str(_val(p_total_raw)),
                         "primary_conf": _conf(p_total_raw),
                         "historical": str(_val(h_total_raw)),
                         "historical_conf": _conf(h_total_raw),
                         "delta": f"{p_total - h_total:+.2f}"})

    hist_by_sku = {str(_val(i.get("sku", ""))).strip().upper(): i
                   for i in historical.get("line_items", [])}
    for p_item in primary.get("line_items", []):
        sku_raw = p_item.get("sku", "")
        sku = str(_val(sku_raw)).strip().upper()
        if not sku:
            continue
        h_item = hist_by_sku.get(sku)
        if h_item is None:
            changes.append({"field": "line_item", "sku": sku,
                             "primary": p_item, "historical": None,
                             "delta": "NEW — not in historical"})
            continue
            
        p_price_raw = p_item.get("unit_price")
        h_price_raw = h_item.get("unit_price")
        p_price = _to_float(_val(p_price_raw))
        h_price = _to_float(_val(h_price_raw))
        if p_price is not None and h_price is not None and p_price != h_price:
            changes.append({"field": "unit_price", "sku": sku,
                             "primary": str(_val(p_price_raw)),
                             "primary_conf": _conf(p_price_raw),
                             "historical": str(_val(h_price_raw)),
                             "historical_conf": _conf(h_price_raw),
                             "delta": f"{p_price - h_price:+.2f}"})
                             
        p_qty_raw = p_item.get("quantity")
        h_qty_raw = h_item.get("quantity")
        p_qty = _to_float(_val(p_qty_raw))
        h_qty = _to_float(_val(h_qty_raw))
        if p_qty is not None and h_qty is not None and p_qty != h_qty:
            changes.append({"field": "quantity", "sku": sku,
                             "primary": _val(p_qty_raw),
                             "primary_conf": _conf(p_qty_raw),
                             "historical": _val(h_qty_raw),
                             "historical_conf": _conf(h_qty_raw),
                             "delta": f"{p_qty - h_qty:+.0f}"})
                             
    primary_skus = {str(_val(i.get("sku", ""))).strip().upper()
                    for i in primary.get("line_items", [])}
    for sku, h_item in hist_by_sku.items():
        if sku and sku not in primary_skus:
            changes.append({"field": "line_item", "sku": sku,
                             "primary": None, "historical": h_item,
                             "delta": "REMOVED — in historical, missing from primary"})
    return changes


def fetch_from_sqlite(po_number: str, db_path: str = "po_database.db") -> dict:
    """Fetch historical PO from SQLite via JOIN query."""
    import sqlite3
    SQL = """
        SELECT h.po_number, h.vendor_name, h.grand_total,
               l.sku, l.quantity, l.unit_price
        FROM po_headers h
        JOIN po_line_items l ON h.po_number = l.po_number
        WHERE h.po_number = ?
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(SQL, (po_number,)).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError(f"PO number '{po_number}' not found in database.")
    first = rows[0]
    return {
        "vendor_name": first["vendor_name"],
        "po_number":   first["po_number"],
        "grand_total": first["grand_total"],
        "line_items":  [{"sku": r["sku"], "quantity": r["quantity"],
                         "unit_price": r["unit_price"]} for r in rows],
    }


def generate_variance_report(primary: dict, historical: dict, source: str) -> dict:
    """Generate variance report dict between two slim PO dicts."""
    from datetime import datetime
    changes = compare_pos(primary, historical)
    
    p_total_raw = primary.get("grand_total")
    p_total = _to_float(_val(p_total_raw))
    h_total_raw = historical.get("grand_total")
    h_total = _to_float(_val(h_total_raw))
    
    gtv = None
    if p_total is not None and h_total is not None:
        gtv = {"primary": str(_val(p_total_raw)),
               "primary_conf": _conf(p_total_raw),
               "historical": str(_val(h_total_raw)),
               "historical_conf": _conf(h_total_raw),
               "delta": f"{p_total - h_total:+.2f}"}
               
    return {
        "generated_at":        datetime.utcnow().isoformat() + "Z",
        "primary_po":          _val(primary.get("po_number")),
        "historical_po":       _val(historical.get("po_number")),
        "historical_source":   source,
        "variance_detected":   len(changes) > 0,
        "grand_total_variance": gtv,
        "changes_found":       changes,
    }


def render_variance_report(report: dict):
    """Render variance report as rich Streamlit UI."""
    import pandas as pd
    if report["variance_detected"]:
        st.error(f"**Variance Detected** — {len(report['changes_found'])} change(s) found")
    else:
        st.success("No variance detected — POs match perfectly.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Primary PO",    report.get("primary_po") or "—")
    c2.metric("Historical PO", report.get("historical_po") or "—")
    c3.metric("Source",        report.get("historical_source", "").upper())
    gtv = report.get("grand_total_variance")
    if gtv:
        p_c = gtv.get("primary_conf")
        h_c = gtv.get("historical_conf")
        p_c_str = f" (CI: {p_c:.0%})" if p_c is not None else ""
        h_c_str = f" (CI: {h_c:.0%})" if h_c is not None else ""
        st.markdown(
            f"**Grand Total:** `{gtv['primary']}{p_c_str}` → `{gtv['historical']}{h_c_str}`  "
            f"**Δ {gtv['delta']}**"
        )
    changes = report.get("changes_found", [])
    if changes:
        st.markdown("### Changes Found")
        rows = []
        for c in changes:
            p_conf = c.get("primary_conf")
            h_conf = c.get("historical_conf")
            p_conf_str = f" ({p_conf:.0%})" if p_conf is not None else ""
            h_conf_str = f" ({h_conf:.0%})" if h_conf is not None else ""
            rows.append({
                "Field": c.get("field"), "SKU": c.get("sku", "—"),
                "Primary": str(c.get("primary", "")) + p_conf_str,
                "Historical": str(c.get("historical", "")) + h_conf_str,
                "Delta": str(c.get("delta", ""))
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No line-item differences found.")
    st.download_button(
        "Download Variance Report",
        data=json.dumps(report, indent=2),
        file_name="variance_report.json",
        mime="application/json",
        use_container_width=True,
    )


# ── Streamlit App ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Document Parser Agent", layout="wide")
    apply_apple_style()

    st.title("Document Parser Agent")
    st.caption(f"Parser: `{PARSE_MODEL}`  •  Chat: `{CHAT_MODEL}`")

    # Session init
    for key, default in [
        ("docs_json", {}), ("messages", []), ("active_doc", None),
        ("compare_mode", None), ("compare_result", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Upload Document")
        uploaded = st.file_uploader(
            "PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        if uploaded and uploaded.name not in st.session_state.docs_json:
            st.info(f"Parsing **{uploaded.name}** with NVIDIA…")
            progress = st.progress(0, text="Reading file…")
            try:
                raw_bytes = uploaded.read()
                progress.progress(20, text="Sending to NVIDIA Parser…")

                fname = uploaded.name.lower()
                if fname.endswith(".pdf"):
                    image_uris = pdf_to_page_images(raw_bytes)
                    progress.progress(50, text=f"Parsed {len(image_uris)} page(s) → structuring JSON…")
                    doc_json = parse_with_nvidia(image_uris, uploaded.name, raw_bytes)
                elif fname.endswith(".docx"):
                    text = docx_to_text(raw_bytes)
                    progress.progress(50, text="Structuring DOCX → JSON…")
                    doc_json = parse_text_doc_with_nvidia(text, uploaded.name)
                else:  # .txt
                    text = raw_bytes.decode("utf-8", errors="replace")
                    progress.progress(50, text="Structuring TXT → JSON…")
                    doc_json = parse_text_doc_with_nvidia(text, uploaded.name)

                progress.progress(80, text="Saving JSON…")

                # Save JSON to disk
                safe_name = "".join(c if c.isalnum() else "_" for c in uploaded.name)
                json_path = OUTPUT_DIR / f"{safe_name}_parsed.json"
                with open(json_path, "w") as f:
                    json.dump(doc_json, f, indent=2)

                st.session_state.docs_json[uploaded.name] = {
                    "json": doc_json,
                    "json_path": str(json_path),
                }
                st.session_state.active_doc = uploaded.name
                progress.progress(100, text="Done!")

                # Rich doc-summary message in chat
                st.session_state.messages = [{
                    "role": "assistant",
                    "is_doc_summary": True,
                    "doc_json": doc_json,
                    "doc_name": uploaded.name,
                    "content": f"JSON saved to `{json_path}` — ask me anything!",
                }]
                st.success("Parsed successfully!")

            except Exception as e:
                st.error(f"Parsing failed: {e}")
                progress.empty()

        # Doc list
        if st.session_state.docs_json:
            st.divider()
            st.markdown("### Parsed Documents")
            for dname in st.session_state.docs_json:
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"`{dname}`")
                with cols[1]:
                    if st.button("Select", key=f"sel_{dname}"):
                        st.session_state.active_doc = dname
                        st.session_state.messages = []
                        st.rerun()

            # JSON preview
            if st.session_state.active_doc:
                active = st.session_state.active_doc
                info = st.session_state.docs_json[active]
                with st.expander("View JSON"):
                    st.json(info["json"])
                st.download_button(
                    "Download JSON",
                    data=json.dumps(info["json"], indent=2),
                    file_name=Path(info["json_path"]).name,
                    mime="application/json",
                    use_container_width=True,
                )

        # ── Second PDF uploader (appears when compare_mode == 'pdf') ───────
        if st.session_state.compare_mode == "pdf" and st.session_state.active_doc:
            st.divider()
            st.markdown("### Upload Comparison PDF")
            comp_file = st.file_uploader(
                "Historical PO PDF",
                type=["pdf"],
                key="compare_pdf_uploader",
                label_visibility="collapsed",
            )
            if comp_file:
                with st.spinner("Parsing comparison PDF & comparing…"):
                    try:
                        _active = st.session_state.active_doc
                        primary_json = st.session_state.docs_json[_active]["json"]
                        comp_bytes = comp_file.read()
                        comp_uris  = pdf_to_page_images(comp_bytes)
                        comp_json  = parse_with_nvidia(comp_uris, comp_file.name, comp_bytes)
                        primary_slim  = extract_slim_po_from_json(primary_json)
                        hist_slim     = extract_slim_po_from_json(comp_json)
                        report = generate_variance_report(primary_slim, hist_slim, "pdf")
                        
                        # Add historical PDF to sidebar dropdown & chat
                        st.session_state.docs_json[comp_file.name] = {
                            "json": comp_json,
                            "json_path": "memory",
                        }
                        st.session_state.messages.append({
                            "role": "assistant",
                            "is_doc_summary": True,
                            "doc_json": comp_json,
                            "doc_name": comp_file.name,
                            "content": f"Historical PDF `{comp_file.name}` parsed successfully. Variance report below.",
                        })

                        st.session_state.compare_result = report
                        st.session_state.compare_mode   = "done"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")

        st.divider()
        if st.button("Reset", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # ── Main: Chat area ───────────────────────────────────────────────────────
    if not st.session_state.docs_json:
        st.info("Upload a document in the sidebar to get started. It will be automatically parsed into a structured JSON.")
        return

    active = st.session_state.active_doc
    if not active:
        st.info("Select a document from the sidebar to chat.")
        return

    st.markdown(f"#### Chatting about: `{active}`")

    # Default greeting
    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Document **{active}** is loaded. Ask me anything!",
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_doc_summary"):
                render_doc_as_cards(msg["doc_json"], msg["doc_name"])
                st.caption(msg["content"])
            else:
                st.markdown(msg["content"])

    # ── Comparison Mode UI (below messages) ──────────────────────────────────
    cmode = st.session_state.compare_mode

    if cmode is None:
        st.divider()
        st.markdown("### Compare this PO against historical data?")
        c1, c2, _ = st.columns([2, 2, 3])
        with c1:
            if st.button("Compare with another PDF", use_container_width=True, key="btn_pdf"):
                st.session_state.compare_mode = "pdf"
                st.rerun()
        with c2:
            if st.button("Compare by PO Number (DB)", use_container_width=True, key="btn_db"):
                st.session_state.compare_mode = "po_number"
                st.rerun()

    elif cmode == "pdf":
        st.divider()
        st.info("Upload the historical PO PDF in the **sidebar** to run the comparison.")

    elif cmode == "po_number":
        st.divider()
        st.markdown("### Compare by PO Number")
        po_num   = st.text_input("Historical PO Number:", key="po_num_input")
        db_path  = st.text_input("SQLite DB path:", value="PO_Pipeline/po_database.db", key="db_path_input")
        if st.button("Fetch & Compare", key="compare_db_btn"):
            if not po_num.strip():
                st.warning("Please enter a PO number.")
            else:
                with st.spinner("Fetching from database and comparing…"):
                    try:
                        primary_json = st.session_state.docs_json[active]["json"]
                        primary_slim = extract_slim_po_from_json(primary_json)
                        historical   = fetch_from_sqlite(po_num.strip(), db_path=db_path)
                        report = generate_variance_report(primary_slim, historical, "sqlite")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Historical PO `{po_num}` fetched from Database successfully. Variance report below."
                        })
                        
                        st.session_state.compare_result = report
                        st.session_state.compare_mode   = "done"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")

    elif cmode == "done":
        st.divider()
        report = st.session_state.compare_result or {}
        if report:
            render_variance_report(report)
        if st.button("Run Another Comparison", key="reset_compare_btn"):
            st.session_state.compare_mode   = None
            st.session_state.compare_result = None
            st.rerun()

    # ── Chat input (always at bottom) ────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing JSON & generating answer…"):
                try:
                    doc_json = st.session_state.docs_json[active]["json"]
                    reply = chat_with_document_json(prompt, doc_json, active)
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
