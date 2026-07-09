from __future__ import annotations

from base64 import b64encode
from pathlib import Path

import streamlit as st


def apply_kinesis_theme() -> None:
    st.markdown(f"<style>{KINESIS_CSS}</style>", unsafe_allow_html=True)


def render_brand_header(wordmark_path: Path) -> None:
    wordmark_src = _asset_data_uri(wordmark_path)
    st.markdown(
        f"""
        <div class="kinesis-brandbar">
          <div class="kinesis-brandbar__copy">
            <img class="kinesis-brandbar__wordmark" src="{wordmark_src}" alt="Kinesis" />
            <div class="kinesis-brandbar__meta">Movement research lab</div>
          </div>
          <div class="kinesis-brandbar__status">Research build</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _asset_data_uri(path: Path) -> str:
    encoded = b64encode(path.read_bytes()).decode("ascii")
    mime_type = {
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime_type};base64,{encoded}"


KINESIS_CSS = """
:root {
  --kinesis-bg: #f6f9fc;
  --kinesis-surface: #ffffff;
  --kinesis-surface-soft: #f9fbff;
  --kinesis-ink: #06143d;
  --kinesis-muted: #66758e;
  --kinesis-line: #dbe5f1;
  --kinesis-blue: #2448ff;
  --kinesis-blue-dark: #092ccf;
  --kinesis-cyan: #12b8d7;
  --kinesis-navy: #020c2f;
}

.stApp {
  background: var(--kinesis-bg);
  color: var(--kinesis-ink);
}

[data-testid="stHeader"] {
  background: rgba(246, 249, 252, 0.92);
  backdrop-filter: blur(14px);
}

[data-testid="stToolbar"] {
  right: 1.25rem;
}

.block-container {
  max-width: 1180px;
  padding-top: 2rem;
  padding-bottom: 4rem;
}

.kinesis-brandbar {
  align-items: center;
  background: var(--kinesis-surface);
  border: 1px solid var(--kinesis-line);
  border-left: 5px solid var(--kinesis-cyan);
  border-radius: 8px;
  box-shadow: 0 18px 42px rgba(6, 20, 61, 0.08);
  display: flex;
  margin: 0 0 1.25rem;
  padding: 14px 18px;
}

.kinesis-brandbar__copy {
  min-width: 0;
}

.kinesis-brandbar__wordmark {
  display: block;
  height: 54px;
  width: auto;
}

.kinesis-brandbar__meta {
  color: var(--kinesis-muted);
  font-size: 0.92rem;
  margin-top: 3px;
}

.kinesis-brandbar__status {
  align-items: center;
  color: var(--kinesis-muted);
  display: flex;
  font-size: 0.88rem;
  gap: 8px;
  margin-left: auto;
  white-space: nowrap;
}

.kinesis-brandbar__status::before {
  background: var(--kinesis-cyan);
  border-radius: 999px;
  content: "";
  display: inline-block;
  height: 8px;
  width: 8px;
}

h1, h2, h3 {
  color: var(--kinesis-ink);
  letter-spacing: 0;
}

h1 {
  font-size: 2.35rem;
  font-weight: 790;
  margin-bottom: 0.35rem;
}

h2 {
  border-bottom: 1px solid var(--kinesis-line);
  font-size: 1.38rem;
  padding-bottom: 0.6rem;
}

h3 {
  font-size: 1.04rem;
  font-weight: 760;
}

[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--kinesis-line);
  gap: 4px;
}

[data-testid="stTabs"] [role="tab"] {
  color: var(--kinesis-muted);
  font-weight: 690;
  padding: 0.65rem 0.9rem;
}

[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--kinesis-ink);
}

[data-testid="stTabs"] [aria-selected="true"]::after {
  background: var(--kinesis-blue);
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--kinesis-surface-soft);
  border-color: rgba(36, 72, 255, 0.28);
  border-radius: 8px;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--kinesis-blue);
}

[data-testid="stExpander"] {
  background: var(--kinesis-surface);
  border: 1px solid var(--kinesis-line);
  border-radius: 8px;
  box-shadow: none;
}

[data-testid="stMetric"] {
  background: var(--kinesis-surface);
  border: 1px solid var(--kinesis-line);
  border-radius: 8px;
  padding: 0.85rem 0.95rem;
}

[data-testid="stMetricLabel"] {
  color: var(--kinesis-muted);
}

.stButton > button,
.stDownloadButton > button {
  border-radius: 8px;
  font-weight: 720;
}

.stButton > button[kind="primary"] {
  background: var(--kinesis-blue);
  border-color: var(--kinesis-blue);
  color: white;
}

.stButton > button[kind="primary"]:hover {
  background: var(--kinesis-blue-dark);
  border-color: var(--kinesis-blue-dark);
}

.stDownloadButton > button {
  border-color: rgba(24, 33, 47, 0.18);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--kinesis-line);
  border-radius: 8px;
  overflow: hidden;
}

video,
img {
  border-radius: 8px;
}

@media (max-width: 700px) {
  .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .kinesis-brandbar {
    align-items: flex-start;
    gap: 12px;
  }

  .kinesis-brandbar__wordmark {
    height: 46px;
    max-width: 70vw;
  }

  .kinesis-brandbar__status {
    display: none;
  }

  h1 {
    font-size: 1.95rem;
  }
}
"""
