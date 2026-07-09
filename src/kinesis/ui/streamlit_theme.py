from __future__ import annotations

from base64 import b64encode
from pathlib import Path

import streamlit as st


def apply_kinesis_theme() -> None:
    st.markdown(f"<style>{KINESIS_CSS}</style>", unsafe_allow_html=True)


def render_brand_header(logo_path: Path) -> None:
    logo_src = _asset_data_uri(logo_path)
    st.markdown(
        f"""
        <div class="kinesis-brandbar">
          <img class="kinesis-brandbar__logo" src="{logo_src}" alt="Kinesis logo" />
          <div class="kinesis-brandbar__copy">
            <div class="kinesis-brandbar__name">Kinesis</div>
            <div class="kinesis-brandbar__meta">Movement research lab</div>
          </div>
          <div class="kinesis-brandbar__status">Research build</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _asset_data_uri(path: Path) -> str:
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


KINESIS_CSS = """
:root {
  --kinesis-bg: #f7f4ea;
  --kinesis-surface: #ffffff;
  --kinesis-surface-soft: #fbfaf6;
  --kinesis-ink: #18212f;
  --kinesis-muted: #637083;
  --kinesis-line: #dfe3dc;
  --kinesis-teal: #2a9d8f;
  --kinesis-teal-dark: #19776d;
  --kinesis-amber: #e9c46a;
  --kinesis-coral: #f26d5b;
}

.stApp {
  background: var(--kinesis-bg);
  color: var(--kinesis-ink);
}

[data-testid="stHeader"] {
  background: rgba(247, 244, 234, 0.92);
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
  border-left: 5px solid var(--kinesis-teal);
  border-radius: 8px;
  box-shadow: 0 18px 42px rgba(24, 33, 47, 0.08);
  display: flex;
  gap: 16px;
  margin: 0 0 1.25rem;
  padding: 16px 18px;
}

.kinesis-brandbar__logo {
  height: 58px;
  width: 58px;
}

.kinesis-brandbar__copy {
  min-width: 0;
}

.kinesis-brandbar__name {
  color: var(--kinesis-ink);
  font-size: 1.4rem;
  font-weight: 780;
  line-height: 1.1;
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
  background: var(--kinesis-teal);
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
  background: var(--kinesis-teal);
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--kinesis-surface-soft);
  border-color: rgba(42, 157, 143, 0.42);
  border-radius: 8px;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--kinesis-teal);
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
  background: var(--kinesis-teal);
  border-color: var(--kinesis-teal);
  color: white;
}

.stButton > button[kind="primary"]:hover {
  background: var(--kinesis-teal-dark);
  border-color: var(--kinesis-teal-dark);
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

  .kinesis-brandbar__logo {
    height: 50px;
    width: 50px;
  }

  .kinesis-brandbar__status {
    display: none;
  }

  h1 {
    font-size: 1.95rem;
  }
}
"""
