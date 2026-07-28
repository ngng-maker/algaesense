"""The AlgaeSense operator dashboard -- run with `streamlit run app.py`."""

# NOTE: this is a `streamlit run` entry point, so it uses plain `#` comments
# rather than this project's usual triple-quoted rationale blocks. Streamlit's
# magic-commands feature renders bare top-level strings as page content.
# See CLAUDE.md.

from pathlib import Path

import streamlit as st


# One page config for the whole app. Streamlit only honours the first call,
# and it has to happen in the entry script -- which is why neither of the
# pages below sets its own any more.
st.set_page_config(page_title="AlgaeSense", page_icon="🌱", layout="wide")

_HERE = Path(__file__).resolve().parent

# Ordered by how often each is reached rather than by where it falls in an
# experiment's lifecycle: calibration happens once at the start of a campaign,
# monitoring happens every day, and the first page listed is the one that
# opens by default.
pages = [
    st.Page(str(_HERE / "streamlit_app.py"), title="Monitoring", icon="📈", default=True),
    st.Page(str(_HERE / "zero_span_app.py"), title="Zero & span calibration", icon="🎯"),
]

# expanded=True is load-bearing, not cosmetic: st.navigation defaults it to
# False, which leaves the sidebar collapsed AND unmounted from the page --
# taking the monitoring page's own edge-URL and view controls with it, and
# leaving no visible way to reach calibration at all.
st.navigation(pages, expanded=True).run()
