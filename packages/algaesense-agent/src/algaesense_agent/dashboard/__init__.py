"""On-demand plot generation: renders a campaign's fitted model as a PNG
image, ready to post directly into Slack.
"""

from algaesense_agent.dashboard.campaign_plot import render_campaign_fit_plot
from algaesense_agent.dashboard.plots import render_fit_curve_png

__all__ = ["render_campaign_fit_plot", "render_fit_curve_png"]
