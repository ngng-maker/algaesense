"""MCP server wrapping the jaxsr-calibration pipeline: fit models and
suggest next experiments, read/compute-only.
"""

from algaesense_agent.mcp_pipeline.pipeline import (
    CampaignNotFoundError,
    FitResult,
    SuggestionResult,
    default_basis_library,
    fit_symbolic_model,
    load_campaign_features,
    suggest_next_experiments,
)

__all__ = [
    "load_campaign_features",
    "fit_symbolic_model",
    "suggest_next_experiments",
    "default_basis_library",
    "FitResult",
    "SuggestionResult",
    "CampaignNotFoundError",
]
