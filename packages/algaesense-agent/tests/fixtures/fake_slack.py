"""A fake Slack channel for testing the human-in-the-loop approval flow
without a real Slack workspace or a real Hermes Agent running.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeSlackChannel:
    """Records every message "posted" to it, and plays back a scripted
    sequence of human replies -- standing in for a real Slack channel plus
    a human operator's responses, per the plan's own testing strategy
    ("a fake-Slack harness testing the approval-gate flow")."""

    scripted_replies: list[str] = field(default_factory=list)
    sent_messages: list[str] = field(default_factory=list)
    _reply_index: int = field(default=0, init=False)

    def post_message(self, text: str) -> None:
        self.sent_messages.append(text)

    def await_reply(self) -> str:
        """Return the next scripted human reply, in order -- standing in
        for the agent's own conversational turn-taking with a real Slack
        user."""
        reply = self.scripted_replies[self._reply_index]
        self._reply_index += 1
        return reply
