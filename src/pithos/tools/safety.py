"""Heuristic command safety checker for the pithos tool calling system.

Evaluates a CLI command string before execution, returning a :class:`SafetyVerdict`
with one of three risk levels:

- ``SAFE``   — No known risk; continue normal execution flow.
- ``REVIEW`` — Potentially destructive (e.g. force-flags, sensitive paths); route
               through the user-confirmation step regardless of ``confirm`` mode.
- ``BLOCK``  — Shell injection or known-dangerous pattern detected; deny outright
               without calling subprocess.

Checks are applied in priority order:

1. Shell injection operators  → BLOCK
2. Known dangerous patterns   → BLOCK
3. Destructive argument flags → REVIEW
4. Sensitive path arguments   → REVIEW
5. Extra user-defined patterns (from config) evaluated last

All built-in pattern sets can be extended via the ``safety`` config section using
``extra_block_patterns`` and ``extra_review_patterns``.
"""

import re
from typing import Optional

from .models import RiskLevel, SafetyVerdict

# ---------------------------------------------------------------------------
# Default pattern sets
# ---------------------------------------------------------------------------

# Shell meta-characters that indicate operator injection.
# These allow chaining arbitrary commands regardless of which tool was called.
_SHELL_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r";"),  # command separator
    re.compile(r"&&"),  # AND chaining
    re.compile(r"\|\|"),  # OR chaining
    re.compile(r"(?<!\w)\|(?!\|)"),  # bare pipe (not ||)
    re.compile(r"`[^`]*`"),  # backtick sub-shell
    re.compile(r"\$\("),  # $(...) sub-shell
    re.compile(r"\$\{IFS\}", re.IGNORECASE),  # IFS fork-bomb vector
    re.compile(r">\s*/"),  # redirect to absolute path
    re.compile(r">\s*[A-Za-z]:"),  # redirect to Windows absolute path
    re.compile(r"2>&1"),  # stderr redirect (injection signal)
    re.compile(r"<\("),  # process substitution
    re.compile(r">\("),  # process substitution
]

# Patterns that indicate unambiguously dangerous operations.
# Matched against the full normalised command string (lower-cased, collapsed whitespace).
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    # Unix mass-delete
    re.compile(r"\brm\s+.*-[^\s]*r[^\s]*\s+/"),  # rm -r /... or rm -rf /
    re.compile(r"\brm\s+.*-[^\s]*f[^\s]*\s+[*~/]"),  # rm -rf * / ~
    re.compile(r"\brm\s+-rf\b"),  # rm -rf (bare)
    re.compile(r"\brm\s+-fr\b"),  # rm -fr
    # Piping internet content straight to a shell
    re.compile(r"\|\s*(ba)?sh\b"),  # | sh or | bash
    re.compile(r"\|\s*python[23]?\b"),  # | python
    re.compile(r"\|\s*perl\b"),  # | perl
    re.compile(r"\|\s*ruby\b"),  # | ruby
    # Low-level disk destruction
    re.compile(r"\bdd\b.*\bif=/dev/(zero|random|urandom|null)\b"),
    re.compile(r"\bmkfs\b"),  # format filesystem
    re.compile(r"\bfdisk\b.*-[^\s]*l"),  # fdisk list (mild but notable)
    re.compile(r"\bshred\b"),  # overwrite & delete
    re.compile(r"\bwipefs\b"),  # wipe filesystem signatures
    # Permission escalation
    re.compile(r"\bchmod\s+777\s+/"),  # chmod 777 /
    re.compile(r"\bchown\s+.*\s+/"),  # chown on root
    # Fork bomb
    re.compile(r":\(\)\s*\{"),  # :() { :|: & };:
    # Windows equivalents
    re.compile(r"\bformat\s+[A-Za-z]:"),  # format C:
    re.compile(r"\bdel\s+/[sq].*\\\*"),  # del /s /q path\*
    re.compile(r"\brd\s+/[sq]"),  # rd /s /q
]

# Flags that are often (but not always) destructive.
# Note: leading \b is not used before '--' because '-' is a non-word character
# and \b would not match the boundary between whitespace and '-'.
# Instead we anchor the end with \b (or (?![\w-])) to avoid partial matches.
_DESTRUCTIVE_FLAG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?<!\w)-[^\s]*r[^\s]*f|(?<!\w)-[^\s]*f[^\s]*r"),  # -rf / -fr combos
    re.compile(r"--force(?![\w-])"),
    re.compile(r"--delete(?![\w-])"),
    re.compile(r"--purge(?![\w-])"),
    re.compile(r"--no-verify(?![\w-])"),
    re.compile(r"--wipe(?![\w-])"),
    re.compile(r"--overwrite(?![\w-])"),
    re.compile(r"--hard(?![\w-])"),  # git reset --hard
    re.compile(r"--mirror(?![\w-])"),  # git push --mirror (destructive remote)
    re.compile(r"--allow-empty(?![\w-])"),
]

# Arguments that reference sensitive system paths.
_SENSITIVE_PATH_PATTERNS: list[re.Pattern[str]] = [
    # Unix
    re.compile(r"\s/etc/\S*"),
    re.compile(r"\s/dev/\S*"),
    re.compile(r"\s/sys/\S*"),
    re.compile(r"\s/proc/\S*"),
    re.compile(r"\s/boot/\S*"),
    re.compile(r"\s/var/log/\S*"),
    # Windows
    re.compile(r"\s[A-Za-z]:\\[Ww]indows\\\S*"),
    re.compile(r"\s[A-Za-z]:\\[Ss]ystem32\\\S*"),
    # Windows registry hives
    re.compile(r"\bHKLM\b", re.IGNORECASE),
    re.compile(r"\bHKCU\b", re.IGNORECASE),
    re.compile(r"\bHKEY_LOCAL_MACHINE\b", re.IGNORECASE),
    re.compile(r"\bHKEY_CURRENT_USER\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# CommandSafetyChecker
# ---------------------------------------------------------------------------


class CommandSafetyChecker:
    """Evaluates CLI commands for safety risks before execution.

    Args:
        config: The ``safety`` sub-dict from ``tool_config.yaml``.  All keys
            are optional; defaults are used when absent.

    Example config shape::

        safety:
          enabled: true
          block_shell_injection: true
          block_dangerous_patterns: true
          review_destructive_flags: true
          review_sensitive_paths: true
          extra_block_patterns: []
          extra_review_patterns: []
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.enabled: bool = cfg.get("enabled", True)
        self._block_shell_injection: bool = cfg.get("block_shell_injection", True)
        self._block_dangerous_patterns: bool = cfg.get("block_dangerous_patterns", True)
        self._review_destructive_flags: bool = cfg.get("review_destructive_flags", True)
        self._review_sensitive_paths: bool = cfg.get("review_sensitive_paths", True)

        # Compile user-supplied extra patterns
        self._extra_block: list[re.Pattern[str]] = [
            re.compile(p) for p in cfg.get("extra_block_patterns", [])
        ]
        self._extra_review: list[re.Pattern[str]] = [
            re.compile(p) for p in cfg.get("extra_review_patterns", [])
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, command: str) -> SafetyVerdict:
        """Evaluate *command* and return a :class:`SafetyVerdict`.

        Checks are applied in priority order (highest severity first).  The
        first match determines the returned verdict so that BLOCK always wins
        over REVIEW.

        Args:
            command: Full command string exactly as it would be passed to the
                shell, e.g. ``"python -c 'import os; os.remove(\\"/etc/passwd\\")'"``.

        Returns:
            A :class:`SafetyVerdict` describing the risk level and reason.
        """
        if not self.enabled:
            return SafetyVerdict(level=RiskLevel.SAFE, reason="Safety checker disabled")

        # Normalise for pattern matching (preserve case for path checks)
        cmd_lower = " ".join(command.split()).lower()
        cmd_orig = " ".join(command.split())

        # --- BLOCK: shell injection ---
        if self._block_shell_injection:
            verdict = self._check_patterns(
                cmd_orig,
                _SHELL_INJECTION_PATTERNS,
                RiskLevel.BLOCK,
                "Shell injection operator detected",
            )
            if verdict:
                return verdict

        # --- BLOCK: known dangerous patterns ---
        if self._block_dangerous_patterns:
            verdict = self._check_patterns(
                cmd_lower,
                _DANGEROUS_PATTERNS,
                RiskLevel.BLOCK,
                "Matches a known-dangerous command pattern",
            )
            if verdict:
                return verdict

        # --- BLOCK: extra user-defined block patterns ---
        if self._extra_block:
            verdict = self._check_patterns(
                cmd_orig,
                self._extra_block,
                RiskLevel.BLOCK,
                "Matches a user-defined block pattern",
            )
            if verdict:
                return verdict

        # --- REVIEW: destructive flags ---
        if self._review_destructive_flags:
            verdict = self._check_patterns(
                cmd_lower,
                _DESTRUCTIVE_FLAG_PATTERNS,
                RiskLevel.REVIEW,
                "Destructive argument flag detected",
            )
            if verdict:
                return verdict

        # --- REVIEW: sensitive paths ---
        if self._review_sensitive_paths:
            verdict = self._check_patterns(
                cmd_orig,
                _SENSITIVE_PATH_PATTERNS,
                RiskLevel.REVIEW,
                "Sensitive system path argument detected",
            )
            if verdict:
                return verdict

        # --- REVIEW: extra user-defined review patterns ---
        if self._extra_review:
            verdict = self._check_patterns(
                cmd_orig,
                self._extra_review,
                RiskLevel.REVIEW,
                "Matches a user-defined review pattern",
            )
            if verdict:
                return verdict

        return SafetyVerdict(level=RiskLevel.SAFE, reason="No risk patterns detected")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_patterns(
        text: str,
        patterns: list[re.Pattern[str]],
        level: RiskLevel,
        base_reason: str,
    ) -> Optional[SafetyVerdict]:
        """Return a verdict if any pattern matches *text*, else None."""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return SafetyVerdict(
                    level=level,
                    reason=f"{base_reason}: matched '{match.group(0)}' in command",
                )
        return None
