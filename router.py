"""Router v2: intent classification for the single pipeline.

Pure functions, no I/O — tiers T0/T1 are testable without any model.

  T0 deterministic, whole-message, word-boundary:
       slash command (`^/\\w+`)                -> COMMAND
       greeting / thanks / ack / work-opener   -> CHAT   (Hinglish too)
       filename token + question word or '?'   -> FILE_QUESTION
  T1 scored rules (order matters: question check BEFORE verbs):
       polite-request exception ("can you make ...") counts as imperative
       question lead / trailing '?'            -> QUESTION (FILE_QUESTION when
                                                  it references a file)
       CODE_TASK requires BOTH a word-boundary action verb AND a concrete
       deliverable (filename/extension or a deliverable noun). Verb only or
       deliverable only -> UNCLEAR with ONE clarifying question.
       Score = verb (+1) + deliverable (+1) + filename (+1)
               - question lead (-2) - chat word present (-1).
       Grey zone -> T2: score 0 with 8+ words, conflicting signals (question
       lead + verb + deliverable), or verb-xor-deliverable with 6+ words.
  T2 grey zone only: ONE thinker call through the Phase 3 gateway, Ollama
     `format` schema {intent, confidence, deliverable, clarifying_question},
     temperature 0, last 3 turns as context, instructions in the user prompt.
     think:false only when the capability probe confirmed support; output is
     always run through strip_think. num_predict = 160 (+ think_budget when
     thinking cannot be disabled). One retry without `format` on invalid JSON.
     Below router.min_confidence -> UNCLEAR. T2 may return CODE_TASK only when
     `deliverable` is non-empty AND T1 saw at least a verb or a deliverable.

There is NO fall-through to CODE_TASK: anything unmatched is UNCLEAR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

INTENTS = ("CHAT", "CODE_TASK", "QUESTION", "FILE_QUESTION", "COMMAND", "UNCLEAR")

ROUTER_MIN_CONFIDENCE_DEFAULT = 0.7
ROUTER_MARKER = "ROUTER INTENT JSON:"
ROUTER_NUM_PREDICT = 160
DEFAULT_THINK_BUDGET = 512

# --- T0 phrase sets (whole-message, word-boundary) ----------------------------

_COMMAND_RE = re.compile(r"^\s*/\w+")

_GREETINGS = (
    "hello", "hi", "hey", "yo", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
)
_HINGLISH_GREETINGS = (
    "namaste", "namaskar", "kya haal", "kya haal hai", "kaise ho", "satsriakal",
)
_THANKS = (
    "thanks a lot for that", "thanks a lot", "thank you so much", "thank you",
    "thanks for that", "thanks", "thank u", "thx", "ty", "much appreciated",
    "appreciated", "dhanyavaad", "shukriya",
)
_ACKS = (
    "ok", "okay", "k", "kk", "got it", "understood", "i see", "makes sense",
    "cool", "great", "nice", "perfect", "awesome", "sounds good", "alright",
    "noted", "fine", "roger", "yep", "yeah", "yes", "no", "hmm", "hm",
    "interesting", "right", "theek hai", "achha", "sab theek",
    "bye", "goodbye", "see you",
)
_FILLERS = ("hey", "there", "bro", "please", "again", "everyone", "so much", "for that")
_OPENERS = (
    "lets work together", "let's work together", "lets work", "let's work",
    "lets get started", "let's get started", "lets start", "let's start",
    "lets continue", "let's continue", "lets go", "let's go", "shall we",
    "start", "continue", "go on",
    "chalo shuru karte hain", "chalo shuru karte hai", "chalo",
    "shuru karte hain", "shuru karte hai",
    "aage badhte hain", "aage badhte hai", "aage badho",
)
_HINGLISH_ALL = _HINGLISH_GREETINGS + (
    "dhanyavaad", "shukriya", "theek hai", "achha", "sab theek",
    "chalo shuru karte hain", "chalo shuru karte hai", "chalo",
    "shuru karte hain", "shuru karte hai", "aage badho",
)

# Longest-first so "thank you so much" beats "thank you" when stripping.
_CHAT_PHRASES = tuple(sorted(
    set(_GREETINGS + _HINGLISH_GREETINGS + _THANKS + _ACKS + _FILLERS + _OPENERS),
    key=len, reverse=True,
))
_CHAT_WORD_RE = re.compile(
    "|".join(rf"\b{re.escape(p)}\b" for p in sorted(
        set(_GREETINGS + _HINGLISH_GREETINGS + _THANKS + _ACKS), key=len, reverse=True
    ))
)
_HINGLISH_RE = re.compile(
    "|".join(rf"\b{re.escape(p)}\b" for p in sorted(set(_HINGLISH_ALL), key=len, reverse=True))
)

_FILENAME_RE = re.compile(
    r"\b[\w.-]+\.(?:py|js|ts|html?|css|json|md|txt|ya?ml|toml|sh|c|cpp|java|go|rs)\b"
)
_T0_QUESTION_WORD_RE = re.compile(
    r"\b(?:what|why|how|where|when|who|which|whose|is|are|was|were|does|do|did|"
    r"should|explain|show|describe|summari[sz]e|tell)\b"
)

# --- T1 sets -------------------------------------------------------------------

_ACTION_VERBS = (
    "set up",  # two-word verb first (longest-first regex)
    "make", "create", "build", "write", "add", "fix", "update", "change",
    "edit", "implement", "generate", "refactor", "rename", "convert",
    "remove", "delete", "scaffold", "port", "optimize",
)
_DELIVERABLE_NOUNS = (
    "page", "site", "website", "script", "function", "api", "endpoint",
    "class", "test", "tests", "app", "component", "cli", "bot", "form",
    "button", "route", "module", "game", "calculator",
)
_VERB_ALT = "|".join(re.escape(v) for v in sorted(_ACTION_VERBS, key=len, reverse=True))
_VERB_RE = re.compile(rf"\b(?:{_VERB_ALT})\b")
_NOUN_RE = re.compile(
    "|".join(rf"\b{re.escape(n)}\b" for n in sorted(_DELIVERABLE_NOUNS, key=len, reverse=True))
)
_QLEAD_RE = re.compile(
    r"^(?:what|why|how|where|when|who|which|whose|is|are|was|were|"
    r"should|explain|tell me|show me what)\b"
)
# Auxiliary starters only count as a question lead with a subject pronoun
# ("does this work"), never for imperatives ("do something cool").
_QLEAD_AUX_RE = re.compile(r"^(?:do|does|did)\s+(?:you|i|we|they|he|she|it|this|that|these|those)\b")
_POLITE_RE = re.compile(rf"^(?:can|could|would|will) you (?:please )?(?:{_VERB_ALT})\b")
_IMPERATIVE_RE = re.compile(rf"^(?:please )?(?:{_VERB_ALT})\b")
_FILE_REF_RE = re.compile(
    r"\b(?:my|this|the) files?\b"
    r"|\b(?:my|this|these|the) (?:dir|directory|folder|workspace|project)\b"
    r"|\b(?:what|which) files\b"
    r"|\bthese files\b"
    r"|\bin (?:this|my) (?:dir|directory|folder)\b"
    r"|\b(?:list|show)(?: me)? (?:the )?files\b"
)
_FILE_CMD_RE = re.compile(r"\b(?:summari[sz]e|describe|explain|show|list)\b")


# --- result types ----------------------------------------------------------------

@dataclass
class RouteResult:
    intent: str
    tier: int                       # 0, 1, or 2 (2 = grey zone was reached)
    verb: str | None = None
    deliverable: str | None = None
    confidence: float = 1.0
    clarifying_question: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    raw: str = ""                   # T2 raw classifier output (post strip_think)

    def __str__(self) -> str:
        bits = [f"tier={self.tier}", f"intent={self.intent}"]
        if self.verb:
            bits.append(f"verb={self.verb}")
        if self.deliverable:
            bits.append(f"deliverable={self.deliverable}")
        bits.append(f"confidence={self.confidence:.2f}")
        if self.clarifying_question:
            bits.append(f"clarify={self.clarifying_question!r}")
        return "RouteResult(" + ", ".join(bits) + ")"


@dataclass
class _Signals:
    verb: str | None
    noun: str | None
    filename: str | None
    file_ref: bool
    question_lead: bool
    imperative: bool
    chat_word: bool
    words: int
    score: int

    @property
    def deliverable(self) -> str | None:
        """A filename/extension also satisfies the deliverable requirement."""
        return self.noun or self.filename

    def as_scores(self) -> dict[str, float]:
        return {
            "verb": 1.0 if self.verb else 0.0,
            "deliverable": 1.0 if self.deliverable else 0.0,
            "filename": 1.0 if self.filename else 0.0,
            "question_lead": -2.0 if (self.question_lead and not self.imperative) else 0.0,
            "chat_word": -1.0 if self.chat_word else 0.0,
            "total": float(self.score),
        }


# --- normalization ---------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.!,\s]+$")


def _norm(text: str) -> str:
    """Lowercase, trim, collapse whitespace, strip trailing punctuation."""
    return _TRAILING_PUNCT_RE.sub("", _WS_RE.sub(" ", text.strip().lower()))


def _strip_chat_phrases(msg: str) -> str:
    """Remove leading/trailing chat phrases (+ separators) until fixpoint.
    Returns "" only when the WHOLE message consisted of chat material."""
    s = msg
    for _ in range(24):  # hard bound; every iteration shrinks s or returns
        t = s.strip(" ,.!~")
        changed = t != s
        s = t
        if not s:
            return ""
        for p in _CHAT_PHRASES:
            if s == p:
                return ""
            if s.startswith(p) and s[len(p):len(p) + 1] in (" ", ",", "."):
                s = s[len(p):]
                changed = True
                break
            if s.endswith(p) and s[-len(p) - 1:-len(p)] in (" ", ",", "."):
                s = s[: -len(p)]
                changed = True
                break
        if not changed:
            return s
    return s


# --- T0 -------------------------------------------------------------------------

def _t0(text: str) -> RouteResult | None:
    msg = _norm(text)
    if not msg:
        return RouteResult("UNCLEAR", 0, confidence=0.0,
                           clarifying_question="Say that again?")

    # Slash command -> COMMAND (whole message).
    if _COMMAND_RE.match(text):
        return RouteResult("COMMAND", 0, confidence=1.0)

    had_q = "?" in text
    has_filename = bool(_FILENAME_RE.search(msg))

    # filename + question word/'?' -> FILE_QUESTION
    if has_filename and (had_q or _T0_QUESTION_WORD_RE.search(msg)):
        m = _FILENAME_RE.search(msg)
        return RouteResult("FILE_QUESTION", 0, deliverable=m.group(0) if m else None,
                           confidence=1.0)

    # CHAT: the whole normalized message is chat phrases only (a greeting may
    # be followed by a work-opener: "hello lets work together" strips to "").
    if _CHAT_WORD_RE.search(msg) or msg in _OPENERS:
        if _strip_chat_phrases(msg) == "":
            return RouteResult("CHAT", 0, confidence=1.0)
    return None


# --- T1 -------------------------------------------------------------------------

def _analyze(text: str) -> _Signals:
    msg = _norm(text)
    had_q = "?" in text
    verb_m = _VERB_RE.search(msg)
    noun_m = _NOUN_RE.search(msg)
    file_m = _FILENAME_RE.search(msg)
    imperative = bool(_POLITE_RE.match(msg) or _IMPERATIVE_RE.match(msg))
    question_lead = bool(_QLEAD_RE.match(msg) or _QLEAD_AUX_RE.match(msg)) or (had_q and not imperative)
    chat_word = bool(_CHAT_WORD_RE.search(msg))
    verb = verb_m.group(0) if verb_m else None
    noun = noun_m.group(0) if noun_m else None
    filename = file_m.group(0) if file_m else None
    file_ref = bool(filename) or bool(_FILE_REF_RE.search(msg))
    score = (
        (1 if verb else 0)
        + (1 if (noun or filename) else 0)
        + (1 if filename else 0)
        - (2 if (question_lead and not imperative) else 0)
        - (1 if chat_word else 0)
    )
    return _Signals(
        verb=verb, noun=noun, filename=filename, file_ref=file_ref,
        question_lead=question_lead, imperative=imperative, chat_word=chat_word,
        words=len(msg.split()), score=score,
    )


def _clarify_for(sig: _Signals) -> str:
    """ONE clarifying question built from the missing half."""
    if sig.verb:
        return f"What should I {sig.verb}, and in which file?"
    if sig.deliverable:
        return f"What should I do with the {sig.deliverable} — create or change it, and which file?"
    return "What should I build or change, and in which file?"


def _t1_decide(sig: _Signals, msg: str) -> RouteResult | None:
    """T1 decision; None means grey zone (hand to T2)."""
    scores = sig.as_scores()
    # Conflicting signals (question lead + verb + deliverable) -> grey zone.
    if sig.question_lead and sig.verb and sig.deliverable:
        return None
    # Question check BEFORE verbs.
    if sig.question_lead:
        if sig.file_ref:
            return RouteResult("FILE_QUESTION", 1, deliverable=sig.filename,
                               confidence=0.95, scores=scores)
        return RouteResult("QUESTION", 1, confidence=0.9, scores=scores)
    # Imperative-style file commands with a workspace reference ("summarize my
    # files", "list files") are FILE_QUESTIONs, not code tasks.
    if sig.file_ref and sig.verb is None and _FILE_CMD_RE.search(msg):
        return RouteResult("FILE_QUESTION", 1, deliverable=sig.filename,
                           confidence=0.9, scores=scores)
    if sig.verb and sig.deliverable:
        return RouteResult("CODE_TASK", 1, verb=sig.verb,
                           deliverable=sig.deliverable, confidence=0.85,
                           scores=scores)
    if sig.verb or sig.deliverable:
        if sig.words >= 6:
            return None  # xor with enough context -> grey zone
        return RouteResult("UNCLEAR", 1, verb=sig.verb,
                           deliverable=sig.deliverable, confidence=0.5,
                           clarifying_question=_clarify_for(sig), scores=scores)
    if sig.score == 0 and sig.words >= 8:
        return None  # long but signal-free -> grey zone
    # Short and unmatched: UNCLEAR, never CODE_TASK.
    return RouteResult("UNCLEAR", 1, confidence=0.4, scores=scores,
                       clarifying_question=_clarify_for(sig))


def _t1(text: str) -> tuple[RouteResult | None, _Signals]:
    msg = _norm(text)
    sig = _analyze(text)
    return _t1_decide(sig, msg), sig


# --- T2 -------------------------------------------------------------------------

async def _t2(
    text: str,
    generate,
    thinker_model: str,
    history: list[str] | None,
    min_confidence: float,
    sig: _Signals,
    think_supported: bool | None = None,
    think_budget: int = DEFAULT_THINK_BUDGET,
) -> RouteResult:
    """Grey-zone classifier: ONE thinker call (plus one no-format retry)."""
    import json as _json

    from agent_core.schemas import RouterIntent
    from ollama_client import strip_think

    context = ""
    if history:
        context = "Recent turns:\n" + "\n".join(history[-3:]) + "\n\n"
    prompt = (
        ROUTER_MARKER + "\n"
        "Classify the user's message into exactly one intent: CHAT, CODE_TASK, "
        "QUESTION, FILE_QUESTION, COMMAND, UNCLEAR.\n"
        "CODE_TASK only when the user clearly wants code or files created or "
        "changed. QUESTION = a general question. FILE_QUESTION = asks about a "
        "file. CHAT = small talk. UNCLEAR = cannot tell.\n"
        "Reply with ONLY one JSON object: {\"intent\": \"...\", "
        "\"confidence\": 0.0, \"deliverable\": \"...\", "
        "\"clarifying_question\": \"...\"}\n\n"
        + context
        + "MESSAGE: " + text
    )
    num_predict = ROUTER_NUM_PREDICT + (0 if think_supported else think_budget)
    think_param = False if think_supported else None  # never send unless probed

    raw = ""
    data: dict[str, Any] = {}
    for attempt in range(2):
        # All instructions live in the user turn (inline_system for reasoning
        # distill models); no separate system prompt is sent.
        kwargs: dict[str, Any] = {
            "system": "",
            "temperature": 0.0,
            "num_predict": num_predict,
            "think": think_param,
        }
        if attempt == 0:
            kwargs["_format_schema"] = RouterIntent.model_json_schema()
        else:
            prompt = prompt + "\nReturn ONLY the JSON object."
        raw = await generate(thinker_model, prompt, **kwargs)
        answer, _think, _chars = strip_think(raw)
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        try:
            data = _json.loads(match.group(0)) if match else {}
        except _json.JSONDecodeError:
            data = {}
        if data:
            break
    intent = str(data.get("intent", "UNCLEAR")).upper()
    if intent not in INTENTS:
        intent = "UNCLEAR"
    confidence = float(data.get("confidence", 0.0) or 0.0)
    deliverable = str(data.get("deliverable") or "") or None
    clarify = str(data.get("clarifying_question", "") or "")
    base = RouteResult(intent, 2, verb=sig.verb,
                       deliverable=deliverable or sig.deliverable,
                       confidence=confidence, clarifying_question=clarify,
                       scores=sig.as_scores(), raw=raw[:4_000])

    # Safety guard: T2 may return CODE_TASK only with a non-empty deliverable
    # AND at least a verb or deliverable seen at T1. Never promote from nothing.
    if intent == "CODE_TASK" and (
        not deliverable or (sig.verb is None and sig.deliverable is None)
    ):
        base.intent = "UNCLEAR"
        base.confidence = min(confidence, 0.5)
        base.clarifying_question = clarify or _clarify_for(sig)
        return base
    if confidence < min_confidence:
        base.intent = "UNCLEAR"
        base.clarifying_question = clarify or "Could you clarify what you'd like?"
    return base


# --- canned CHAT replies (zero model calls by default) --------------------------

_CHAT_REPLY_EN = "Hey! What would you like to build or fix today?"
_CHAT_REPLY_HING = "Namaste! Bataiye — aaj kya banayein ya theek karein?"


def canned_chat_reply(text: str) -> str:
    """Deterministic small-talk reply (English or Hinglish), no model call."""
    if _HINGLISH_RE.search(_norm(text)):
        return _CHAT_REPLY_HING
    return _CHAT_REPLY_EN


# --- public entries ----------------------------------------------------------------

def route_rules_only(text: str) -> RouteResult:
    """T0 -> T1 only, no model call. Grey zone (or classifier failure) is
    UNCLEAR — never CODE_TASK."""
    r = _t0(text)
    if r is not None:
        return r
    r, _sig = _t1(text)
    if r is not None:
        return r
    return RouteResult("UNCLEAR", 2, confidence=0.0,
                       clarifying_question="Could you tell me more — create/change code, or answer a question?")


async def route(
    text: str,
    generate=None,
    thinker_model: str = "",
    history: list[str] | None = None,
    min_confidence: float = ROUTER_MIN_CONFIDENCE_DEFAULT,
    think_supported: bool | None = None,
    think_budget: int = DEFAULT_THINK_BUDGET,
) -> RouteResult:
    """Classify `text`. T0 -> T1 -> (T2 only when a `generate` seam is given
    and the message is genuinely grey). Never falls through to CODE_TASK."""
    r = _t0(text)
    if r is not None:
        return r
    r, sig = _t1(text)
    if r is not None:
        return r
    if generate is not None and thinker_model:
        try:
            return await _t2(text, generate, thinker_model, history,
                             min_confidence, sig, think_supported, think_budget)
        except Exception:
            pass  # classifier failed -> UNCLEAR, never guess
    # Grey zone with no classifier available: UNCLEAR, never CODE_TASK.
    return RouteResult("UNCLEAR", 2, confidence=0.0, scores=sig.as_scores(),
                       clarifying_question="Could you tell me more — create/change code, or answer a question?")


async def diagnose(
    text: str,
    generate=None,
    thinker_model: str = "",
    history: list[str] | None = None,
    min_confidence: float = ROUTER_MIN_CONFIDENCE_DEFAULT,
    think_supported: bool | None = None,
    think_budget: int = DEFAULT_THINK_BUDGET,
) -> dict[str, Any]:
    """For `/router <text>`: tier, matched verb/deliverable, scores, whether T2
    ran, and the raw classifier JSON (post strip_think)."""
    r = _t0(text)
    if r is not None:
        t2_ran = False
    else:
        r, sig = _t1(text)
        if r is not None:
            t2_ran = False
        elif generate is not None and thinker_model:
            try:
                r = await _t2(text, generate, thinker_model, history,
                              min_confidence, sig, think_supported, think_budget)
                t2_ran = True
            except Exception:
                r = RouteResult("UNCLEAR", 2, confidence=0.0, scores=sig.as_scores(),
                                clarifying_question="Could you tell me more — create/change code, or answer a question?")
                t2_ran = True
        else:
            r = RouteResult("UNCLEAR", 2, confidence=0.0, scores=sig.as_scores(),
                            clarifying_question="Could you tell me more — create/change code, or answer a question?")
            t2_ran = False
    return {
        "text": text,
        "intent": r.intent,
        "tier": r.tier,
        "matched_verb": r.verb,
        "matched_deliverable": r.deliverable,
        "scores": r.scores,
        "confidence": r.confidence,
        "clarifying_question": r.clarifying_question,
        "t2_ran": t2_ran,
        "t2_raw": r.raw,
    }
