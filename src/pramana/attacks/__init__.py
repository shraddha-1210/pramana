"""Layer 5 - attack library: reproduced published cryptanalysis, plus derived attacks.

The registry is the single source of truth for how many attacks exist. Any count
claimed in the docs, the deck or the UI must equal ``len(ATTACK_REGISTRY)``.
"""

from pramana.attacks.base import ATTACK_REGISTRY, AttackResult, ThreatClass, register
from pramana.attacks.blind_forgery import BlindForgery
from pramana.attacks.choi_2011 import Choi2011
from pramana.attacks.intercept_resend import InterceptResend
from pramana.attacks.replay import Replay

register(BlindForgery())
register(InterceptResend())
register(Choi2011())
register(Replay())

__all__ = [
    "ATTACK_REGISTRY",
    "AttackResult",
    "BlindForgery",
    "Choi2011",
    "InterceptResend",
    "Replay",
    "ThreatClass",
    "register",
]
