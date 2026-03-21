from voice_ai_banking_support_agent.runtime.bank_detector import BankDetector
from voice_ai_banking_support_agent.runtime.followup_resolver import FollowUpResolver
from voice_ai_banking_support_agent.runtime.models import SessionState


def test_bank_detector_aliases() -> None:
    det = BankDetector()
    assert det.detect("What about ACBA deposits?") is not None
    assert det.detect("իսկ Ամերիայի դեպքում?") is not None


def test_followup_resolver_uses_last_context() -> None:
    resolver = FollowUpResolver()
    state = SessionState(session_id="s1", last_topic="deposit", last_bank="ameriabank")
    out = resolver.resolve("իսկ դոլարով?", state)
    assert out.used_followup_context is True
    assert "ameriabank" in out.resolved_query.lower()


def test_followup_resolver_city_carryover() -> None:
    resolver = FollowUpResolver()
    state = SessionState(session_id="s2", last_topic="branch", last_bank="idbank", last_city="գյումրի")
    out = resolver.resolve("իսկ այդ մասնաճյուղի հասցեն?", state)
    assert out.used_followup_context is True
    assert "գյումրի" in out.resolved_query.lower()

