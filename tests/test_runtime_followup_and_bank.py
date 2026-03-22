from voice_ai_banking_support_agent.runtime.bank_detector import BankDetector
from voice_ai_banking_support_agent.runtime.followup_resolver import FollowUpResolver
from voice_ai_banking_support_agent.runtime.models import SessionState


def test_bank_detector_aliases() -> None:
    det = BankDetector()
    assert det.detect("What about ACBA deposits?") is not None
    assert det.detect("իսկ Ամերիայի դեպքում?") is not None


def test_bank_detector_detect_all_finds_multiple_banks_in_order() -> None:
    det = BankDetector()
    matches = det.detect_all("ACBA և Ameriabank ավանդ, հետո IDBank")
    keys = [m.bank_key for m in matches]
    assert keys == ["acba", "ameriabank", "idbank"]


def test_bank_detector_detect_all_single_when_one_named() -> None:
    det = BankDetector()
    assert [m.bank_key for m in det.detect_all("միայն Ամերիաբանկի վարկ")] == ["ameriabank"]


def test_bank_detector_armenian_spaced_ameria_bank() -> None:
    det = BankDetector()
    q = "ինչ ավանդներ կան Ամերիա բանկում այս տարի"
    assert [m.bank_key for m in det.detect_all(q)] == ["ameriabank"]


def test_bank_detector_uppercase_armenian_ameriabank() -> None:
    det = BankDetector()
    m = det.detect("ՎԱՐԿԵՐ ԱՄԵՐԻԱԲԱՆԿՈՒՄ")
    assert m is not None
    assert m.bank_key == "ameriabank"


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


def test_followup_resolver_interest_phrase_carries_deposit_topic() -> None:
    resolver = FollowUpResolver()
    state = SessionState(session_id="s4", last_topic="deposit", last_bank="ameriabank")
    out = resolver.resolve("իսկ տոկոսադրույքը?", state)
    assert out.used_followup_context is True
    assert "ավանդ" in out.resolved_query


def test_followup_resolver_does_not_inject_old_topic_when_user_names_new_product() -> None:
    resolver = FollowUpResolver()
    state = SessionState(session_id="s5", last_topic="deposit", last_bank="acba")
    out = resolver.resolve("իսկ վարկերը?", state)
    assert out.used_followup_context is True
    assert not out.resolved_query.strip().lower().startswith("ավանդ")


def test_followup_resolver_all_banks_phrase_drops_last_bank_prefix() -> None:
    resolver = FollowUpResolver()
    state = SessionState(session_id="s6", last_topic="deposit", last_bank="acba")
    out = resolver.resolve("իսկ բոլոր բանկերում ինչ ավանդներ կան", state)
    assert "acba" not in out.resolved_query.lower()


def test_followup_resolver_bank_pivot_does_not_reinject_previous_bank() -> None:
    aliases = {
        "ameriabank": ["ameriabank", "ameria", "ամերիա"],
        "acba": ["acba", "ակբա"],
        "idbank": ["idbank", "իդբանկ"],
    }
    resolver = FollowUpResolver(bank_aliases=aliases)
    state = SessionState(session_id="s3", last_topic="deposit", last_bank="acba")
    out = resolver.resolve("Ameriabank", state)
    assert out.used_followup_context is True
    assert "ավանդ" in out.resolved_query
    assert "acba" not in out.resolved_query.lower()

