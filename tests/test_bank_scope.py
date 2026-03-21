from voice_ai_banking_support_agent.runtime.bank_scope import (
    query_implies_all_banks,
    query_implies_comparison,
    should_diversify_across_banks,
)


def test_query_implies_all_banks_covers_hy_and_en() -> None:
    assert query_implies_all_banks("ինչ բանկերում են ավանդներ")
    assert query_implies_all_banks("compare deposits across banks")
    assert not query_implies_all_banks("ամերիաբանկի ավանդները")


def test_query_implies_comparison_phrases() -> None:
    assert query_implies_comparison("ով ունի ավելի բարձր տոկոսադրույք")
    assert query_implies_comparison("համեմատի ավանդները acba և ameriabank")
    assert query_implies_comparison("compare acba vs idbank loans")
    assert not query_implies_comparison("idbank-ում ինչ ավանդներ կան")


def test_should_diversify_single_bank_false() -> None:
    assert should_diversify_across_banks(None) is True
    assert should_diversify_across_banks(frozenset({"acba", "idbank"})) is True
    assert should_diversify_across_banks(frozenset({"acba"})) is False
