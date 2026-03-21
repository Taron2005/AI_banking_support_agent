from voice_ai_banking_support_agent.runtime.topic_classifier import TopicClassifier


def test_topic_classifier_in_scope_credit() -> None:
    clf = TopicClassifier()
    out = clf.classify("Ամերիաբանկը ինչ սպառողական վարկեր ունի")
    assert out.label == "credit"


def test_topic_classifier_out_of_scope() -> None:
    clf = TopicClassifier()
    out = clf.classify("What is the weather today?")
    assert out.label == "out_of_scope"


def test_topic_classifier_unsupported() -> None:
    clf = TopicClassifier()
    out = clf.classify("Which bank is best for loans?")
    assert out.label == "unsupported_request_type"


def test_topic_classifier_in_scope_english_deposit() -> None:
    clf = TopicClassifier()
    out = clf.classify("What are deposit options in ACBA?")
    assert out.label == "deposit"


def test_topic_classifier_ambiguous_when_only_weak_signals() -> None:
    clf = TopicClassifier()
    out = clf.classify("What is the interest?")
    assert out.label == "ambiguous"
    assert out.reason == "weak_topic_signals_only_need_explicit_product"

