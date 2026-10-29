from nlp.component import AdminNLP
from nlp.component.demo import DemoCopilot


def test_demo_fallback_builds_structured_id_card_request():
    out = AdminNLP(copilot=DemoCopilot(), pipeline="demo").analyze(
        "I lost my ID card. Roll no J054. What should I do?"
    )

    assert out.model.pipeline == "demo"
    assert out.intent.intent == "ID_CARDS"
    assert out.routing is not None
    assert out.routing.sub_intent == "DUPLICATE_ID_CARD"
    assert out.service_request is not None
    assert out.service_request.next_action
    assert out.sources
