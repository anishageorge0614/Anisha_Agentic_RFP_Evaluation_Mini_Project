from rfp_eval.document_tool import DocumentPage
from rfp_eval.llm import LLMConfig, answer_question
from rfp_eval.rag import retrieve


def test_retrieval_is_relevant_and_deterministic():
    pages = [
        DocumentPage(1, "The implementation includes weekly milestones and a named project manager."),
        DocumentPage(2, "The security program uses AES-256 encryption and multi-factor authentication."),
    ]
    first = retrieve("security AES-256 encryption", pages)
    second = retrieve("security AES-256 encryption", pages)
    assert first == second
    assert first[0]["source_ref"] == "P2-C1"
    assert first[0]["page_number"] == 2


def test_offline_assistant_returns_retrieved_citations():
    passages = [
        {
            "citation_id": "Apex Systems | P2-C1",
            "source_ref": "P2-C1",
            "page_number": 2,
            "text": "The security program uses AES-256 encryption.",
        }
    ]
    response = answer_question(LLMConfig(), "What encryption is used?", passages)
    assert "AES-256" in response["answer"]
    assert response["citations"] == ["Apex Systems | P2-C1"]


def test_assistant_does_not_invent_answer_without_retrieved_sources():
    response = answer_question(LLMConfig(), "What is the renewal price?", [])
    assert response["citations"] == []
    assert "could not find" in response["answer"]