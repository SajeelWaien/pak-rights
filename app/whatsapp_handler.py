import logging

from app.rag import ask_stream

from app.whatsapp import send_whatsapp_message

from app.whatsapp_store import (
    claim_message,
    get_or_create_conversation,
    create_new_conversation
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "paranine/gpt-oss-120b(high)"


def process_whatsapp_message(
    phone_number_id: str,
    wa_id: str,
    message_id: str,
    question: str
):

    # Prevent duplicate processing
    if not claim_message(message_id):
        return

    # Handle a new conversation command
    if question.strip().lower() == "/new":

        create_new_conversation(phone_number_id, wa_id)

        send_whatsapp_message(
            phone_number_id,
            wa_id,
            "Started a new legal discussion. How can I help?"
        )

        return

    conversation_id = get_or_create_conversation(
        phone_number_id,
        wa_id
    )

    try:
        send_whatsapp_message(
            phone_number_id,
            wa_id,
            "I'm processing your question. I'll send the answer shortly."
        )

        answer_parts = []

        sources = []

        for event in ask_stream(
            question=question,
            model_name=DEFAULT_MODEL,
            conversation_id=conversation_id
        ):

            if event.get("type") == "token":
                answer_parts.append(
                    event.get("text", "")
                )

            elif event.get("type") == "sources":
                sources = event.get("sources", [])

            elif event.get("type") == "error":
                raise RuntimeError(event.get("message"))

        answer = "".join(answer_parts).strip()

        if sources:
            source_lines = []

            for source in sources:
                source_lines.append(
                    f"• {source.get('file', 'Legal document')} "
                    f"(Page {source.get('page', 'N/A')})"
                )

            answer += "\n\nSources:\n" + "\n".join(source_lines)

        send_whatsapp_message(
            phone_number_id,
            wa_id,
            answer or "I couldn't generate an answer. Please try again."
        )

    except Exception:

        logger.exception("WhatsApp RAG processing failed")

        send_whatsapp_message(
            phone_number_id,
            wa_id,
            "Sorry, I couldn't process your question. Please try again."
        )