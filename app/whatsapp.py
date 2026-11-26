import os
import httpx

WHATSAPP_TOKEN = os.environ["WHATSAPP_ACCESS_TOKEN"]

GRAPH_VERSION = os.getenv(
    "WHATSAPP_GRAPH_API_VERSION",
    "v27.0"
)

MAX_MESSAGE_LENGTH = 4096


def _split_message(message: str) -> list[str]:
    parts = []

    while len(message) > MAX_MESSAGE_LENGTH:
        window = message[:MAX_MESSAGE_LENGTH]
        split_at = window.rfind("\n\n")

        if split_at >= MAX_MESSAGE_LENGTH // 2:
            split_at += 2
        else:
            split_at = max(
                window.rfind("\n"),
                window.rfind(" ")
            )
            if split_at >= MAX_MESSAGE_LENGTH // 2:
                split_at += 1
            else:
                split_at = MAX_MESSAGE_LENGTH

        parts.append(message[:split_at])
        message = message[split_at:]

    if message or not parts:
        parts.append(message)

    return parts


def send_whatsapp_message(
    phone_number_id: str,
    recipient: str,
    message: str
):
    url = (
        f"https://graph.facebook.com/"
        f"{GRAPH_VERSION}/{phone_number_id}/messages"
    )

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    result = None
    try:
        for part in _split_message(message):
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": part
                }
            }

            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

    except httpx.HTTPStatusError as exc:
        print("Meta API Error:", exc.response.text)

    return result