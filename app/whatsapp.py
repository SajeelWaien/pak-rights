import os
import httpx

WHATSAPP_TOKEN = os.environ["WHATSAPP_ACCESS_TOKEN"]

GRAPH_VERSION = os.getenv(
    "WHATSAPP_GRAPH_API_VERSION",
    "v27.0"
)


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

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message
        }
    }

    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json()
    
    except httpx.HTTPStatusError as exc:
        print("Meta API Error:", exc.response.text)