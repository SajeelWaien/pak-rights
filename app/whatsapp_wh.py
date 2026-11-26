import os
import json
import hmac
import hashlib

from fastapi import (
    Request,
    BackgroundTasks,
    HTTPException
)

from fastapi.responses import PlainTextResponse

from fastapi import APIRouter

from app.whatsapp_store import init_whatsapp_db
from app.whatsapp_handler import process_whatsapp_message


VERIFY_TOKEN = os.environ["WHATSAPP_VERIFY_TOKEN"]
APP_SECRET = os.environ["WHATSAPP_APP_SECRET"]

router = APIRouter()
init_whatsapp_db()


def verify_signature(body: bytes, signature: str | None):

    if not signature:
        return False

    expected = "sha256=" + hmac.new(
        APP_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


@router.get("/webhooks/whatsapp")
async def verify_whatsapp_webhook(request: Request):

    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if (
        mode == "subscribe"
        and token
        and hmac.compare_digest(token, VERIFY_TOKEN)
        and challenge
    ):
        return PlainTextResponse(challenge)

    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhooks/whatsapp")
async def receive_whatsapp_message(
    request: Request,
    background_tasks: BackgroundTasks
):

    body = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256"
    )

    if not verify_signature(body, signature):
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    for entry in payload.get("entry", []):

        for change in entry.get("changes", []):

            if change.get("field") != "messages":
                continue

            value = change.get("value", {})

            phone_number_id = value.get(
                "metadata", {}
            ).get("phone_number_id")

            for message in value.get("messages", []):

                if message.get("type") != "text":
                    continue

                wa_id = message.get("from")
                message_id = message.get("id")
                question = message.get("text", {}).get("body", "")

                if not all([
                    phone_number_id,
                    wa_id,
                    message_id,
                    question
                ]):
                    continue

                background_tasks.add_task(
                    process_whatsapp_message,
                    phone_number_id,
                    wa_id,
                    message_id,
                    question
                )

    return {"status": "received"}