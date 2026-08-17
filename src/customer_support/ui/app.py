"""
Gradio chat interface for the customer support agentic graph.

Location: src/customer_support/ui/app.py

Each browser session gets its own thread_id (a UUID stored in gr.State,
generated once per session via demo.load). The checkpointer scopes
conversation history per thread_id, so two tabs never see each other's
messages -- but the preference-memory store (agents/memory.py) is keyed
by customer_id, not thread_id, so two sessions verified as the SAME real
customer correctly share saved preferences. That's intentional, not a
leak.
"""

import logging
import re
import time
import uuid

import gradio as gr
from langchain_core.messages import AIMessage
from langgraph.types import Command

from customer_support.config import PORT
from customer_support.graph.build import compiled_graph
from customer_support.identity import extract_email, extract_phone

logger = logging.getLogger(__name__)


def new_session():
    thread_id = str(uuid.uuid4())
    logger.info("new_session: thread_id=%s", thread_id)
    return thread_id, [], "Ready."


def _config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _parse_verification_reply(text: str) -> dict:
    """Pulls an email, a phone number, or a customer_id + last name out
    of free text. Email and phone take priority over a bare number --
    hitl_verify_node actually looks those up against the Customer
    table, whereas a bare customer_id is still trusted as-is (a stub,
    not real auth).

    Unlike hitl_verify_node's own pre-check against the message that
    triggers a turn, this runs on a direct reply to "what's your ID?"
    -- so a bare digit run is unambiguously the answer, not just a
    number that happens to appear in an arbitrary sentence, and is
    accepted as a customer_id without requiring an explicit label."""

    email = extract_email(text)
    if email:
        return {"email": email}

    phone = extract_phone(text)
    if phone:
        return {"phone": phone}

    match = re.search(r"\d+", text)
    customer_id = match.group() if match else None
    last_name = re.sub(r"[^A-Za-z\s]", "", text).strip() or None
    return {"customer_id": customer_id, "last_name": last_name}


def send_message(user_message, history, thread_id):
    if not user_message or not user_message.strip():
        yield history, "", thread_id, "Type a message first."
        return

    history = history + [{"role": "user", "content": user_message}]
    yield history, "", thread_id, "⏳ Processing..."

    config = _config_for(thread_id)
    start = time.monotonic()

    try:
        snap = compiled_graph.get_state(config)
        # Messages already in the thread before this turn -- used below
        # to isolate just what THIS turn added, since a mixed-intent
        # turn can run more than one sub-agent and each appends its own
        # final answer.
        prior_message_count = len(snap.values.get("messages", [])) if snap.values else 0

        if snap.next:
            # Graph is paused at hitl_verify from a previous turn -- resume
            # it rather than sending a fresh state dict.
            resume_payload = _parse_verification_reply(user_message)
            for _ in compiled_graph.stream(
                Command(resume=resume_payload), config=config, stream_mode="updates"
            ):
                pass
        elif not snap.values:
            # Brand-new thread: every plain (non-reducer) key nodes index
            # directly -- customer_id, customer_verified, intents -- must
            # be present from the first turn or a downstream node KeyErrors.
            for _ in compiled_graph.stream(
                {
                    "messages": [{"role": "user", "content": user_message}],
                    "session_id": thread_id,
                    "customer_id": None,
                    "customer_verified": False,
                    "intents": [],
                },
                config=config,
                stream_mode="updates",
            ):
                pass
        else:
            # Continuing an already-started thread. Deliberately omit
            # customer_id/customer_verified here: LangGraph leaves absent
            # keys untouched in the checkpoint rather than resetting them,
            # but if we included customer_verified=False on every turn
            # (mirroring the first-turn payload) we'd silently re-lock an
            # already-verified customer on every single message.
            for _ in compiled_graph.stream(
                {
                    "messages": [{"role": "user", "content": user_message}],
                    "session_id": thread_id,
                    "intents": [],
                },
                config=config,
                stream_mode="updates",
            ):
                pass
    except Exception:
        logger.exception("send_message: graph run failed")
        history = history + [
            {"role": "assistant", "content": "Sorry, something went wrong processing that."}
        ]
        yield history, "", thread_id, "⚠️ Error"
        return

    snap = compiled_graph.get_state(config)
    elapsed = time.monotonic() - start

    if snap.next:
        message = snap.tasks[0].interrupts[0].value.get(
            "message", "I need more information to continue."
        )
        history = history + [{"role": "assistant", "content": message}]
        yield history, "", thread_id, "⏳ Waiting for your input"
    else:
        # A mixed-intent turn (e.g. catalog + invoice) runs more than one
        # sub-agent, and each produces its own final AIMessage -- taking
        # only messages[-1] silently drops every answer but the last.
        # Collect every new, non-empty AIMessage this turn produced.
        new_messages = snap.values["messages"][prior_message_count:]
        answers = [
            m.content for m in new_messages if isinstance(m, AIMessage) and m.content
        ]
        answer = "\n\n".join(answers) if answers else "I didn't get a response for that."
        history = history + [{"role": "assistant", "content": answer}]
        yield history, "", thread_id, f"✅ Answered in {elapsed:.1f}s"


with gr.Blocks(title="Multi-Agent Music Store Customer Support") as demo:
    gr.Markdown("# Multi-Agent Music Store Customer Support")

    thread_id_state = gr.State()
    chatbot = gr.Chatbot(type="messages", height=500)
    status = gr.Markdown("Ready.")

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask about our catalog or your orders...",
            scale=8,
            show_label=False,
        )
        send_btn = gr.Button("Send", scale=1)

    new_conversation_btn = gr.Button("New Conversation")

    msg_box.submit(
        send_message,
        [msg_box, chatbot, thread_id_state],
        [chatbot, msg_box, thread_id_state, status],
    )
    send_btn.click(
        send_message,
        [msg_box, chatbot, thread_id_state],
        [chatbot, msg_box, thread_id_state, status],
    )
    new_conversation_btn.click(
        new_session, None, [thread_id_state, chatbot, status]
    )
    demo.load(new_session, None, [thread_id_state, chatbot, status])

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)