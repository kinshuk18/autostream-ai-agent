# AutoStream — Social-to-Lead Agentic Workflow

A production-ready conversational AI agent built with LangGraph and Gemini 2.5 Flash that qualifies inbound leads for AutoStream, a SaaS video editing platform. The agent classifies user intent, answers questions via RAG from a local knowledge base, and captures qualified leads through a structured multi-turn collection flow.

## Project Structure

```text
autostream_agent/
├── agent.py             # LangGraph workflow — all nodes, routing, and CLI loop
├── knowledge_base.json  # RAG source — pricing plans, policies, FAQs
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key (get one for free at Google AI Studio)

### Step 1 — Clone or download the project

```bash
# If using git
git clone https://github.com/your-org/autostream-agent.git
cd autostream-agent

# Or simply place all four files in a single directory and cd into it
```

### Step 2 — Create and activate a virtual environment

```bash
# Create the venv
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set your Google API key

```bash
# macOS / Linux
export GOOGLE_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set GOOGLE_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your-api-key-here"
```

Tip: Add this export to your `~/.bashrc` or `~/.zshrc` to persist it across sessions.

### Step 5 — Run the agent

```bash
python agent.py
```

## Example Conversation

```text
=======================================================
  AutoStream AI Assistant
  Powered by LangGraph + Gemini 2.5 Flash
=======================================================
  Type 'quit' or 'exit' to end the session.

You: Hey there!
Alex: Hey! Welcome to AutoStream — we help content creators edit videos on autopilot.
      Ask me anything about our plans or features!

You: What's the difference between the Basic and Pro plans?
Alex: Great question! The Basic plan is $29/month and includes 10 videos/month at 720p.
      The Pro plan is $79/month with unlimited videos, 4K export, and AI captions —
      perfect if you're scaling your content. Want to get started?

You: I'm ready to sign up for Pro!
Alex: That's great to hear! I'd love to get you set up.
  Could you share your full name?

You: Jane Doe
Alex: Perfect, Jane! Now, what's the best email address where our team can reach you?

You: jane@example.com
Alex: Almost there! Which creator platform are you mainly active on?
      (e.g., YouTube, Instagram, TikTok, etc.)

You: YouTube

=======================================================
LEAD CAPTURED SUCCESSFULLY
=======================================================
  Name     : Jane Doe
  Email    : jane@example.com
  Platform : Youtube
=======================================================

Alex: Fantastic, Jane! You're all set. I've passed your details to our team
      and someone will reach out to your Youtube account shortly. Welcome to AutoStream!
```

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over a simple chain or single-prompt approach because it models the conversation as a stateful, cyclical graph rather than a linear pipeline. This is essential for two reasons: intent can shift across turns (a user might casually ask about pricing and then suddenly decide to sign up), and the lead collection sub-flow is inherently multi-turn by design, requiring memory of which fields have already been collected.

### How State Is Managed

The `AgentState` `TypedDict` is the single source of truth for the entire session. It is initialized once and passed into every graph invocation. LangGraph merges each node's return dictionary back into this shared state object using the `Annotated[list, operator.add]` reducer on messages (which appends rather than overwrites). All other fields — `intent`, `lead_name`, `lead_email`, `lead_platform`, `lead_captured`, `awaiting_lead` — are plain values that nodes overwrite as needed.

This architecture means that across 5–6 turns, the `lead_collection_node` can always inspect what it has already collected (`lead_name`, `lead_email`) and ask only for what remains, without the LLM having to “remember” prior turns through prompt stuffing alone. State is handled natively by LangGraph's `MemorySaver`.

## WhatsApp Deployment via Webhooks

Deploying this agent on WhatsApp requires three components: Meta's WhatsApp Business API, a webhook server, and a session store. Here's the full integration architecture:

### 1. Register a WhatsApp Webhook

In the Meta Developer Portal, create a WhatsApp Business App, generate a permanent access token, and register a webhook URL pointing to your server. Subscribe to the `messages` webhook field.

### 2. Build a Webhook Server (FastAPI example)

```python
from fastapi import FastAPI, Request
import httpx, json, os

app = FastAPI()
WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]
PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"]

# In-memory session store (use Redis in production)
sessions: dict = {}

@app.get("/webhook")
async def verify(request: Request):
    """Meta webhook verification handshake."""
    params = dict(request.query_params)
    if params.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
        return int(params["hub.challenge"])
    return {"error": "Invalid token"}, 403

@app.post("/webhook")
async def receive_message(request: Request):
    """Receive inbound WhatsApp messages and run the agent."""
    body = await request.json()

    # Extract sender and message text
    entry = body["entry"][0]["changes"][0]["value"]
    sender_id = entry["messages"][0]["from"]
    user_text = entry["messages"][0]["text"]["body"]

    # Load or initialize per-user session state
    if sender_id not in sessions:
        from agent import create_initial_state, build_graph
        sessions[sender_id] = {
            "state": create_initial_state(),
            "app": build_graph()
        }

    session = sessions[sender_id]
    from langchain_core.messages import HumanMessage, AIMessage

    # Append user message and invoke agent
    session["state"]["messages"] += [HumanMessage(content=user_text)]
    result = session["app"].invoke(session["state"])
    session["state"].update(result)

    # Extract latest AI reply
    ai_msgs = [m for m in session["state"]["messages"] if isinstance(m, AIMessage)]
    reply = ai_msgs[-1].content if ai_msgs else "Sorry, I couldn't process that."

    # Send reply back via WhatsApp Cloud API
    await send_whatsapp_message(sender_id, reply)
    return {"status": "ok"}

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=payload)
```

### 3. Session Persistence

Each WhatsApp user gets their own `AgentState` dictionary keyed by their phone number (`sender_id`). For production, replace the in-memory `sessions` dict with Redis (`redis-py`) so state survives server restarts and scales horizontally.

### 4. Deployment Checklist

| Step | Detail |
|---|---|
| Host the FastAPI server | Use Railway, Render, or AWS Lambda (must be HTTPS) |
| Set env variables | `GOOGLE_API_KEY`, `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `VERIFY_TOKEN` |
| Register webhook URL | `https://your-domain.com/webhook` in Meta Developer Portal |
| Use Redis for sessions | Replace `sessions: dict` with `redis.Redis()` client |
| Add message deduplication | Store processed `message_ids` to avoid double-processing |

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google API key (required) |
| `WHATSAPP_TOKEN` | Meta permanent access token (WhatsApp deployment only) |
| `PHONE_NUMBER_ID` | WhatsApp Business phone number ID (WhatsApp deployment only) |
| `VERIFY_TOKEN` | Your custom webhook verification token (WhatsApp deployment only) |

## License

MIT License — free to use, modify, and distribute.
