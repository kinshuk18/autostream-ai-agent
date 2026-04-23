"""
AutoStream Social-to-Lead Agentic Workflow
==========================================
A LangGraph-powered conversational AI agent for AutoStream SaaS.
Handles intent classification, RAG-based Q&A, and lead capture.
"""

import json
import os
import re
from typing import Optional
from typing_extensions import TypedDict, Annotated
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. KNOWLEDGE BASE LOADER (RAG Source)

def load_knowledge_base(path: str = "knowledge_base.json") -> dict:
    """Load the local JSON knowledge base for RAG retrieval."""
    with open(path, "r") as f:
        return json.load(f)

def retrieve_knowledge(query: str, kb: dict) -> str:
    """
    Simple keyword-based RAG retrieval from the JSON knowledge base.
    In production, this would use vector embeddings (e.g., FAISS/Chroma).
    Returns a formatted string of relevant KB sections to inject into the prompt.
    """
    query_lower = query.lower()
    relevant_chunks = []

    # --- Pricing / Plan retrieval ---
    plan_keywords = ["price", "cost", "plan", "pricing", "basic", "pro", "month",
                     "4k", "720p", "resolution", "video", "caption", "unlimited"]
    if any(kw in query_lower for kw in plan_keywords):
        for plan in kb.get("plans", []):
            chunk = (
                f"Plan: {plan['name']} | "
                f"Price: ${plan['price_monthly']}/month | "
                f"Features: {', '.join(plan['features'])}"
            )
            relevant_chunks.append(chunk)

    # --- Policy retrieval ---
    policy_keywords = ["refund", "cancel", "support", "help", "policy", "return", "money back", "24/7"]
    if any(kw in query_lower for kw in policy_keywords):
        for policy in kb.get("policies", []):
            chunk = f"Policy [{policy['title']}]: {policy['description']}"
            relevant_chunks.append(chunk)

    # --- FAQ retrieval ---
    faq_keywords = ["trial", "free", "upgrade", "format", "publish", "platform", "youtube",
                    "instagram", "tiktok", "mp4", "mov"]
    if any(kw in query_lower for kw in faq_keywords):
        for faq in kb.get("faqs", []):
            chunk = f"FAQ: Q: {faq['question']} A: {faq['answer']}"
            relevant_chunks.append(chunk)

    # --- Fallback: return full plan+policy summary if nothing matched ---
    if not relevant_chunks:
        for plan in kb.get("plans", []):
            relevant_chunks.append(
                f"Plan: {plan['name']} | Price: ${plan['price_monthly']}/month | "
                f"Features: {', '.join(plan['features'])}"
            )
        for policy in kb.get("policies", []):
            relevant_chunks.append(f"Policy [{policy['title']}]: {policy['description']}")

    return "\n".join(relevant_chunks)


# 2. LEAD CAPTURE TOOL (Mock API)

def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """
    Mock API call that simulates submitting a qualified lead to a CRM system.
    In production, replace with a real HTTP POST to your CRM (e.g., HubSpot, Salesforce).
    """
    print("\n" + "=" * 55)
    print("LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print("=" * 55 + "\n")


# 3. STATE DEFINITION (LangGraph TypedDict)

class AgentState(TypedDict):
    """
    Persistent state object carried across all turns of the conversation.

    Fields
    ------
    messages        : Full conversation history (HumanMessage + AIMessage).
    intent          : Last classified intent — 'greeting' | 'inquiry' | 'high_intent'.
    lead_name       : Captured lead name (None until provided).
    lead_email      : Captured lead email (None until provided).
    lead_platform   : Captured creator platform (None until provided).
    lead_captured   : True once mock_lead_capture() has been successfully called.
    awaiting_lead   : True when the agent is in the middle of collecting lead details.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    awaiting_lead: bool


# 4. LLM INITIALIZATION

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.3,
    api_key=os.environ.get("GOOGLE_API_KEY"),
)


# 5. NODE DEFINITIONS

def intent_classifier_node(state: AgentState) -> dict:
    """
    Node 1 — Intent Classifier
    Reads the latest user message and classifies intent into one of:
      - 'greeting'     : Casual hello / small talk
      - 'inquiry'      : Question about product, pricing, or policies
      - 'high_intent'  : User signals readiness to sign up / buy
    Updates state['intent'].
    """
    last_message = state["messages"][-1].content

    classification_prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing company.

Classify the user's message into EXACTLY ONE of these intents:
1. greeting      — Casual greetings, small talk, or off-topic chat
2. inquiry       — Questions about product features, pricing, or policies
3. high_intent   — User clearly wants to sign up, start a trial, buy, or is ready to proceed

User message: "{last_message}"

Respond with ONLY the single word: greeting, inquiry, or high_intent.
No punctuation, no explanation."""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    intent = response.content.strip().lower()

    # Sanitize — default to 'inquiry' if the LLM returns something unexpected
    if intent not in ("greeting", "inquiry", "high_intent"):
        intent = "inquiry"

    return {"intent": intent}


def rag_response_node(state: AgentState) -> dict:
    """
    Node 2 — RAG-Powered Response Generator
    Retrieves relevant KB chunks and generates a grounded answer.
    Used for 'inquiry' intent.
    """
    kb = load_knowledge_base()
    last_message = state["messages"][-1].content
    context = retrieve_knowledge(last_message, kb)

    system_prompt = f"""You are Alex, a friendly and knowledgeable sales assistant for AutoStream — 
an automated video editing SaaS platform for content creators.

Answer the user's question using ONLY the information provided below.
If the answer is not in the context, say you'll connect them with the team.
Keep responses concise, warm, and helpful (2-4 sentences max).
End with a subtle call-to-action when appropriate.

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END CONTEXT ---"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {"messages": [AIMessage(content=response.content)]}


def greeting_node(state: AgentState) -> dict:
    """
    Node 3 — Greeting Handler
    Handles casual conversation with a warm, brand-consistent response.
    """
    system_prompt = """You are Alex, a friendly sales assistant for AutoStream — 
an automated video editing SaaS for content creators.
Respond warmly to the greeting. Briefly mention AutoStream and invite them 
to ask about plans or features. Keep it under 2 sentences."""

    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {"messages": [AIMessage(content=response.content)]}


def lead_collection_node(state: AgentState) -> dict:
    """
    Node 4 — Lead Collection & Capture
    Manages a multi-turn sub-flow to collect Name, Email, and Platform.
    Only calls mock_lead_capture() once all three values are confirmed.
    """
    last_message = state["messages"][-1].content

    # --- Attempt to extract missing lead fields from the latest message ---
    updated_fields = {}

    if not state.get("lead_name"):
        name_match = _extract_name(last_message, state["messages"])
        if name_match:
            updated_fields["lead_name"] = name_match

    if not state.get("lead_email"):
        email_match = _extract_email(last_message)
        if email_match:
            updated_fields["lead_email"] = email_match

    if not state.get("lead_platform"):
        platform_match = _extract_platform(last_message)
        if platform_match:
            updated_fields["lead_platform"] = platform_match

    # Merge extracted fields with existing state
    current_name = updated_fields.get("lead_name") or state.get("lead_name")
    current_email = updated_fields.get("lead_email") or state.get("lead_email")
    current_platform = updated_fields.get("lead_platform") or state.get("lead_platform")

    # --- Check if we have all three fields ---
    if current_name and current_email and current_platform:
        # Fire the lead capture tool
        mock_lead_capture(current_name, current_email, current_platform)

        confirmation_msg = (
            f"Fantastic, {current_name}! You're all set. "
            f"I've passed your details to our team and someone will reach out to your {current_platform} "
            f"account shortly. In the meantime, feel free to start your free trial at autostream.io. "
            f"Welcome to AutoStream!"
        )
        return {
            "messages": [AIMessage(content=confirmation_msg)],
            "lead_name": current_name,
            "lead_email": current_email,
            "lead_platform": current_platform,
            "lead_captured": True,
            "awaiting_lead": False,
            **updated_fields,
        }

    # --- Otherwise, ask for the next missing field ---
    next_prompt = _build_lead_prompt(current_name, current_email, current_platform)

    return {
        "messages": [AIMessage(content=next_prompt)],
        "awaiting_lead": True,
        "lead_name": current_name,
        "lead_email": current_email,
        "lead_platform": current_platform,
        **updated_fields,
    }


# 6. LEAD EXTRACTION HELPERS

def _extract_email(text: str) -> Optional[str]:
    """Extract email address from free text using regex."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def _extract_platform(text: str) -> Optional[str]:
    """Extract creator platform name from free text."""
    platforms = ["youtube", "instagram", "tiktok", "facebook", "twitter", "x", "twitch",
                 "linkedin", "snapchat", "pinterest", "reddit"]
    text_lower = text.lower()
    for platform in platforms:
        if platform in text_lower:
            return platform.capitalize()
    return None


def _extract_name(text: str, messages: list) -> Optional[str]:
    """
    Use the LLM to extract a person's name from a short message.
    Avoids false positives from general sentences.
    """
    extract_prompt = f"""Extract the person's full name from this message. 
If no name is present, respond with exactly: NONE

Message: "{text}"

Respond with ONLY the name or NONE."""

    response = llm.invoke([HumanMessage(content=extract_prompt)])
    result = response.content.strip()
    if result.upper() == "NONE" or len(result) > 50:
        return None
    return result


def _build_lead_prompt(name: Optional[str], email: Optional[str], platform: Optional[str]) -> str:
    """Build a natural, conversational prompt for the next missing lead field."""
    if not name:
        return (
            "That's great to hear! I'd love to get you set up. "
            "To get started, could you share your full name?"
        )
    if not email:
        return (
            f"Perfect, {name}! Now, what's the best email address "
            "where our team can reach you?"
        )
    if not platform:
        return (
            f"Almost there! Which creator platform are you mainly active on? "
            "(e.g., YouTube, Instagram, TikTok, etc.)"
        )
    return "Thank you! Let me get everything set up for you."


# 7. ROUTER FUNCTION

def route_intent(state: AgentState) -> str:
    """
    Conditional edge router.
    Directs flow to the appropriate node based on intent and lead collection status.
    """
    # If we're mid-lead-collection flow, stay in lead_collection
    if state.get("awaiting_lead") and not state.get("lead_captured"):
        return "lead_collection"

    intent = state.get("intent", "inquiry")

    if intent == "greeting":
        return "greeting"
    elif intent == "high_intent":
        return "lead_collection"
    else:
        return "rag_response"


# 8. GRAPH CONSTRUCTION

def build_graph():
    """
    Assemble the LangGraph StateGraph with all nodes and edges.
    Uses MemorySaver to seamlessly persist memory across conversation turns.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("rag_response", rag_response_node)
    graph.add_node("lead_collection", lead_collection_node)

    # Entry point
    graph.set_entry_point("intent_classifier")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "greeting": "greeting",
            "rag_response": "rag_response",
            "lead_collection": "lead_collection",
        },
    )

    # All response nodes terminate the graph turn
    graph.add_edge("greeting", END)
    graph.add_edge("rag_response", END)
    graph.add_edge("lead_collection", END)

    # Compile with MemorySaver to automatically handle state across turns
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# 9. COMMAND-LINE CHAT LOOP

def run_cli():
    """
    Interactive command-line loop for testing the AutoStream agent.
    """
    print("\n" + "=" * 55)
    print("  AutoStream AI Assistant  ")
    print(f"  Powered by LangGraph + {MODEL_NAME}")
    print("=" * 55)
    print("  Type 'quit' or 'exit' to end the session.\n")

    # Validate API key early
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("   Get a free API key from Google AI Studio and run:")
        print("   export GOOGLE_API_KEY='your-key-here'\n")
        return

    app = build_graph()
    
    # Required for LangGraph's MemorySaver to track this specific conversation instance
    config = {"configurable": {"thread_id": "session_001"}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\nAlex: Thanks for chatting! Have a great day.\n")
            break

        try:
            # We simply pass the new message, LangGraph handles appending to state internally
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        except Exception as e:
            error_text = str(e)
            if "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
                print("\nAgent error: Gemini API quota exceeded for the current key/model.")
                print("Please wait and retry, or switch model/key.")
                print("Example: export GEMINI_MODEL='gemini-2.0-flash'\n")
            else:
                print(f"\nAgent error: {e}\n")
            continue

        # Print the latest AI response
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print(f"\nAlex: {ai_messages[-1].content}\n")

        # Session ends gracefully after lead is captured
        if result.get("lead_captured"):
            print("─" * 55)
            print("Session complete — lead successfully captured.")
            print("─" * 55 + "\n")
            break

# ENTRY POINT

if __name__ == "__main__":
    run_cli()