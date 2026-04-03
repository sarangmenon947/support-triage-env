"""
tools.py — External tool definitions for the Support Triage OpenEnv.

Agents can call these tools during the respond task to look up real information
rather than relying on hardcoded KB articles. This mirrors how real support agents
use internal tools like Zendesk, Salesforce, order management systems, etc.

Available tools:
  search_kb(query)            — semantic search over knowledge base articles
  lookup_customer(ticket_id)  — retrieve customer account history
  check_order_status(order_id)— look up order/subscription details
  get_similar_tickets(subject)— find similar resolved tickets for reference
"""

import hashlib
import random
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
#  Tool definitions (schema for agent)
# ─────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_kb",
        "description": "Search the knowledge base for articles relevant to a query. Returns matching articles with titles and content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'how to reset password' or 'refund policy'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "lookup_customer",
        "description": "Look up a customer's account history, plan details, and previous support interactions.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to look up customer details for"
                }
            },
            "required": ["ticket_id"]
        }
    },
    {
        "name": "check_order_status",
        "description": "Check the status of a customer's subscription or recent order/invoice.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to look up order details for"
                }
            },
            "required": ["ticket_id"]
        }
    },
    {
        "name": "get_similar_tickets",
        "description": "Find similar previously resolved support tickets to use as reference for crafting a response.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The subject of the current ticket"
                },
                "category": {
                    "type": "string",
                    "description": "The category of the ticket"
                }
            },
            "required": ["subject", "category"]
        }
    }
]


# ─────────────────────────────────────────────
#  KB article store (expanded from data.py)
# ─────────────────────────────────────────────

_KB_STORE = {
    "billing": [
        {"id": "KB001", "title": "How to update payment method",
         "content": "Log in → Settings → Billing → Payment Methods → click Update. Changes take effect on next billing cycle."},
        {"id": "KB002", "title": "Requesting a refund",
         "content": "Refunds available within 30 days of charge. Contact billing@support.example with invoice number. Processing takes 5-7 business days."},
        {"id": "KB003", "title": "Understanding your invoice",
         "content": "Invoices are generated on your billing date. PDF available in Settings → Billing → Invoice History."},
        {"id": "KB004", "title": "Promo codes and discounts",
         "content": "Apply promo codes at checkout or in Settings → Billing → Promo Code. Codes cannot be applied retroactively."},
    ],
    "technical": [
        {"id": "KB005", "title": "Setting up Slack integration",
         "content": "Go to Integrations → Slack → Connect. Authorise the app and copy the Webhook URL. Send a test message to confirm."},
        {"id": "KB006", "title": "CSV export encoding issues",
         "content": "If special characters appear corrupted, open CSV in Excel using Data → From Text/CSV and select UTF-8 encoding."},
        {"id": "KB007", "title": "API rate limits explained",
         "content": "Rate limits per API key per hour: Free=100, Starter=500, Pro=2000, Enterprise=10000. Upgrade plan for higher limits."},
        {"id": "KB008", "title": "Troubleshooting login issues",
         "content": "Clear browser cache, try incognito mode, check caps lock. If locked out, use Forgot Password on login page."},
    ],
    "account": [
        {"id": "KB009", "title": "Inviting team members",
         "content": "Settings → Team → Invite Members. Enter email addresses. Invitation link valid for 48 hours."},
        {"id": "KB010", "title": "Enabling two-factor authentication",
         "content": "Settings → Security → Two-Factor Auth. Choose SMS or Authenticator App. SMS codes may go to spam folder."},
        {"id": "KB011", "title": "Transferring account ownership",
         "content": "Settings → Team → Members → click member name → Transfer Ownership. Both parties receive confirmation email."},
        {"id": "KB012", "title": "SAML SSO configuration",
         "content": "Settings → Security → SSO. Enter your IdP metadata URL. Common error: Audience URI must match exactly."},
    ],
    "feature_request": [
        {"id": "KB013", "title": "How to submit a feature request",
         "content": "Use our feedback portal at feedback.example.com or email product@example.com. Include your use case and business impact."},
        {"id": "KB014", "title": "Product roadmap",
         "content": "Public roadmap available at roadmap.example.com. Vote on features to help prioritise development."},
    ],
    "spam": [
        {"id": "KB015", "title": "Reporting spam or abuse",
         "content": "Forward spam to abuse@example.com. Include original headers. We investigate all reports within 24 hours."},
    ],
}

_SIMILAR_TICKETS = {
    "billing": [
        {"id": "TKT-OLD-001", "subject": "Double charged last month",
         "resolution": "Confirmed duplicate charge, issued full refund within 3 business days. Customer satisfied."},
        {"id": "TKT-OLD-002", "subject": "Promo code not applied",
         "resolution": "Applied discount manually and sent updated invoice. Apologised for checkout issue."},
    ],
    "technical": [
        {"id": "TKT-OLD-003", "subject": "API returning 503 errors",
         "resolution": "Identified rate limit misconfiguration on customer's account. Updated limits and provided monitoring guide."},
        {"id": "TKT-OLD-004", "subject": "Slack integration broken after update",
         "resolution": "Webhook URL had changed. Customer updated URL and integration restored immediately."},
    ],
    "account": [
        {"id": "TKT-OLD-005", "subject": "Cannot change account email",
         "resolution": "Browser caching issue. Customer used incognito mode and email updated successfully."},
        {"id": "TKT-OLD-006", "subject": "SSO not working",
         "resolution": "Audience URI mismatch in SAML config. Provided correct URI format and customer fixed within 10 minutes."},
    ],
    "feature_request": [
        {"id": "TKT-OLD-007", "subject": "Request for bulk export",
         "resolution": "Logged feature request. Shared roadmap link. Customer subscribed to updates."},
    ],
    "spam": [
        {"id": "TKT-OLD-008", "subject": "Suspicious link in contact form",
         "resolution": "Flagged as spam, blocked sender IP, no action required from customer."},
    ],
}


# ─────────────────────────────────────────────
#  Tool executor
# ─────────────────────────────────────────────

def _seed(ticket_id: str, salt: str = "") -> int:
    return int(hashlib.md5(f"{ticket_id}{salt}".encode()).hexdigest(), 16) % (2**31)


def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    ticket_id: str,
    category: str,
    customer_plan: str = "free",
    customer_since_days: int = 0,
    previous_tickets: int = 0,
) -> Dict[str, Any]:
    """
    Execute a tool call and return the result.
    All tools are deterministic given the same inputs.
    """

    if tool_name == "search_kb":
        query = str(tool_args.get("query", "")).lower()
        # Find articles whose title or content matches query words
        results = []
        for cat_articles in _KB_STORE.values():
            for article in cat_articles:
                relevance = sum(
                    1 for word in query.split()
                    if word in article["title"].lower() or word in article["content"].lower()
                )
                if relevance > 0:
                    results.append({**article, "relevance": relevance})
        # Sort by relevance, return top 3
        results.sort(key=lambda x: x["relevance"], reverse=True)
        top = [{k: v for k, v in r.items() if k != "relevance"} for r in results[:3]]
        # Always include at least one category-relevant article
        cat_articles = _KB_STORE.get(category, [])
        if cat_articles and not any(a["id"] == cat_articles[0]["id"] for a in top):
            top = [cat_articles[0]] + top[:2]
        return {
            "tool": "search_kb",
            "query": query,
            "results": top,
            "count": len(top),
        }

    elif tool_name == "lookup_customer":
        rng = random.Random(_seed(ticket_id, "customer"))
        account_health = rng.choice(["good", "good", "good", "at_risk", "churned"])
        open_invoices  = rng.randint(0, 2) if account_health != "good" else 0
        return {
            "tool": "lookup_customer",
            "ticket_id": ticket_id,
            "customer": {
                "plan": customer_plan,
                "account_since_days": customer_since_days,
                "previous_tickets": previous_tickets,
                "account_health": account_health,
                "open_invoices": open_invoices,
                "preferred_contact": rng.choice(["email", "email", "phone"]),
                "last_login_days_ago": rng.randint(0, 30),
            }
        }

    elif tool_name == "check_order_status":
        rng = random.Random(_seed(ticket_id, "order"))
        statuses = ["active", "active", "active", "payment_failed", "cancelled", "pending"]
        status = rng.choice(statuses)
        days_since = rng.randint(1, 60)
        return {
            "tool": "check_order_status",
            "ticket_id": ticket_id,
            "order": {
                "subscription_status": status,
                "last_invoice_days_ago": days_since,
                "last_invoice_amount": rng.choice([49, 99, 199, 299, 499]),
                "payment_method": "card_ending_" + str(rng.randint(1000, 9999)),
                "next_renewal_days": rng.randint(1, 30) if status == "active" else None,
            }
        }

    elif tool_name == "get_similar_tickets":
        similar = _SIMILAR_TICKETS.get(category, [])
        return {
            "tool": "get_similar_tickets",
            "category": category,
            "similar_tickets": similar,
            "count": len(similar),
        }

    else:
        return {"tool": tool_name, "error": f"Unknown tool: {tool_name}"}


# ─────────────────────────────────────────────
#  Tool usage tracker (for grading)
# ─────────────────────────────────────────────

def score_tool_usage(tool_calls: List[Dict[str, Any]], category: str) -> float:
    """
    Score how well the agent used tools. Returns 0.0-1.0.
    Rewards: using relevant tools, using search_kb, not spamming calls.
    Penalises: zero tool use, excessive calls (>5).
    """
    if not tool_calls:
        return 0.0

    tools_used = [t.get("tool_name", "") for t in tool_calls]
    num_calls  = len(tool_calls)

    # Penalise excessive tool use
    if num_calls > 5:
        return 0.3

    score = 0.0

    # Used search_kb — most important tool
    if "search_kb" in tools_used:
        score += 0.5

    # Used customer lookup — shows thoroughness
    if "lookup_customer" in tools_used or "check_order_status" in tools_used:
        score += 0.3

    # Used similar tickets — shows learning from history
    if "get_similar_tickets" in tools_used:
        score += 0.2

    return min(score, 1.0)