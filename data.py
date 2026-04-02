"""
Deterministic synthetic data for all three tasks.
Seeded by episode_id so reset() always returns the same ticket for the same seed.
"""

import hashlib
import random
from typing import Dict, List, Tuple

from models import Ticket, KBArticle


# ─────────────────────────────────────────────
#  Ticket corpus  
# ─────────────────────────────────────────────

_TICKETS_RAW = [
    # Billing
    ("billing", "P2", "billing_team", "Invoice #4821 shows double charge",
     "Hi, I was charged twice for my March subscription — $49 appeared on my card on March 1 and again on March 3. Please refund the duplicate.", "negative", "pro", 180, 2),
    ("billing", "P3", "billing_team", "How do I update my payment method?",
     "I got a new credit card and want to update billing details before my renewal on the 15th.", "neutral", "starter", 45, 0),
    ("billing", "P2", "billing_team", "Refund request for cancelled account",
     "I cancelled my account on Apr 2 but the annual fee of $299 was still charged. I need a full refund.", "angry", "pro", 92, 3),
    ("billing", "P3", "billing_team", "Promo code not applied",
     "I used SAVE20 at checkout but my invoice doesn't show a discount. Can you fix this?", "neutral", "free", 10, 0),
    ("billing", "P1", "billing_team", "Unauthorized charge of $999",
     "There is a $999 charge on my account I did not authorise. This looks like fraud. Please investigate immediately.", "angry", "enterprise", 730, 5),
    ("billing", "P3", "billing_team", "Can I get an annual invoice for taxes?",
     "I need a PDF invoice for the full year 2024 for my accounting records.", "positive", "pro", 400, 1),
    ("billing", "P4", "billing_team", "Question about upgrade pricing",
     "What is the price difference between Starter and Pro plans if I upgrade mid-cycle?", "neutral", "starter", 20, 0),
    ("billing", "P2", "billing_team", "VAT not shown on invoice",
     "As a UK business we need VAT shown on all invoices. Our last three invoices are missing this.", "neutral", "enterprise", 300, 2),
    ("billing", "P3", "billing_team", "Billing cycle change request",
     "Can I switch from monthly to annual billing to save money?", "positive", "starter", 60, 0),
    ("billing", "P2", "billing_team", "Free trial ended but was still charged",
     "My 14-day trial ended and I was immediately charged $49 without a warning email.", "angry", "free", 14, 0),
    # Technical
    ("technical", "P1", "tech_support", "App completely down — production outage",
     "Our entire team cannot log in. We're getting 503 errors on every page. This is a production outage affecting 50 users.", "angry", "enterprise", 540, 8),
    ("technical", "P2", "tech_support", "CSV export is corrupted",
     "Every time I export a report to CSV, special characters like é and ñ appear as question marks. Using Chrome on Mac.", "negative", "pro", 200, 2),
    ("technical", "P3", "tech_support", "Integration with Slack not working",
     "I followed the docs to connect Slack but the test message never arrives. Webhook URL is set correctly.", "neutral", "starter", 90, 1),
    ("technical", "P1", "tech_support", "Data loss — records deleted",
     "Around 200 customer records were deleted from our account between 2am-3am. We did not do this. Need immediate recovery.", "angry", "enterprise", 900, 4),
    ("technical", "P2", "tech_support", "Login loop on mobile app",
     "On iPhone 15 iOS 17.4, the app logs me out immediately after logging in, creating an infinite loop.", "negative", "pro", 110, 1),
    ("technical", "P3", "tech_support", "API rate limit hit unexpectedly",
     "We're hitting 429 errors at only ~500 requests/hour, well below our stated 2000/hour limit.", "neutral", "pro", 180, 0),
    ("technical", "P4", "tech_support", "Dark mode colours look off",
     "In dark mode, the sidebar text is very hard to read — dark grey on dark background.", "neutral", "free", 5, 0),
    ("technical", "P2", "tech_support", "2FA codes not arriving by SMS",
     "Since yesterday I'm not receiving the 6-digit 2FA codes by SMS. I've tried multiple times.", "negative", "starter", 75, 0),
    ("technical", "P3", "tech_support", "Webhook timeout errors",
     "Our webhooks are failing with timeout errors after exactly 5 seconds. Payload is only 2KB.", "neutral", "pro", 250, 2),
    ("technical", "P4", "tech_support", "Search results seem slow",
     "The search bar takes about 3-4 seconds to return results. It used to be instant.", "neutral", "free", 30, 0),
    # Account (10)
    ("account", "P2", "account_team", "Cannot change email address",
     "I've been trying to update my email to my new company domain but the Save button stays greyed out.", "negative", "pro", 150, 1),
    ("account", "P3", "account_team", "How to add team members?",
     "I just upgraded to Pro. How do I invite my two colleagues to join the workspace?", "positive", "pro", 3, 0),
    ("account", "P2", "account_team", "Account locked after wrong password",
     "I entered my password incorrectly three times and now my account is locked. I need access urgently.", "negative", "starter", 60, 0),
    ("account", "P1", "account_team", "Suspected account compromise",
     "I received a login notification from an IP in another country. I did not do this. Please lock the account.", "angry", "enterprise", 600, 3),
    ("account", "P3", "account_team", "Transfer account ownership to colleague",
     "I'm leaving the company next week. How do I transfer the account owner role to my manager?", "neutral", "pro", 400, 0),
    ("account", "P4", "account_team", "How to delete my account?",
     "I no longer need this service. What is the process to permanently delete my account and data?", "neutral", "free", 20, 0),
    ("account", "P3", "account_team", "Change organisation name",
     "Our company rebranded. I need to update the organisation name shown on the dashboard and invoices.", "neutral", "starter", 200, 1),
    ("account", "P2", "account_team", "SSO configuration not working",
     "We set up SAML SSO per the docs but employees get an error: 'Assertion audience mismatch'.", "negative", "enterprise", 365, 2),
    ("account", "P4", "account_team", "Change timezone setting",
     "All my reports show timestamps in UTC. I need them in IST (UTC+5:30).", "neutral", "free", 45, 0),
    ("account", "P3", "account_team", "Remove a user from the workspace",
     "An employee has left and I need to revoke their access and reassign their tasks.", "neutral", "pro", 300, 1),
    # Feature Request
    ("feature_request", "P4", "product_team", "Please add dark mode",
     "It would be great to have a dark mode option for evening use. Many competitors have this.", "positive", "free", 10, 0),
    ("feature_request", "P4", "product_team", "Bulk export all data to Excel",
     "We need to export all records at once to Excel for board reporting. Currently we have to do it page by page.", "neutral", "pro", 180, 1),
    ("feature_request", "P4", "product_team", "Mobile app for Android",
     "Is there an Android app planned? We have team members who only use Android phones.", "positive", "starter", 40, 0),
    ("feature_request", "P4", "product_team", "Calendar integration with Google Calendar",
     "Syncing tasks with Google Calendar would save us a lot of manual work.", "positive", "pro", 200, 0),
    ("feature_request", "P4", "product_team", "Zapier integration",
     "We use Zapier extensively. A native Zapier integration would help us automate our workflow.", "positive", "pro", 350, 2),
    ("feature_request", "P4", "product_team", "Custom report builder",
     "The current reports are too rigid. A drag-and-drop report builder would be very useful.", "neutral", "enterprise", 600, 3),
    ("feature_request", "P4", "product_team", "Multi-language support",
     "Our team is international. French and Spanish UI would help non-English speakers.", "positive", "starter", 80, 0),
    ("feature_request", "P4", "product_team", "Audit log for all user actions",
     "For compliance we need a full audit log showing who did what and when.", "neutral", "enterprise", 720, 1),
    ("feature_request", "P4", "product_team", "Keyboard shortcuts",
     "Power users like me would love keyboard shortcuts for common actions like creating and archiving.", "positive", "pro", 500, 0),
    ("feature_request", "P4", "product_team", "Two-way email sync",
     "Replies sent from Gmail should automatically be logged in the system.", "neutral", "pro", 300, 2),
    # Spam
    ("spam", "P4", "spam_filter", "Congratulations! You have won $5,000",
     "Click here to claim your prize now! Limited time offer. Enter your card details to verify.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "SEO services for your website",
     "We offer guaranteed first page Google rankings. Contact us for a free audit.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Investment opportunity — 300% returns",
     "Our proprietary trading algorithm delivers 300% annual returns. Join thousands of investors today.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Cheap meds no prescription needed",
     "Buy prescription medications online without a doctor. Huge discounts. Fast shipping.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Your account needs urgent verification",
     "Dear user, your account will be suspended. Click this link immediately to verify: http://scam-site.xyz", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Make money from home — $500/day",
     "Work from home opportunity! No experience needed. Earn $500+ per day filling out surveys.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Crypto trading signals — join our VIP group",
     "Get our daily crypto signals. Our members made 10x last month. Join the Telegram group now.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Weight loss pill — lose 30lbs in 2 weeks",
     "Doctors hate this one weird trick! Our pill melts fat while you sleep.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "Rolex watches — 90% off",
     "Authentic luxury watches at 90% discount. Ships worldwide. Limited stock.", "neutral", "free", 0, 0),
    ("spam", "P4", "spam_filter", "You have been selected for a survey",
     "Complete a 2-minute survey and win a $1000 Amazon gift card! Click to start.", "neutral", "free", 0, 0),
]

# ─────────────────────────────────────────────
#  Knowledge-base articles
# ─────────────────────────────────────────────

_KB_ARTICLES = [
    KBArticle(
        article_id="KB001",
        title="How to update your payment method",
        category="billing",
        content="Log in → Settings → Billing → Payment Methods → click 'Update'. Changes take effect on the next billing cycle."
    ),
    KBArticle(
        article_id="KB002",
        title="Requesting a refund",
        category="billing",
        content="Refunds are available within 30 days of charge. Contact billing@support.example with your invoice number. Processing takes 5–7 business days."
    ),
    KBArticle(
        article_id="KB003",
        title="How to reset your password",
        category="account",
        content="Click 'Forgot password' on the login page. Enter your email and check your inbox for a reset link valid for 15 minutes."
    ),
    KBArticle(
        article_id="KB004",
        title="Inviting team members",
        category="account",
        content="Go to Settings → Team → Invite Members. Enter their email addresses (comma-separated). They'll receive an invitation link valid for 48 hours."
    ),
    KBArticle(
        article_id="KB005",
        title="Setting up Slack integration",
        category="technical",
        content="Go to Integrations → Slack → Connect. Authorise the app in Slack and copy the Webhook URL into the field provided. Send a test message to confirm."
    ),
    KBArticle(
        article_id="KB006",
        title="CSV export encoding issues",
        category="technical",
        content="If special characters appear corrupted, open the CSV in Excel using Data → From Text/CSV and select UTF-8 encoding. Alternatively, use Google Sheets which handles UTF-8 automatically."
    ),
    KBArticle(
        article_id="KB007",
        title="API rate limits explained",
        category="technical",
        content="Rate limits are per API key per hour: Free=100, Starter=500, Pro=2000, Enterprise=10000. If you receive 429 errors, wait for the next hour window or upgrade your plan."
    ),
    KBArticle(
        article_id="KB008",
        title="Enabling two-factor authentication",
        category="account",
        content="Settings → Security → Two-Factor Auth. Choose SMS or Authenticator App. If SMS codes aren't arriving, check your phone's spam filter or switch to an authenticator app."
    ),
]


def _seed_from_id(episode_id: str) -> int:
    """Derive a numeric seed from an episode_id string."""
    return int(hashlib.md5(episode_id.encode()).hexdigest(), 16) % (2 ** 31)


def get_ticket_for_episode(episode_id: str, task: str = "classify") -> Tuple[Ticket, dict]:
    """
    Return a deterministic (Ticket, ground_truth) pair for a given episode_id.
    ground_truth contains the correct answers for grading.
    """
    rng = random.Random(_seed_from_id(episode_id))
    idx = rng.randint(0, len(_TICKETS_RAW) - 1)
    row = _TICKETS_RAW[idx]
    category, priority, team, subject, body, sentiment, plan, days, prev = row

    ticket = Ticket(
        ticket_id=f"TKT-{episode_id[:6].upper()}",
        subject=subject,
        body=body,
        customer_plan=plan,
        customer_since_days=days,
        previous_tickets=prev,
        sentiment=sentiment,
    )
    ground_truth = {
        "category": category,
        "priority": priority,
        "assigned_team": team,
    }
    return ticket, ground_truth


def get_kb_articles_for_ticket(ticket: Ticket, ground_truth: dict, n: int = 3) -> List[KBArticle]:
    """Return relevant KB articles (includes at least one relevant one)."""
    category = ground_truth["category"]
    relevant = [a for a in _KB_ARTICLES if a.category == category]
    others = [a for a in _KB_ARTICLES if a.category != category]
    rng = random.Random(hash(ticket.ticket_id))
    selected = relevant[:1] + rng.sample(others, min(n - 1, len(others)))
    rng.shuffle(selected)
    return selected


# ─────────────────────────────────────────────
#  Simulated customer replies (for respond step 2)
# ─────────────────────────────────────────────

_CUSTOMER_REPLIES: dict = {
    "billing": [
        "Yes, I was charged on March 1st and again on March 3rd. The amount was $49 each time. My card ending in 4242.",
        "The charge appeared on my last statement. I have the invoice number: INV-2024-0892. I would like a full refund.",
        "I used the promo code SAVE20 during checkout but the discount was not applied. The order total was $99.",
    ],
    "technical": [
        "The error message says 503 Service Unavailable. This started about 2 hours ago and affects all our users.",
        "I am using Chrome version 120 on Windows 11. The issue started after the last update. Cache cleared already.",
        "The integration was working fine last week. I have not changed any settings. The webhook URL is correct.",
    ],
    "account": [
        "My email is old@company.com and I need to change it to new@company.com. The save button stays greyed out.",
        "I have been locked out since this morning. I need urgent access as I have a client meeting in 1 hour.",
        "I am the account owner and need to transfer ownership to john@company.com before I leave next week.",
    ],
    "feature_request": [
        "We have about 15 team members who would use this feature daily. It would save us around 2 hours per week.",
        "We are currently exporting data manually page by page which takes a very long time. Bulk export would help.",
        "Our team is spread across 3 time zones and a mobile app would help us stay on top of tasks on the go.",
    ],
    "spam": [
        "I did not send this message intentionally. My account may have been compromised.",
        "This was sent by mistake. Please ignore it.",
        "I am not sure how this was submitted. Please disregard.",
    ],
}


def simulate_customer_reply(ticket: "Ticket", question: str, ground_truth: dict) -> str:
    """
    Simulate a deterministic customer reply to the agent's clarifying question.
    Uses the ticket category to pick a realistic reply.
    """
    category = ground_truth.get("category", "technical")
    replies = _CUSTOMER_REPLIES.get(category, _CUSTOMER_REPLIES["technical"])
    rng = random.Random(hash(ticket.ticket_id + question[:10]))
    return rng.choice(replies)