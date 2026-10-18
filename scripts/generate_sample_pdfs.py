"""Generate four fictional supplier RFP responses (PDF) for the same procurement request.

All companies, people, prices and references are invented. Run:  python scripts/generate_sample_pdfs.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "rfps"

RFP_TITLE = "RFP-2026-017: Supplier Portal and Contract Lifecycle Management Platform"
BUYER = "Meridian Foods Ltd. (fictional)"
REQUIREMENT = (
    "Meridian Foods Ltd. requires a cloud-based Supplier Portal and Contract Lifecycle Management (CLM) platform "
    "for about 2,500 internal users and 1,200 external suppliers. Mandatory needs: integration with the SAP S/4HANA "
    "ERP and corporate single sign-on, workflow-based contract approvals, auditability, data protection aligned with "
    "ISO 27001, go-live within 6 months and 3 years of support."
)

SUPPLIERS: list[dict] = [
    {
        "file": "Apex_Systems_RFP_Response.pdf",
        "name": "Apex Systems",
        "tagline": "Engineered for scale, secured by design",
        "sections": [
            ("1. Executive Summary and Understanding of the Requirement", [
                "Apex Systems proposes a microservices-based Supplier Portal and CLM platform deployed on a dedicated "
                "Kubernetes cluster in a regional cloud data centre. We understand that Meridian Foods needs a single "
                "system of record for supplier onboarding and contracts, tightly integrated with SAP S/4HANA and "
                "corporate identity, and capable of passing external security audits.",
                "Our proposal prioritises technical robustness and security assurance. It is the most "
                "engineering-intensive option and is priced accordingly.",
            ]),
            ("2. Proposed Solution and Implementation Approach", [
                "Architecture: API-first microservices (contract service, supplier service, workflow engine, "
                "document service) behind an API gateway, with PostgreSQL and an event bus for asynchronous "
                "processing. Horizontal autoscaling has been load-tested to 10,000 concurrent users, four times the "
                "stated requirement.",
                "Integrations: SAP S/4HANA integration uses SAP-certified connectors over OData and IDoc, with a "
                "documented mapping for vendor master, purchase order and payment status. SSO is delivered through "
                "SAML 2.0 and OpenID Connect against the corporate directory. A REST and webhook API is provided for "
                "future systems.",
                "Delivery follows five phases: discovery, architecture and build, integration, hardening and UAT, "
                "and phased go-live.",
            ]),
            ("3. Timeline, Team Structure and Milestones", [
                "Total duration is 26 weeks from contract signature to production go-live.",
                [
                    "Weeks 1-4: discovery and solution design sign-off",
                    "Weeks 5-14: core platform build and SSO",
                    "Weeks 15-20: SAP integration and data migration",
                    "Weeks 21-24: security hardening, performance test and UAT",
                    "Weeks 25-26: go-live and hypercare",
                ],
                "Core team of 11 FTE: engagement manager, solution architect, 5 engineers, 2 QA engineers, "
                "security engineer and a DevOps engineer.",
            ]),
            ("4. Commercial Proposal", [
                "All prices are in USD, fixed price for the scope described in this document.",
                {"table": [
                    ["Item", "Amount (USD)"],
                    ["Platform licence (3 years)", "195,000"],
                    ["Implementation and integration services", "210,000"],
                    ["Security hardening and penetration test", "35,000"],
                    ["Training and documentation", "15,000"],
                    ["Support, years 1-3 (premium tier)", "30,000"],
                    ["Total three-year cost", "485,000"],
                ]},
                "Assumptions: up to 2,500 internal users and 1,200 suppliers; one SAP S/4HANA instance; travel "
                "excluded; change requests billed at 1,100 USD per day. Prices valid for 90 days.",
            ]),
            ("5. Security, Compliance and Risk Controls", [
                "Apex holds ISO 27001:2022 and SOC 2 Type II certifications (reports available under NDA). Data is "
                "encrypted with AES-256 at rest and TLS 1.3 in transit, with customer-managed keys optional. Role-based "
                "access control, mandatory MFA for administrators and immutable audit logs retained for 7 years are "
                "standard. Annual third-party penetration tests are performed and a GDPR data processing agreement is "
                "included. Data residency is configurable by region.",
                "Key risks and mitigations: SAP interface complexity (mitigated by early integration spike and "
                "certified connectors); schedule pressure on a 26-week plan (mitigated by phased go-live).",
            ]),
            ("6. Support Model, Experience and References", [
                "Support is provided during business hours (8x5) with 4-hour response for P1 incidents; 24x7 cover is "
                "available as a paid option. A named technical account manager is assigned.",
                "Apex has delivered three comparable CLM and supplier-portal programmes for manufacturing and retail "
                "clients over the last four years. References: Harborline Retail (fictional), CTO office; Castell "
                "Manufacturing (fictional), Head of Procurement Systems.",
            ]),
        ],
    },
    {
        "file": "BrightPath_Tech_RFP_Response.pdf",
        "name": "BrightPath Tech",
        "tagline": "Fast, simple, affordable procurement software",
        "sections": [
            ("1. Executive Summary and Understanding of the Requirement", [
                "BrightPath Tech offers a ready-to-use multi-tenant SaaS supplier portal that can be configured and "
                "launched quickly. We understand Meridian Foods wants a portal for suppliers and contracts at a "
                "competitive price. Our offer is the lowest cost and the fastest to deliver.",
            ]),
            ("2. Proposed Solution and Implementation Approach", [
                "Architecture: a standard multi-tenant SaaS application with configurable templates for onboarding "
                "forms and contract workflows. The platform exposes a standard REST API.",
                "Integrations: SAP data exchange is handled through scheduled CSV import and export files. Real-time "
                "SAP integration is not included in the base scope. SSO via SAML is supported.",
                "Scalability details for 2,500 users are not included in this proposal.",
                "We configure the platform from our template library and migrate supplier records supplied by "
                "Meridian in spreadsheet form.",
            ]),
            ("3. Timeline, Team Structure and Milestones", [
                "Go-live in 14 weeks from contract signature.",
                [
                    "Weeks 1-2: kickoff and configuration workshop",
                    "Weeks 3-9: configuration and CSV integration",
                    "Weeks 10-12: user acceptance testing",
                    "Weeks 13-14: go-live",
                ],
                "A small team of 5 people will deliver the project: a project lead, 2 configuration consultants, a "
                "tester and a trainer. A formal risk plan is not part of this response.",
            ]),
            ("4. Commercial Proposal", [
                "All prices are in USD.",
                {"table": [
                    ["Item", "Amount (USD)"],
                    ["Subscription (3 years)", "150,000"],
                    ["Configuration and onboarding", "95,000"],
                    ["Training", "10,000"],
                    ["Support (included in subscription)", "0"],
                    ["Total three-year cost", "255,000"],
                ]},
                "Assumptions: standard templates only; additional integrations, custom workflows and change requests "
                "are charged separately.",
            ]),
            ("5. Security, Compliance and Risk Controls", [
                "BrightPath follows industry best practices for information security. ISO 27001 certification is in "
                "progress. Data is encrypted. Details of audit logging, penetration testing and data residency are "
                "available on request.",
            ]),
            ("6. Support Model, Experience and References", [
                "Support is available by email and a ticket portal during working hours. Response targets will be "
                "agreed after contract signature.",
                "BrightPath was founded in 2022 and has completed one comparable deployment, for a small food "
                "distributor. References are available on request.",
            ]),
        ],
    },
    {
        "file": "NexaWorks_RFP_Response.pdf",
        "name": "NexaWorks",
        "tagline": "A dependable delivery partner",
        "sections": [
            ("1. Executive Summary and Understanding of the Requirement", [
                "NexaWorks proposes a configurable CLM and supplier-portal platform delivered through a structured, "
                "low-risk programme. We understand that Meridian Foods needs reliable integration with SAP S/4HANA "
                "and SSO, a smooth transition for 2,500 users and strong support after go-live. Our proposal balances "
                "capability, security, cost and delivery certainty.",
            ]),
            ("2. Proposed Solution and Implementation Approach", [
                "Architecture: a modular cloud application with a configurable workflow engine and a managed "
                "relational database; it scales to 5,000 users on the proposed sizing.",
                "Integrations: SAP S/4HANA is integrated through standard OData services with a pre-built "
                "vendor-master and purchase-order mapping; SSO uses SAML 2.0. Integration design is signed off in "
                "week 3 and tested in a sandbox before production.",
                "Our delivery method has six gated phases (mobilise, design, build, integrate, test and train, go-live) "
                "with a formal acceptance criterion at the end of each phase.",
            ]),
            ("3. Timeline, Team Structure and Milestones", [
                "Total duration is 20 weeks from contract signature, with a pilot with 50 users in week 15.",
                [
                    "Week 2: project charter, RACI and communication plan approved",
                    "Week 3: integration design sign-off",
                    "Week 8: build complete for core workflows (milestone payment 1)",
                    "Week 12: SAP sandbox integration passed",
                    "Week 15: pilot go-live with 50 users",
                    "Week 18: UAT and training of 40 key users and train-the-trainer sessions complete",
                    "Week 20: production go-live; 6 weeks of hypercare",
                ],
                "Staffing: 14 named roles including a dedicated project manager, change manager, solution architect, "
                "3 developers, 2 integration engineers, 2 testers and a security officer. A risk register with owners "
                "and weekly review is maintained; typical top risks include SAP data quality, user adoption and key-person "
                "availability, each with a named mitigation.",
            ]),
            ("4. Commercial Proposal", [
                "All prices are in USD, fixed price, with milestone-based payments (20% at signature, 30% at week 8, "
                "30% at pilot go-live, 20% at production acceptance).",
                {"table": [
                    ["Item", "Amount (USD)"],
                    ["Platform licence (3 years)", "150,000"],
                    ["Implementation and SAP integration", "145,000"],
                    ["Training and change management", "25,000"],
                    ["Support, years 1-3 (gold tier)", "40,000"],
                    ["Total three-year cost", "360,000"],
                ]},
                "Assumptions: up to 2,500 internal users and 1,200 suppliers; one SAP S/4HANA instance; two "
                "environments (test and production); change requests at 900 USD per day; prices valid for 120 days.",
            ]),
            ("5. Security, Compliance and Risk Controls", [
                "NexaWorks is ISO 27001 certified and holds a SOC 2 Type I report; the Type II audit is scheduled "
                "for next year. Data is encrypted at rest and in transit, access is role-based with MFA, and "
                "audit logs are retained for 3 years. Penetration tests are run annually. A data processing "
                "agreement is provided.",
            ]),
            ("6. Support Model, Experience and References", [
                "24x7 support with three tiers. P1 incidents get a 30-minute response and a 4-hour target resolution; "
                "P2 incidents a 2-hour response. Each customer has a named customer success manager, quarterly "
                "service reviews and a monthly service report.",
                "NexaWorks has delivered five comparable programmes in the last five years. References: Orchard "
                "Beverages (fictional), Procurement Director; Lumen Pharma (fictional), IT Director; "
                "Tidewater Logistics (fictional), Head of Supply Chain.",
            ]),
        ],
    },
    {
        "file": "Orbit_Digital_RFP_Response.pdf",
        "name": "Orbit Digital",
        "tagline": "Nine years of procurement transformation",
        "sections": [
            ("1. Executive Summary and Understanding of the Requirement", [
                "Orbit Digital has specialised in procurement and contract management platforms for nine years. We "
                "have delivered more than twelve comparable programmes and our customers consistently rate us highly "
                "for partnership. We understand Meridian Foods' need for a supplier portal and CLM system and "
                "propose a proven product with a delivery team that has done this before.",
            ]),
            ("2. Proposed Solution and Implementation Approach", [
                "Architecture: a proven CLM product with a supplier portal module hosted in a public cloud.",
                "Integrations: Orbit will integrate with the customer's systems using suitable approaches. The "
                "detailed integration design will be finalised in the discovery phase. SAP and SSO integration "
                "methods are to be confirmed once requirements are better understood.",
                "Delivery follows a standard methodology with discovery, configuration, testing and rollout.",
            ]),
            ("3. Timeline, Team Structure and Milestones", [
                "Indicative duration is 22 weeks.",
                [
                    "Weeks 1-3: discovery",
                    "Weeks 4-14: configuration",
                    "Weeks 15-19: testing",
                    "Weeks 20-22: rollout",
                ],
                "The team will be a project manager, solution consultants and engineers, with exact numbers to be "
                "confirmed.",
            ]),
            ("4. Commercial Proposal", [
                "All prices are in USD.",
                {"table": [
                    ["Item", "Amount (USD)"],
                    ["Platform licence (3 years)", "170,000"],
                    ["Implementation services", "175,000"],
                    ["Training", "20,000"],
                    ["Support, years 1-3", "30,000"],
                    ["Total three-year cost", "395,000"],
                ]},
                "Assumptions: integration effort is estimated and may be adjusted after discovery.",
            ]),
            ("5. Security, Compliance and Risk Controls", [
                "Orbit is ISO 27001 certified. Data is encrypted at rest and in transit and access is role based. "
                "Audit logs are available. Further controls are described in our security whitepaper, available on "
                "request.",
            ]),
            ("6. Support Model, Experience and References", [
                "Support is provided 8x5 with a named service manager, with extended hours available at extra cost. "
                "Response targets: P1 within 2 hours.",
                "Experience: more than twelve CLM and procurement-portal programmes since 2017, including four in "
                "food and consumer goods. Customer references (all fictional): Saffron Foods, Chief Procurement "
                "Officer; Greenfield Dairy, VP Operations; Northgate Retail, Head of Procurement; Bluewave Packaging, "
                "IT Director. Each has agreed to take reference calls and a case study for Saffron Foods is "
                "attached.",
            ]),
        ],
    },
]


def build_pdf(data: dict) -> Path:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20, textColor=colors.HexColor("#1b2a49"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, textColor=colors.HexColor("#2563eb"), spaceBefore=10)
    meta = ParagraphStyle("meta", parent=body, textColor=colors.HexColor("#555555"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / data["file"]
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"{data['name']} - Response to {RFP_TITLE}", author=data["name"],
    )

    def footer(canvas, d):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 10 * mm, f"{data['name']} | Response to {RFP_TITLE} | Fictional sample data")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {d.page}")

    story = [
        Paragraph(f"{data['name']}", h1),
        Paragraph(f"<i>{data['tagline']}</i>", meta),
        Paragraph(f"Response to {RFP_TITLE}<br/>Issued by {BUYER}", meta),
        Spacer(1, 4),
        Paragraph("<b>Requirement summary.</b> " + REQUIREMENT, body),
    ]
    for heading, items in data["sections"]:
        story.append(Paragraph(heading, h2))
        for item in items:
            if isinstance(item, str):
                story.append(Paragraph(item, body))
            elif isinstance(item, list):
                story.append(ListFlowable([ListItem(Paragraph(x, body)) for x in item], bulletType="bullet", leftIndent=14))
            else:
                t = Table(item["table"], colWidths=[110 * mm, 50 * mm])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a49")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3e3")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ]))
                story += [t, Spacer(1, 6)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


if __name__ == "__main__":
    from pypdf import PdfReader

    for s in SUPPLIERS:
        p = build_pdf(s)
        print(f"{p.name}: {len(PdfReader(str(p)).pages)} pages")
