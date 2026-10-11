"""Shared knowledge base for the demos: a small, made-up college FAQ."""

from langchain_core.documents import Document

COLLEGE_FAQ = [
    Document(
        page_content="Admissions: Applications for the B.Tech program open on 1 May and close on 30 June. "
        "Students apply online through the college portal.",
        metadata={"source": "admissions"},
    ),
    Document(
        page_content="Hostel: Hostel rooms are allotted on a first come, first served basis. "
        "The hostel fee is paid every semester at the accounts office.",
        metadata={"source": "hostel"},
    ),
    Document(
        page_content="Scholarships: Students with 90% or above in Class 12 can apply for a merit scholarship "
        "within 30 days of admission. The scholarship covers part of the tuition fee.",
        metadata={"source": "scholarships"},
    ),
    Document(
        page_content="Fee refund: If a student withdraws before classes begin, the tuition fee is refunded "
        "after deducting a processing charge. No refund is given after classes begin.",
        metadata={"source": "fee_refund"},
    ),
    Document(
        page_content="Library: The library is open from 8 AM to 8 PM on weekdays. "
        "Students can borrow up to 4 books for 14 days.",
        metadata={"source": "library"},
    ),
]
