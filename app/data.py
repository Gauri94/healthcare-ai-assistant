"""Fake payer data: a stand-in for the real member, claims and coverage systems.

CONCEPT: in production these would be separate services (a member database,
a claims API, a benefits/coverage engine). Keeping them as plain dictionaries
lets us build and test the agent before any real system exists. All names are
synthetic, so no real PHI (protected health information) is ever used.
"""

# member_id -> member record.
# CONCEPT (healthcare domain): the DEDUCTIBLE is what the member pays each year
# before the plan starts sharing costs. "met" = already paid this year.
MEMBERS = {
    "M123": {"name": "Asha Rao", "plan_id": "GOLD-PPO",
             "deductible_total": 2000.0, "deductible_met": 1150.0},
    "M456": {"name": "Ben Ortiz", "plan_id": "SILVER-HMO",
             "deductible_total": 3500.0, "deductible_met": 400.0},
}

# claim_id -> claim record.
# CONCEPT: every claim stores its OWNER (member_id). The tools compare this to the
# logged-in member to decide whether the caller may see it (authorization).
# C-2001 belongs to M456, so M123 must never be able to read it.
CLAIMS = {
    "C-1001": {"member_id": "M123", "service": "Physical therapy",
               "status": "paid", "billed": 240.0, "member_owes": 40.0},
    "C-1002": {"member_id": "M123", "service": "MRI, knee",
               "status": "pending", "billed": 1800.0, "member_owes": None},  # None = not decided yet
    "C-2001": {"member_id": "M456", "service": "Office visit",
               "status": "denied", "billed": 150.0, "member_owes": 150.0},
}

# (plan_id, CPT code) -> coverage rule.
# PYTHON: a TUPLE like ("GOLD-PPO", "73721") can be a dict key because tuples are
# immutable (unchangeable). Lists can't be keys. This gives a two-part lookup key.
# CONCEPT (healthcare domain):
#   - CPT codes identify procedures: 73721 = MRI of a lower-limb joint, 97110 = physical therapy.
#   - prior_auth = the plan must approve the procedure BEFORE it happens.
#   - coinsurance = the member pays a PERCENTAGE (0.20 = 20%) after the deductible.
#   - copay = the member pays a FIXED amount per visit.
COVERAGE = {
    ("GOLD-PPO", "73721"): {"covered": True, "prior_auth": True, "coinsurance": 0.20},
    ("GOLD-PPO", "97110"): {"covered": True, "prior_auth": False, "copay": 30.0},
    ("SILVER-HMO", "73721"): {"covered": True, "prior_auth": True, "coinsurance": 0.30},
}
