"""Semantic runtime identity, separate from immutable neural weight hashes.

Legacy evidence stays readable. Only explicitly current observations may train
the current runtime; loading old weights does not rewrite their provenance.
"""

LEGACY_PHASE_SCHEMA = "combat0_upgrade1_weapon2_shop3_character4-v1"
PHASE_SCHEMA = "combat0_upgrade_or_loot1_weapon2_shop3_character4-v2"
LEGACY_RUNTIME_CONTRACT = "brotato-recurrent-pre-neural-loot-v1"
RUNTIME_CONTRACT = "brotato-recurrent-context64-candidate16-neural-loot-v2"


def contract_fields():
    return {"runtime_contract": RUNTIME_CONTRACT, "phase_schema": PHASE_SCHEMA}


def contract_identity(document):
    """Missing legacy tags are classified in memory, never written over source."""
    identity = (document.get("runtime_contract", LEGACY_RUNTIME_CONTRACT),
                document.get("phase_schema", LEGACY_PHASE_SCHEMA))
    if identity not in ((LEGACY_RUNTIME_CONTRACT, LEGACY_PHASE_SCHEMA),
                        (RUNTIME_CONTRACT, PHASE_SCHEMA)):
        raise ValueError("unknown or inconsistent runtime/phase contract")
    return identity


def require_current_contract(document):
    if contract_identity(document) != (RUNTIME_CONTRACT, PHASE_SCHEMA):
        raise ValueError("legacy runtime evidence is read-only under the current contract")
