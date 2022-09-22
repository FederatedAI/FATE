from typing import Protocol
from ._party import Party, Parties

class FederationEngine(Protocol):
    guest: Party
    hosts: Parties
    arbiter: Party
    parties: Parties
