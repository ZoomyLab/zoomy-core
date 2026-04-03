from dataclasses import dataclass, field
import time

from zoomy_core.workspace.card import Card


@dataclass
class Session:
    id: str = ""
    title: str = "Default session"
    description: str = "Simulation session."
    cards: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"s-{int(time.time() * 1000)}"

    def add_card(self, card):
        self.cards[card.id] = card

    def get_card(self, card_id):
        return self.cards.get(card_id)

    def to_dict(self):
        return {"id": self.id, "title": self.title, "description": self.description}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d.get("id", ""), title=d.get("title", ""), description=d.get("description", ""))
