import json
import copy
from dataclasses import dataclass, field

from zoomy_core.workspace.card import Card
from zoomy_core.workspace.session import Session
from zoomy_core.workspace.io import save_project_zip, load_project_zip


@dataclass
class Project:
    sessions: dict = field(default_factory=dict)
    active_session: str = ""
    selections: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    cards: dict = field(default_factory=dict)

    def add_session(self, session):
        self.sessions[session.id] = session
        if not self.active_session:
            self.active_session = session.id

    def get_active_session(self):
        return self.sessions.get(self.active_session)

    def switch_session(self, session_id):
        if session_id in self.sessions:
            self.active_session = session_id

    def select(self, tab, card_id):
        self.selections[tab] = card_id

    def selected(self, tab):
        card_id = self.selections.get(tab)
        if card_id:
            return self.cards.get(card_id)
        return None

    def status(self):
        result = {}
        for tab in ["model", "mesh", "solver"]:
            card = self.selected(tab)
            result[tab] = card.title if card else "Not selected"
        session = self.get_active_session()
        result["session"] = session.title if session else "No session"
        return result

    def build_case(self):
        model = self.selected("model")
        mesh = self.selected("mesh")
        solver = self.selected("solver")

        if not model or not mesh or not solver:
            missing = [t for t in ["model", "mesh", "solver"] if not self.selected(t)]
            raise ValueError(f"Missing selections: {', '.join(missing)}")

        case = {"version": "1.0"}
        case["model"] = {
            "class_path": model.class_path or model.id,
            "init": model.init,
            "parameters": model.params,
        }

        mesh_init = mesh.init or {}
        if "n_cells" in mesh_init:
            case["mesh"] = {
                "type": "create_1d",
                "domain": [mesh_init.get("x_min", 0), mesh_init.get("x_max", 1)],
                "n_cells": mesh_init.get("n_cells", 100),
            }
        elif "nx" in mesh_init:
            case["mesh"] = {
                "type": "create_2d",
                "x_min": mesh_init.get("x_min", 0), "x_max": mesh_init.get("x_max", 1),
                "y_min": mesh_init.get("y_min", 0), "y_max": mesh_init.get("y_max", 1),
                "nx": mesh_init.get("nx", 50), "ny": mesh_init.get("ny", 50),
            }
        else:
            case["mesh"] = {"type": "create_1d", "domain": [0, 1], "n_cells": 100}

        case["solver"] = {"time_end": 0.1, "cfl": 0.45, "output_snapshots": 10}
        return case

    def save(self, path):
        save_project_zip(self, path)

    def load(self, path):
        load_project_zip(path, self)

    def list_cards(self, tab=None):
        if tab:
            return [c for c in self.cards.values() if c.tab == tab]
        return list(self.cards.values())

    @classmethod
    def from_config(cls, cards_json):
        if isinstance(cards_json, str):
            with open(cards_json) as f:
                cards_json = json.load(f)

        proj = cls()
        session = Session()
        proj.add_session(session)

        for tab in cards_json.get("tabs", []):
            tab_id = tab["id"]
            if tab.get("type") != "cards":
                continue

            first_card_id = None
            for entry in tab.get("cards", []):
                subtab = entry.get("subtab", "")
                card = Card.from_config_entry(entry, tab_id, subtab)
                proj.cards[card.id] = card
                proj.defaults[card.id] = copy.deepcopy(card)
                if first_card_id is None:
                    first_card_id = card.id

            if first_card_id:
                proj.selections[tab_id] = first_card_id

        return proj
