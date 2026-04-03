import argparse
import json
import os
import sys

from zoomy_core.workspace import Project, BackendRegistry

STATE_DIR = ".zoomy"
STATE_FILE = os.path.join(STATE_DIR, "state.json")
CONFIG_FILE = os.path.join("library", "zoomy_gui", "standalone", "cards.json")


def find_config():
    for candidate in [CONFIG_FILE, "cards.json", os.path.join("standalone", "cards.json")]:
        if os.path.exists(candidate):
            return candidate
    return None


def get_project():
    config = find_config()
    proj = Project.from_config(config) if config else Project()

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            data = json.load(f)
        if data.get("selections"):
            proj.selections.update(data["selections"])

    return proj


def save_state(proj):
    os.makedirs(STATE_DIR, exist_ok=True)
    data = {
        "selections": proj.selections,
        "active_session": proj.active_session,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def cmd_status(proj, args):
    s = proj.status()
    for k, v in s.items():
        print(f"  {k:10s}: {v}")


def cmd_list(proj, args):
    tab = args.tab
    cards = proj.list_cards(tab)
    if not cards:
        print(f"No cards found" + (f" for tab '{tab}'" if tab else ""))
        return
    sel = proj.selections.get(tab, "")
    for c in cards:
        marker = " *" if c.id == sel else ""
        print(f"  {c.id:30s} {c.title}{marker}")


def cmd_select(proj, args):
    card_id = args.card_id
    if not card_id.startswith("card-"):
        card_id = "card-" + card_id
    if card_id not in proj.cards:
        print(f"Unknown card: {card_id}")
        print("Available:", ", ".join(proj.cards.keys()))
        return
    card = proj.cards[card_id]
    proj.select(card.tab, card_id)
    save_state(proj)
    print(f"Selected {card.title} in {card.tab}")


def cmd_save(proj, args):
    proj.save(args.path)
    print(f"Project saved to {args.path}")


def cmd_load(proj, args):
    count = proj.load(args.path)
    save_state(proj)
    print(f"Loaded {count} cards from {args.path}")


def cmd_show(proj, args):
    card_id = args.card_id
    if not card_id.startswith("card-"):
        card_id = "card-" + card_id
    card = proj.cards.get(card_id)
    if not card:
        print(f"Unknown card: {card_id}")
        return
    print(f"Title:       {card.title}")
    print(f"Tab:         {card.tab}")
    print(f"Description: {card.description[:100]}")
    if card.class_path:
        print(f"Class:       {card.class_path}")
    if card.code:
        print(f"Code:\n{card.code}")


def cmd_params(proj, args):
    card_id = args.card_id
    if not card_id.startswith("card-"):
        card_id = "card-" + card_id
    card = proj.cards.get(card_id)
    if not card or not card.class_path:
        print("No class_path for this card")
        return
    from zoomy_core.workspace.params import extract_params
    schema = extract_params(card.class_path, card.init)
    for name, p in schema["params"].items():
        print(f"  {name:25s} {p['type']:15s} default={p.get('default')}")


def cmd_case(proj, args):
    try:
        case = proj.build_case()
        print(json.dumps(case, indent=2))
    except ValueError as e:
        print(f"Error: {e}")


def cmd_backends(proj, args):
    reg = BackendRegistry()
    if args.connect:
        tag = reg.connect(args.connect)
        if tag:
            print(f"Connected to {tag} at {args.connect}")
        else:
            print(f"Failed to connect to {args.connect}")
        return
    print("Available backends:")
    for tag in reg.available_tags():
        url = reg.get_url(tag) or "(built-in)"
        print(f"  {tag:15s} {url}")


def cmd_run(proj, args):
    try:
        case = proj.build_case()
    except ValueError as e:
        print(f"Error: {e}")
        return

    from zoomy_client import ZoomyClient
    client = ZoomyClient(args.url or "http://localhost:8080")
    try:
        job_id = client.submit(case)
        print(f"Job submitted: {job_id}")
        if args.wait:
            print("Waiting for completion...")
            result = client.wait(job_id, timeout=args.timeout)
            print(f"Status: {result['status']}")
            if result.get("error"):
                print(f"Error: {result['error'][:200]}")
    except Exception as e:
        print(f"Failed: {e}")


def main():
    parser = argparse.ArgumentParser(prog="zoomy", description="Zoomy CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show current selections")

    p_list = sub.add_parser("list", help="List available cards")
    p_list.add_argument("tab", nargs="?", help="Filter by tab (model, mesh, solver, visualization)")

    p_sel = sub.add_parser("select", help="Select a card")
    p_sel.add_argument("card_id", help="Card ID (e.g. smt, mesh-1d)")

    p_save = sub.add_parser("save", help="Save project to zip")
    p_save.add_argument("path", default="zoomy-project.zip", nargs="?")

    p_load = sub.add_parser("load", help="Load project from zip")
    p_load.add_argument("path")

    p_show = sub.add_parser("show", help="Show card details")
    p_show.add_argument("card_id")

    p_params = sub.add_parser("params", help="Show extractable params")
    p_params.add_argument("card_id")

    sub.add_parser("case", help="Print ZoomyCase JSON from current selections")

    p_back = sub.add_parser("backends", help="List/connect backends")
    p_back.add_argument("--connect", help="URL to connect")

    p_run = sub.add_parser("run", help="Submit simulation job")
    p_run.add_argument("--url", help="Backend URL", default="http://localhost:8080")
    p_run.add_argument("--wait", action="store_true", help="Wait for completion")
    p_run.add_argument("--timeout", type=float, default=600)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    proj = get_project()

    cmds = {
        "status": cmd_status, "list": cmd_list, "select": cmd_select,
        "save": cmd_save, "load": cmd_load, "show": cmd_show,
        "params": cmd_params, "case": cmd_case, "backends": cmd_backends,
        "run": cmd_run,
    }
    cmds[args.command](proj, args)


if __name__ == "__main__":
    main()
