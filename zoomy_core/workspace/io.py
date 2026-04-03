import json
import zipfile
import io
import re


def _safe_name(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_-]", "_", s)).strip("_").lower()


def save_project_zip(project, path):
    buf = io.BytesIO() if not isinstance(path, str) else None
    target = buf or path

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        meta = {
            "version": "1.0",
            "sessions": [s.to_dict() for s in project.sessions.values()],
            "activeSession": project.active_session,
            "selections": project.selections,
        }
        zf.writestr("project.json", json.dumps(meta, indent=2))

        saved = 0
        for session in project.sessions.values():
            session_folder = _safe_name(session.title)

            for card_id, card in project.cards.items():
                default = project.defaults.get(card_id)
                if default and not card.is_modified(default):
                    continue

                folder = session_folder + "/" + card.tab
                if card.subtab:
                    folder += "/" + card.subtab
                folder += "/" + _safe_name(card.title)

                card_meta = card.to_dict()
                zf.writestr(folder + "/card.json", json.dumps(card_meta, indent=2))

                if card.code:
                    zf.writestr(folder + "/code.py", card.code)
                saved += 1

    if buf:
        return buf.getvalue()
    return saved


def load_project_zip(path, project):
    if isinstance(path, bytes):
        zf = zipfile.ZipFile(io.BytesIO(path))
    else:
        zf = zipfile.ZipFile(path, "r")

    with zf:
        project_json = None
        card_files = {}

        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if name.endswith("project.json"):
                project_json = json.loads(zf.read(name))
                continue

            parts = name.split("/")
            filename = parts[-1]
            folder = "/".join(parts[:-1])

            if folder not in card_files:
                card_files[folder] = {}
            if filename == "card.json":
                card_files[folder]["meta"] = json.loads(zf.read(name))
            elif filename == "code.py":
                card_files[folder]["code"] = zf.read(name).decode()

        if project_json:
            from zoomy_core.workspace.session import Session
            if project_json.get("sessions"):
                project.sessions = {}
                for sd in project_json["sessions"]:
                    s = Session.from_dict(sd)
                    project.sessions[s.id] = s
                project.active_session = project_json.get("activeSession", "")
            if project_json.get("selections"):
                project.selections.update(project_json["selections"])

        restored = 0
        for folder, files in card_files.items():
            meta = files.get("meta")
            if not meta:
                continue

            target_id = meta.get("id")
            if target_id and target_id in project.cards:
                card = project.cards[target_id]
            else:
                target_id = None
                for cid, c in project.cards.items():
                    if c.title == meta.get("title"):
                        target_id = cid
                        card = c
                        break
                if not target_id:
                    for cid, c in project.defaults.items():
                        if c.title == meta.get("title"):
                            target_id = cid
                            card = project.cards.get(cid)
                            if not card:
                                continue
                            break

            if not target_id:
                continue

            card.title = meta.get("title", card.title)
            card.description = meta.get("description", card.description)
            card.params = meta.get("params", card.params)
            if files.get("code"):
                card.code = files["code"]

            restored += 1

    return restored
