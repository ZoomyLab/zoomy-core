from dataclasses import dataclass, field


@dataclass
class Card:
    id: str
    tab: str = ""
    subtab: str = ""
    title: str = ""
    description: str = ""
    code: str = ""
    params: dict = field(default_factory=dict)
    class_path: str = ""
    init: dict = field(default_factory=dict)
    requires_tag: str = ""
    mesh_sizes: list = field(default_factory=list)
    snippet: str = ""
    template: str = ""
    has_timeline: bool = False
    preview: str = ""

    def is_modified(self, default):
        return (self.title != default.title
                or self.description != default.description
                or self.code != default.code
                or self.params != default.params)

    def to_dict(self):
        d = {}
        for k in ["id", "tab", "subtab", "title", "description", "params", "class_path", "init"]:
            v = getattr(self, k)
            if v:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_config_entry(cls, entry, tab_id, subtab=""):
        code = entry.get("template", "") or entry.get("snippet", "")
        return cls(
            id="card-" + entry["id"],
            tab=tab_id,
            subtab=subtab,
            title=entry.get("title", ""),
            description=entry.get("description", ""),
            code=code,
            params={},
            class_path=entry.get("class", ""),
            init=entry.get("init", {}),
            requires_tag=entry.get("requires_tag", ""),
            mesh_sizes=entry.get("mesh_sizes", []),
            snippet=entry.get("snippet", ""),
            template=entry.get("template", ""),
            has_timeline=entry.get("has_timeline", False),
            preview=entry.get("preview", ""),
        )
