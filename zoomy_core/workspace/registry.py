import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field


@dataclass
class BackendRegistry:
    connections: dict = field(default_factory=dict)

    def connect(self, url, timeout=2):
        url = url.rstrip("/")
        try:
            req = urllib.request.Request(url + "/api/v1/health")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    tag = data.get("tag", "unknown")
                    self.connections[tag] = url
                    return tag
        except Exception:
            pass
        return None

    def disconnect(self, tag):
        self.connections.pop(tag, None)

    def is_connected(self, tag):
        if tag == "numpy":
            return True
        return tag in self.connections

    def get_url(self, tag):
        return self.connections.get(tag)

    def available_tags(self):
        tags = ["numpy"]
        for tag in self.connections:
            if tag not in tags:
                tags.append(tag)
        return tags
