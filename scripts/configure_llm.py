"""Set local model configuration through hidden stdin, never process arguments."""

import getpass
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
values = json.loads(getpass.getpass("Model configuration JSON (hidden): "))
allowed = {"LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_THINKING"}
if not set(values).issubset(allowed):
    raise ValueError("Unexpected model setting")
path = root / ".env"
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
seen = set()
for i, line in enumerate(lines):
    key = line.partition("=")[0]
    if key in values:
        lines[i] = key + "=" + str(values[key])
        seen.add(key)
lines.extend(key + "=" + str(value) for key, value in values.items() if key not in seen)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Local model configuration saved. Secret was not printed.")
