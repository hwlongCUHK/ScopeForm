"""Phase-local notes: store actual tool excerpts and agent-written interpretations."""

import copy

_KEY_FIELDS = ("item_id", "document_id", "anchor_id")
_NOTE_FIELDS = ("scope", "assertion", "action_stage", "conditions")
_SOURCE_FIELDS = ("document_id", "anchor_id", "text", "title", "role", "label")


def _patch_schema():
    value = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}
    properties = {key: {"type": "string"} for key in _KEY_FIELDS}
    properties.update({key: copy.deepcopy(value) for key in _NOTE_FIELDS})
    return {
        "type": "array",
        "description": "可选证据笔记。同一item_id/document_id/anchor_id的新笔记替换旧笔记，其他来源保留。",
        "items": {"type": "object", "properties": properties,
                  "required": list(_KEY_FIELDS), "additionalProperties": False},
    }


class Notes:
    """Create a fresh instance each phase; interpretation remains the agent's job."""

    def __init__(self):
        self._sources = {}
        self._notes = {}

    def capture(self, name, result):
        """Store only successful read_source/search_sources return values."""
        if not isinstance(result, dict) or "error" in result or result.get("ok") is False:
            return 0
        if name == "read_source":
            rows = [{**block, "document_id": result["document_id"], "title": result["title"]}
                    for block in result["blocks"]]
        elif name == "search_sources":
            rows = result["matches"]
        else:
            return 0
        for row in rows:
            key = (row["document_id"], row["anchor_id"])
            record = {field: copy.deepcopy(row[field]) for field in _SOURCE_FIELDS if field in row}
            # A search need not repeat a title already returned by read_source.
            self._sources[key] = {**self._sources.get(key, {}), **record}
        return len(rows)

    def update(self, patch):
        """Replace each keyed note, without interpreting or repairing its contents."""
        if patch is None:
            return 0
        if not isinstance(patch, list):
            raise TypeError("evidence_patch must be an array")
        for note in patch:
            key = tuple(note[field] for field in _KEY_FIELDS)
            if not all(isinstance(value, str) for value in key):
                raise TypeError("item_id, document_id and anchor_id must be strings")
            self._notes[key] = copy.deepcopy(note)
        return len(patch)

    def tool_specs(self, original_tools):
        """Add an optional patch to copies of the existing native tool schemas."""
        copied = copy.deepcopy(original_tools)
        for tool in copied:
            tool["function"]["parameters"]["properties"]["evidence_patch"] = _patch_schema()
        return copied

    def context(self, include_source_text=False):
        """Group notes by claim; optionally include deduplicated excerpts for debug."""
        claims, referenced = {}, {}
        for note in self._notes.values():
            item_id = note["item_id"]
            key = (note["document_id"], note["anchor_id"])
            entry = {field: copy.deepcopy(value) for field, value in note.items() if field != "item_id"}
            claims.setdefault(item_id, {"item_id": item_id, "sources": []})["sources"].append(entry)
            if include_source_text and key in self._sources:
                referenced[key] = copy.deepcopy(self._sources[key])
        result = {"claims": list(claims.values())}
        if include_source_text:
            result["sources"] = list(referenced.values())
        return result

    def to_dict(self):
        """Return a JSON-serializable checkpoint for this phase."""
        return {
            "observed_sources": copy.deepcopy(list(self._sources.values())),
            "notes": copy.deepcopy(list(self._notes.values())),
        }

    @classmethod
    def from_dict(cls, state):
        """Restore the same phase's checkpoint without consulting a World."""
        notes = cls()
        notes._sources = {(row["document_id"], row["anchor_id"]): copy.deepcopy(row)
                          for row in state.get("observed_sources", [])}
        notes.update(state.get("notes", []))
        return notes
