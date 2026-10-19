"""Incremental SSE and sentence JSON decoding; never expose wire metadata to speech."""

import json
import re

import httpx


class SentenceDecoder:
    """Yield completed objects from {"segments":[...]} before the JSON document ends."""

    def __init__(self):
        self.buffer = ""
        self.state = "prefix"
        self.count = 0
        self.total = 0
        self.decoder = json.JSONDecoder()

    def feed(self, text: str):
        self.buffer += text
        self.total += len(text)
        if self.total > 32768 or len(self.buffer) > 8192:
            raise ValueError("streamed answer exceeded size limit")
        while True:
            self.buffer = self.buffer.lstrip()
            if self.state == "prefix":
                match = re.match(r'^\{\s*"segments"\s*:\s*\[', self.buffer)
                if not match:
                    return
                self.buffer = self.buffer[match.end() :]
                self.state = "item"
            elif self.state == "item":
                if not self.buffer:
                    return
                if self.buffer.startswith("]"):
                    raise ValueError("empty sentence or trailing comma")
                try:
                    item, end = self.decoder.raw_decode(self.buffer)
                except json.JSONDecodeError:
                    return  # The next SSE delta may complete the object or an escaped string.
                if not isinstance(item, dict):
                    raise ValueError("sentence must be an object")
                self.count += 1
                if self.count > 16:
                    raise ValueError("too many sentences")
                self.buffer = self.buffer[end:]
                self.state = "separator"
                yield item
            elif self.state == "separator":
                if not self.buffer:
                    return
                char, self.buffer = self.buffer[0], self.buffer[1:]
                if char == ",":
                    self.state = "item"
                elif char == "]":
                    self.state = "close"
                else:
                    raise ValueError("invalid sentence separator")
            elif self.state == "close":
                if not self.buffer:
                    return
                if self.buffer[0] != "}":
                    raise ValueError("invalid answer envelope")
                self.buffer = self.buffer[1:]
                self.state = "done"
            else:
                if self.buffer:
                    raise ValueError("unexpected content after answer")
                return

    def finish(self):
        if self.state != "done" or self.buffer.strip() or not self.count:
            raise ValueError("incomplete streamed answer")


async def chat_events(response: httpx.Response):
    """Read SSE events across arbitrary TCP/UTF-8 boundaries, including keepalives."""
    data = []
    size = 0
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            size += len(line)
            if size > 65536:
                raise ValueError("oversized model event")
        elif not line and data:
            raw = "\n".join(data)
            data, size = [], 0
            if raw == "[DONE]":
                return
            event = json.loads(raw)
            if not isinstance(event, dict) or "error" in event:
                raise ValueError("invalid model event")
            yield event
    raise ValueError("model stream ended without DONE")
