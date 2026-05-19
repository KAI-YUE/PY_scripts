#!/usr/bin/env python3
"""Inspect HM slot save files without running the game.

Reads ROOT/slots/slot_XX.hm and ROOT/slots/slot_XX_meta.hm, decompresses the
LÖVE deflate payload, parses the Lua table subset emitted by FileIO.pickle_pack,
and prints JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Any

class LuaTableParser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        if self.text.startswith("return"):
            self.text = self.text[6:].lstrip()
        self.i = 0

    def parse(self) -> Any:
        value = self._value()
        self._ws()
        if self.i != len(self.text):
            raise ValueError(f"trailing input at byte {self.i}: {self.text[self.i:self.i + 40]!r}")
        return value

    def _ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def _peek(self) -> str:
        self._ws()
        return self.text[self.i] if self.i < len(self.text) else ""

    def _expect(self, ch: str) -> None:
        self._ws()
        if self.i >= len(self.text) or self.text[self.i] != ch:
            got = self.text[self.i:self.i + 1]
            raise ValueError(f"expected {ch!r} at byte {self.i}, got {got!r}")
        self.i += 1

    def _value(self) -> Any:
        self._ws()
        ch = self._peek()
        if ch == "{":
            return self._table()
        if ch == '"':
            return self._string()
        if self.text.startswith("true", self.i):
            self.i += 4
            return True
        if self.text.startswith("false", self.i):
            self.i += 5
            return False
        if self.text.startswith("nil", self.i):
            self.i += 3
            return None
        return self._number()

    def _table(self) -> Any:
        self._expect("{")
        out: dict[Any, Any] = {}
        while True:
            self._ws()
            if self._peek() == "}":
                self.i += 1
                return self._maybe_list(out)
            self._expect("[")
            key = self._value()
            self._expect("]")
            self._expect("=")
            out[key] = self._value()
            self._ws()
            if self._peek() == ",":
                self.i += 1

    def _string(self) -> str:
        self._expect('"')
        out: list[str] = []
        while self.i < len(self.text):
            ch = self.text[self.i]
            self.i += 1
            if ch == '"':
                return "".join(out)
            if ch != "\\":
                out.append(ch)
                continue
            if self.i >= len(self.text):
                raise ValueError("unterminated string escape")
            esc = self.text[self.i]
            self.i += 1
            if esc in {'"', "\\", "'"}:
                out.append(esc)
            elif esc == "n":
                out.append("\n")
            elif esc == "r":
                out.append("\r")
            elif esc == "t":
                out.append("\t")
            elif esc == "a":
                out.append("\a")
            elif esc == "b":
                out.append("\b")
            elif esc == "f":
                out.append("\f")
            elif esc == "v":
                out.append("\v")
            elif esc.isdigit():
                digits = esc
                while self.i < len(self.text) and len(digits) < 3 and self.text[self.i].isdigit():
                    digits += self.text[self.i]
                    self.i += 1
                out.append(chr(int(digits, 10)))
            else:
                out.append(esc)
        raise ValueError("unterminated string")

    def _number(self) -> int | float:
        self._ws()
        m = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", self.text[self.i :])
        if not m:
            raise ValueError(f"expected value at byte {self.i}: {self.text[self.i:self.i + 40]!r}")
        raw = m.group(0)
        self.i += len(raw)
        if any(c in raw for c in ".eE"):
            return float(raw)
        return int(raw)

    @staticmethod
    def _maybe_list(table: dict[Any, Any]) -> Any:
        if not table:
            return {}
        if all(isinstance(k, int) and k >= 1 for k in table):
            keys = sorted(table)
            if keys == list(range(1, len(keys) + 1)):
                return [table[i] for i in keys]
        return table


def decode_hm(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"return"):
        return data.decode("utf-8")
    errors: list[str] = []
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16):
        try:
            return zlib.decompress(data, wbits).decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - report all attempted codecs.
            errors.append(f"wbits={wbits}: {exc}")
    raise ValueError("could not zlib-decompress file:\n" + "\n".join(errors))


def load_hm(path: Path) -> tuple[str, Any]:
    lua = decode_hm(path)
    return lua, LuaTableParser(lua).parse()


def short_count(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "keys": list(value.keys())[:20]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    return value


def summarize_slot(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"value": data}
    world = data.get("world") or {}
    pawns = (world.get("pawns") if isinstance(world, dict) else {}) or {}
    cards = data.get("cards") or {}
    return {
        "schema_version": data.get("schema_version"),
        "top_keys": list(data.keys()),
        "meta": data.get("meta"),
        "story": short_count(data.get("story")),
        "world_field_keys": list(((world.get("field") or {}) if isinstance(world, dict) else {}).keys()),
        "pawn_counts": {k: len(v) if isinstance(v, list) else 0 for k, v in pawns.items()} if isinstance(pawns, dict) else {},
        "card_zones": short_count((cards.get("zones") if isinstance(cards, dict) else None)),
        "run": short_count(data.get("run")),
    }


def summarize_meta(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"value": data}
    out = dict(data)
    for key in ("unlocked", "discovered", "alerted"):
        if key in out:
            out[key] = short_count(out[key])
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect HM slot save files.")
    ap.add_argument("slot_idx", type=int, help="Slot number, e.g. 1 for slot_01.hm")
    ap.add_argument("--root", default="saves", help="Save root directory. Default: saves")
    ap.add_argument("--slots-root", help="Explicit slots directory. Default: ROOT/slots")
    ap.add_argument("--full", action="store_true", help="Print full parsed JSON instead of summaries")
    ap.add_argument("--raw-lua", action="store_true", help="Also include decompressed Lua table text")
    args = ap.parse_args()

    slots_root = Path(args.slots_root) if args.slots_root else Path(args.root) / "slots"
    slot_path = slots_root / f"slot_{args.slot_idx:02d}.hm"
    meta_path = slots_root / f"slot_{args.slot_idx:02d}_meta.hm"

    result: dict[str, Any] = {"slot_idx": args.slot_idx, "paths": {"slot": str(slot_path), "meta": str(meta_path)}}
    for label, path, summary_fn in (("slot", slot_path, summarize_slot), ("meta", meta_path, summarize_meta)):
        if not path.exists():
            result[label] = {"missing": True}
            continue
        try:
            lua, data = load_hm(path)
        except Exception as exc:  # noqa: BLE001 - inspector should report parse failures as data.
            result[label] = {"error": str(exc)}
            continue
        result[label] = json_ready(data if args.full else summary_fn(data))
        if args.raw_lua:
            result[f"{label}_lua"] = lua

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
