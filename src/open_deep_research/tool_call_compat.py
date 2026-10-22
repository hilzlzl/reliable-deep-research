"""Compatibility parsing for local models that emit native tool-call text."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


_CALL_START = re.compile(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*\{")


def _tool_schemas(tools: list[Any]) -> dict[str, dict[str, Any]]:
    schemas = {}
    for tool in tools:
        converted = convert_to_openai_tool(tool)["function"]
        schemas[converted["name"]] = converted.get("parameters", {})
    return schemas


def _extract_calls(text: str) -> list[tuple[str, str]]:
    calls = []
    position = 0
    while match := _CALL_START.search(text, position):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            break
        calls.append((match.group(1), text[match.end() : cursor - 1].strip()))
        position = cursor
    return calls


def _strip_native_quotes(value: str) -> str:
    value = value.strip()
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[5:-5]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _coerce(value: str, schema: dict[str, Any]) -> Any:
    value = value.strip()
    kind = schema.get("type", "string")
    if kind == "array" and value.startswith("[") and value.endswith("]"):
        item_schema = schema.get("items", {})
        return [
            _coerce(item, item_schema)
            for item in value[1:-1].split(",")
            if item.strip()
        ]
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        return value.lower() == "true"
    return _strip_native_quotes(value)


def _parse_arguments(raw: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads("{" + raw + "}")
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    properties = parameters.get("properties", {})
    locations = []
    for key in properties:
        match = re.search(rf"(?:^|,)\s*{re.escape(key)}\s*:", raw)
        if match:
            locations.append((match.start(), match.end(), key))
    locations.sort()
    arguments = {}
    for index, (_, value_start, key) in enumerate(locations):
        value_end = locations[index + 1][0] if index + 1 < len(locations) else len(raw)
        value = raw[value_start:value_end].strip().rstrip(",")
        arguments[key] = _coerce(value, properties[key])
    return arguments


def normalize_text_tool_calls(message: AIMessage, tools: list[Any]) -> AIMessage:
    """Convert recognized native call text into standard LangChain tool calls."""
    if message.tool_calls:
        return message
    schemas = _tool_schemas(tools)
    parsed_calls = []
    for index, (name, raw_arguments) in enumerate(_extract_calls(str(message.content))):
        if name not in schemas:
            continue
        digest = hashlib.sha256(
            f"{name}\0{raw_arguments}\0{index}".encode("utf-8")
        ).hexdigest()[:16]
        parsed_calls.append(
            {
                "name": name,
                "args": _parse_arguments(raw_arguments, schemas[name]),
                "id": f"compat-{digest}",
                "type": "tool_call",
            }
        )
    if not parsed_calls:
        return message
    return message.model_copy(update={"content": "", "tool_calls": parsed_calls})
