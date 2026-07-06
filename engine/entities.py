from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern

STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "that",
    "this", "these", "those", "it", "its", "as", "from", "which", "such",
    "can", "may", "will", "shall", "should", "has", "have", "had",
}

REF_PATTERNS: List[Pattern] = [
    re.compile(r"\b(Section|Sec\.?|Clause)\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE),
    re.compile(r"\b(Table)\s+([0-9]+)", re.IGNORECASE),
    re.compile(r"\b(Figure|Fig\.?|Figures?)\s+([0-9]+)", re.IGNORECASE),
    re.compile(r"\b(Appendix)\s+([A-Z0-9]+)", re.IGNORECASE),
]


@dataclass
class Query:
    text: str
    keywords: List[str]
    entities: List[str]
    raw_refs: List[str]


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"\b[0-9]+(?:\.[0-9]+)?%?|[a-zA-Z][a-zA-Z0-9\-']*\b", text.lower())
    return [token for token in tokens if token]


def stem(tokens: List[str]) -> List[str]:
    stems = []
    for token in tokens:
        if len(token) <= 3:
            stems.append(token)
            continue
        if token.endswith("ing"):
            stems.append(token[:-3])
        elif token.endswith("ed"):
            stems.append(token[:-2])
        elif token.endswith("es"):
            stems.append(token[:-2])
        elif token.endswith("s") and not token.endswith("ss"):
            stems.append(token[:-1])
        elif token.endswith("ly"):
            stems.append(token[:-2])
        else:
            stems.append(token)
    return stems


def extract_keywords(text: str, stopwords: Optional[set[str]] = None) -> List[str]:
    if stopwords is None:
        stopwords = STOPWORDS
    source = text or ""
    tokens = tokenize(source)
    keywords = [stem([token])[0] for token in tokens if token not in stopwords]
    return [t for t in keywords if t]


def extract_entities(text: str) -> List[str]:
    if not text:
        return []
    entities: List[str] = []
    seen = set()

    def add_entity(candidate: str) -> None:
        normalized = candidate.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            entities.append(normalized)

    for quote in re.findall(r'"([^"]+)"|\“([^”]+)\”|\‘([^’]+)\’', text):
        for match in quote:
            if match:
                add_entity(match)

    for match in re.findall(r"\b\d+(?:\.\d+)?(?:\s*(?:dim|dims|heads|tokens|kb|mb|gb|m|cm|mm|s|ms|μs|%)\b)?", text, re.IGNORECASE):
        add_entity(match)

    for acronym in re.findall(r"\b[A-Z]{2,}\b", text):
        add_entity(acronym)

    for span in re.findall(r"\b(?:[A-Z][a-z0-9]+\s+){1,}[A-Z][a-z0-9]+\b", text):
        add_entity(span)

    return entities


def detect_raw_refs(text: str) -> List[str]:
    refs: List[str] = []
    seen = set()
    for pattern in REF_PATTERNS:
        for match in pattern.finditer(text or ""):
            raw = match.group(0)
            key = raw.lower().strip()
            if key not in seen:
                seen.add(key)
                refs.append(raw)
    return refs


def build_query(text: str) -> Query:
    keywords = extract_keywords(text)
    entities = extract_entities(text)
    raw_refs = detect_raw_refs(text)
    return Query(text=text, keywords=keywords, entities=entities, raw_refs=raw_refs)
