from __future__ import annotations

from typing import List, Tuple

from engine.index import DASTIndex


def _node_to_block(node) -> str:
    header = f"[node_id: {node.node_id}] "
    title = f"Title: {node.title}\n" if node.title else ""
    body = node.text or ""
    return header + title + body.strip()


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def build_context(results: List, index: DASTIndex, policy: str = "node_plus_parent", max_tokens: int = 2000) -> Tuple[str, List[str]]:
    included_ids: List[str] = []
    blocks: List[str] = []
    seen = set()

    def add_node(node) -> None:
        if node.node_id in seen:
            return
        seen.add(node.node_id)
        included_ids.append(node.node_id)
        blocks.append(_node_to_block(node))

    for result in results:
        node = index.nodes_by_id.get(result.node_id)
        if node is None:
            continue

        if policy == "node_only":
            add_node(node)
        elif policy == "node_plus_parent":
            add_node(node)
            parent_id = index.parent_of.get(node.node_id)
            if parent_id:
                parent = index.nodes_by_id.get(parent_id)
                if parent:
                    add_node(parent)
        elif policy == "node_plus_siblings":
            add_node(node)
            parent_id = index.parent_of.get(node.node_id)
            if parent_id:
                for sibling_id in index.children_of.get(parent_id, []):
                    sibling = index.nodes_by_id.get(sibling_id)
                    if sibling:
                        add_node(sibling)
        elif policy == "subtree":
            add_node(node)
            for descendant_id in index.descendants(node.node_id):
                descendant = index.nodes_by_id.get(descendant_id)
                if descendant:
                    add_node(descendant)
        else:
            add_node(node)

    context = []
    token_sum = 0
    truncated_ids: List[str] = []
    for block, node_id in zip(blocks, included_ids):
        block_tokens = _token_count(block)
        if token_sum + block_tokens > max_tokens:
            break
        context.append(block)
        truncated_ids.append(node_id)
        token_sum += block_tokens

    return "\n\n".join(context), truncated_ids
