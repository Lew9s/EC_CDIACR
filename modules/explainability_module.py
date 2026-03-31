import json
import re
import ollama
import numpy as np
import networkx as nx
from collections import OrderedDict
from pyvis.network import Network


class ExplainabilityModule:
    def __init__(self, llm, kg_client, embedding_model="bge-m3"):
        self.llm = llm
        self.kg = kg_client
        self.embedding_model = embedding_model

        self.embedding_cache = {}
        self.VALID_ENTITIES = {"CHANGE_ORDER", "COMPONENT", "DEPARTMENT", "REASON", "TIME_POINT"}

        self.VALID_RELATIONS = {
            "MODIFIES",
            "REQUIRES",
            "AFFECTS",
            "CONSTRAINED_BY",
            "PART_OF"
        }

        self.REL_MAP = {
            "needs": "REQUIRES",
            "depends_on": "REQUIRES",
            "influences": "AFFECTS"
        }

        self.TYPE_COLOR = {
            "CHANGE_ORDER": "red",
            "COMPONENT": "blue",
            "DEPARTMENT": "green",
            "REASON": "orange",
            "TIME_POINT": "pink"
        }

    # ========================
    # Utils
    # ========================
    def embed(self, text):
        text = text.strip()

        if text in self.embedding_cache:
            return self.embedding_cache[text]

        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )

        emb = response["embedding"]
        self.embedding_cache[text] = emb
        return emb

    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ========================
    # JSON Clean
    # ========================
    def clean_llm_output(self, text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group() if match else text

    def safe_json_parse(self, text):
        try:
            return json.loads(text)["triples"]
        except:
            print("⚠️ JSON parse failed")
            return []

    # ========================
    # Text Chunking
    # ========================
    def split_text(self, text, max_len=500):
        sentences = re.split(r"[。；\n]", text)
        chunks = []
        current = ""

        for s in sentences:
            if len(current) + len(s) < max_len:
                current += s
            else:
                chunks.append(current)
                current = s

        if current:
            chunks.append(current)

        return chunks

    # ========================
    # Triple Extraction
    # ========================
    def extract_triples(self, text):
        prompt = f"""
You are an information extraction system specialized in marine engineering.

Extract triples from the engineering proposal text.

STRICT RULES:
1. Output ONLY valid JSON.
2. Do NOT explain, add markdown, or extra text.
3. Use ONLY the given entity and relation types.
4. Do NOT invent new relations or entities.
5. If no valid triples exist, return an empty list.
6. **CRITICAL**: Do NOT force a relation if the text does not explicitly support it.

Entities Definition:

- CHANGE_ORDER: The specific engineering change order document or proposal ID.
- COMPONENT: **Tangible, identifiable hardware, machinery, or structural elements of the vessel.**
  - ✅ INCLUDE: Engines, pumps, valves, pipes, hull plates, decks, ladders, generators, tanks.
  - ❌ EXCLUDE: Materials (steel, paint, oil), Processes (welding, cutting, installation), Documents, Departments, Abstract concepts (safety, efficiency, pressure), Personnel.
- DEPARTMENT: Organizational unit responsible for signing or executing changes.
- REASON: Motivation or justification for the change.
- TIME_POINT: Specific date or time associated with the change.

Relations Definition:

- MODIFIES: (CHANGE_ORDER -> COMPONENT) The order directly alters, adds, or removes a physical structure.
- REQUIRES: (COMPONENT -> COMPONENT) One component strictly needs another specific component to function. **Do not use for materials or general conditions.**
- AFFECTS: (COMPONENT -> COMPONENT) A component impacts another component or a specific performance metric.
- CONSTRAINED_BY: (COMPONENT/CHANGE_ORDER -> REASON) Restricted by codes, regulations, or specs.
- PART_OF: (COMPONENT -> COMPONENT) Hierarchical relationship (e.g., Piston PART_OF Engine).

Common Mistakes to Avoid:
- "Welding" is a process, NOT a COMPONENT.
- "Steel" is a material, NOT a COMPONENT.
- "Safety" is a concept, NOT a COMPONENT.
- "Approval" is a process/status, NOT a COMPONENT.

Output Format:
{{
    "triples": [
        {{
            "head": "entity text",
            "head_type": "ENTITY_TYPE",
            "relation": "RELATION_TYPE",
            "tail": "entity text",
            "tail_type": "ENTITY_TYPE"
        }}
    ]
}}

Example 1 (Positive):
Text: Install new main engine and connect to fuel pump.
Output:
{{
    "triples": [
        {{
            "head": "main engine",
            "head_type": "COMPONENT",
            "relation": "REQUIRES",
            "tail": "fuel pump",
            "tail_type": "COMPONENT"
        }}
    ]
}}

Example 2 (Negative Constraint):
Text: Reinforce deck with steel plates and welding.
Output:
{{
    "triples": [
        {{
            "head": "steel plates",
            "head_type": "COMPONENT",
            "relation": "PART_OF",
            "tail": "deck",
            "tail_type": "COMPONENT"
        }}
    ]
}}
*Note: "welding" is ignored as it is a process, not a COMPONENT.*

Now extract:

Text:
{text}
"""
        response = ollama.chat(
            model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            think=False
        )
        content = response.get("message", {}).get("content", "")
        content = self.clean_llm_output(content)
        return self.safe_json_parse(content)

    # ========================
    # Hybrid Entity Matching
    # ========================
    def match_entity(self, name, expected_type=None, threshold=0.85):
        name = name.strip()
        # ---- 1. Exact Match ----
        query = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($name)
        RETURN n LIMIT 1
        """
        result = self.kg.run(query, {"name": name}).data()

        if result:
            return result[0]["n"]["name"], False, "exact", 1.0

        # ---- 2. Embedding Match ----
        embedding = self.embed(name)

        query = """
        CALL db.index.vector.queryNodes(
            'node_embedding_index',
            5,
            $embedding
        )
        YIELD node, score
        RETURN node.name AS name, node.type AS type, score
        """

        results = self.kg.run(query, {"embedding": embedding}).data()

        best = None
        best_score = 0

        for r in results:
            if expected_type and r["type"] != expected_type:
                continue

            if r["score"] > best_score:
                best = r
                best_score = r["score"]

        if best and best_score >= threshold:
            return best["name"], False, "semantic", float(best_score)

        return name, True, "new", 0.0

    # ========================
    # Relation Normalize
    # ========================
    def normalize_relation(self, rel):
        return self.REL_MAP.get(rel.lower(), rel.upper())

    def relation_exists(self, head, rel, tail):
        query = f"""
        MATCH (a {{name:$head}})-[r:{rel}]->(b {{name:$tail}})
        RETURN r LIMIT 1
        """
        result = self.kg.run(query, {"head": head, "tail": tail}).data()
        return len(result) > 0

    # ========================
    # Filter triples
    # ========================
    def filter_valid_triples(self, triples):
        valid = []
        for t in triples:
            try:
                if not all([t.get("head"), t.get("tail"), t.get("relation"),
                            t.get("head_type"), t.get("tail_type")]):
                    continue

                if t["head_type"] not in self.VALID_ENTITIES:
                    continue
                if t["tail_type"] not in self.VALID_ENTITIES:
                    continue

                rel = self.normalize_relation(t["relation"])
                if rel not in self.VALID_RELATIONS:
                    continue

                valid.append({
                    "head": t["head"].strip(),
                    "tail": t["tail"].strip(),
                    "relation": rel,
                    "head_type": t["head_type"],
                    "tail_type": t["tail_type"]
                })
            except:
                continue

        return valid

    # ========================
    # Align triples
    # ========================
    def align_triples(self, triples):
        aligned = []

        for t in triples:
            head, head_new, h_type, h_score = self.match_entity(
                t["head"], t["head_type"]
            )
            tail, tail_new, t_type, t_score = self.match_entity(
                t["tail"], t["tail_type"]
            )

            exists = self.relation_exists(head, t["relation"], tail)

            aligned.append({
                "head": head,
                "tail": tail,
                "relation": t["relation"],
                "is_new": not exists,
                "head_new": head_new,
                "tail_new": tail_new,
                "head_type": t["head_type"],
                "tail_type": t["tail_type"],
                "head_match_type": h_type,
                "tail_match_type": t_type,
                "head_score": h_score,
                "tail_score": t_score
            })

        return aligned

    # ========================
    # Build subgraph
    # ========================
    def build_subgraph(self, triples):
        nodes = OrderedDict()
        edges = []

        for t in triples:
            nodes[t["head"]] = {
                "name": t["head"],
                "type": t["head_type"],
                "is_new": t["head_new"],
                "match_type": t["head_match_type"]
            }

            nodes[t["tail"]] = {
                "name": t["tail"],
                "type": t["tail_type"],
                "is_new": t["tail_new"],
                "match_type": t["tail_match_type"]
            }

            edges.append({
                "head": t["head"],
                "tail": t["tail"],
                "relation": t["relation"],
                "is_new": t["is_new"]
            })

        return {"nodes": list(nodes.values()), "edges": edges}

    # ========================
    # Hybrid Graph
    # ========================
    def build_hybrid_graph(self, subgraph):
        G = nx.DiGraph()

        for e in subgraph["edges"]:
            G.add_edge(e["head"], e["tail"], relation=e["relation"], is_new=e["is_new"])

        for node in subgraph["nodes"]:
            query = """
            MATCH (n {name:$name})-[r]-(m)
            WHERE n.name IS NOT NULL AND m.name IS NOT NULL
            RETURN n.name as n, type(r) as rel, m.name as m
            LIMIT 20
            """
            results = self.kg.run(query, {"name": node["name"]}).data()

            for r in results:
                G.add_edge(r["n"], r["m"], relation=r["rel"], is_new=False)

        return G

    # ========================
    # Path
    # ========================
    def find_hybrid_paths(self, G, subgraph):
        paths = []

        for e in subgraph["edges"]:
            if e["head"] not in G or e["tail"] not in G:
                continue

            try:
                nodes = nx.shortest_path(G, e["head"], e["tail"])
                paths.append({"nodes": nodes})
            except:
                continue

        return paths

    # ========================
    # Visualization
    # ========================
    def visualize(self, subgraph, paths, output_file="fig7.html"):

        # ===== 强制 Node ⊇ Edge =====
        node_names = {n["name"] for n in subgraph["nodes"]}

        for e in subgraph["edges"]:
            if e["head"] not in node_names:
                subgraph["nodes"].append({"name": e["head"], "type": "UNKNOWN", "is_new": True})
            if e["tail"] not in node_names:
                subgraph["nodes"].append({"name": e["tail"], "type": "UNKNOWN", "is_new": True})

        net = Network(height="750px", width="100%")

        for node in subgraph["nodes"]:
            node_type = node.get("type", "UNKNOWN")
            is_new = node.get("is_new", True)
            match_type = node.get("match_type", "new")  
            base_color = self.TYPE_COLOR.get(node_type, "#9E9E9E")
            if is_new:
                color = {
                    "border": base_color,
                    "background": "#FFFFFF"
                }
            else:
                color = {
                    "border": base_color,
                    "background": base_color
                }

            net.add_node(
                node["name"],
                label=node["name"],
                color=color,
                borderWidth=3 if is_new else 0,
            )

        for e in subgraph["edges"]:
            net.add_edge(
                e["head"],
                e["tail"],
                label=e["relation"],
                color="red" if e["is_new"] else "gray",
                dashes=e["is_new"]
            )

        for p in paths:
            for i in range(len(p["nodes"]) - 1):
                net.add_edge(p["nodes"][i], p["nodes"][i + 1], color="blue", width=4)
            
        legend_html = """
<div style="
position:absolute;
top:20px;
left:20px;
background:white;
padding:10px 15px;
border:1px solid #ccc;
border-radius:8px;
font-size:14px;
z-index:999;
box-shadow:0 2px 6px rgba(0,0,0,0.2);
">
<b>Legend</b><br>

<span style="display:inline-block;width:12px;height:12px;border:2px solid black;background:white;margin-right:6px;"></span> New Node<br>
<br>
<span style="color:red;">■</span> CHANGE_ORDER<br>
<span style="color:blue;">■</span> COMPONENT<br>
<span style="color:green;">■</span> DEPARTMENT<br>
<span style="color:orange;">■</span> REASON<br>
<span style="color:pink;">■</span> TIME_POINT<br>
</div>
"""
        # Generate HTML content instead of writing to file
        html_content = net.html()
        html_content = html_content.replace("<body>", "<body>" + legend_html)
        return html_content

    # ========================
    # Pipeline
    # ========================
    def run(self, proposal_text, visualize=False):
        chunks = self.split_text(proposal_text)

        all_triples = []
        for c in chunks:
            triples = self.extract_triples(c)
            triples = self.filter_valid_triples(triples)
            all_triples.extend(triples)
        aligned = self.align_triples(all_triples)
        subgraph = self.build_subgraph(aligned)

        G = self.build_hybrid_graph(subgraph)
        paths = self.find_hybrid_paths(G, subgraph)

        if visualize:
            self.visualize(subgraph, paths)

        return {
            "nodes": subgraph["nodes"],
            "edges": subgraph["edges"],
            "paths": paths
        }