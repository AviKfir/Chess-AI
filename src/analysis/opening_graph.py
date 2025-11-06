import re
import pandas as pd
from collections import Counter, defaultdict

# ---- Helpers ----
_SAN_NUM_RE = re.compile(r"\b\d+\.")

def _san_tokens(pgn: str) -> list[str]:
    """Split SAN string, remove move numbers like '12.' """
    if not isinstance(pgn, str) or not pgn:
        return []
    toks = [t for t in pgn.split() if not _SAN_NUM_RE.fullmatch(t)]
    return toks

def _prefix_label(tokens: list[str], ply: int = 4) -> str:
    """Return a coarse 'opening prefix' from first N half-moves (ply)."""
    if not tokens:
        return "UNKNOWN_PREFIX"
    k = min(len(tokens), ply)
    return " ".join(tokens[:k])

# ---- Public: build transitions on a DataFrame ----
def transitions_prefix_to_opening(df: pd.DataFrame, ply: int = 4) -> pd.DataFrame:
    """
    Build transitions from first 'ply' SAN tokens -> dataset 'Opening' name.
    Returns a DataFrame: [prefix, Opening, count]
    """
    rows = []
    pgn_col = None
    for col in ["AN", "Moves", "SAN", "pgn", "PGN"]:
        if col in df.columns:
            pgn_col = col
            break

    if pgn_col is None:
        # nothing to do
        return pd.DataFrame(columns=["prefix", "Opening", "count"])

    # Fill missing opening names
    d = df.copy()
    d["Opening"] = d.get("Opening", "Unknown").fillna("Unknown").astype(str)

    for pgn, opening in zip(d[pgn_col], d["Opening"]):
        toks = _san_tokens(pgn)
        pref = _prefix_label(toks, ply=ply)
        rows.append((pref, opening))

    c = Counter(rows)
    out = pd.DataFrame([(p, o, n) for (p, o), n in c.items()],
                       columns=["prefix", "Opening", "count"])
    out.sort_values(["count"], ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

# ---- Streaming version over full CSV ----
def transitions_prefix_to_opening_stream(reader, ply: int = 4, min_count: int = 100) -> pd.DataFrame:
    """
    Stream chunks from CSV reader and aggregate prefix->Opening counts.
    Returns a DataFrame [prefix, Opening, count] filtered by min_count.
    """
    agg: dict[tuple[str, str], int] = defaultdict(int)

    # find pgn column by peeking at first chunk
    first = next(iter(reader))
    # Re-yield the first chunk (we'll re-create a new iterator)
    pgn_col = None
    for col in ["AN", "Moves", "SAN", "pgn", "PGN"]:
        if col in first.columns:
            pgn_col = col
            break
    if pgn_col is None:
        return pd.DataFrame(columns=["prefix", "Opening", "count"])

    def process_chunk(chunk: pd.DataFrame):
        ch = chunk.copy()
        ch["Opening"] = ch.get("Opening", "Unknown").fillna("Unknown").astype(str)
        for pgn, opening in zip(ch[pgn_col], ch["Opening"]):
            toks = _san_tokens(pgn)
            pref = _prefix_label(toks, ply=ply)
            agg[(pref, opening)] += 1

    # process first chunk and the rest:
    process_chunk(first)
    for chunk in reader:
        process_chunk(chunk)

    rows = [(p, o, n) for (p, o), n in agg.items() if n >= min_count]
    out = pd.DataFrame(rows, columns=["prefix", "Opening", "count"])
    out.sort_values(["count"], ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ========= OPENING PROJECTION, COMMUNITIES, CENTRALITY =========
import networkx as nx
import pandas as pd

def opening_projection_from_transitions(transitions_df: pd.DataFrame, min_shared: int = 50, top_per_node: int = 10) -> nx.Graph:
    """
    Build an opening–opening weighted graph by projecting the prefix–opening bipartite graph.
    Weight(i,j) = sum over prefixes of ( count_i(prefix) * count_j(prefix) )
    Keep only edges with weight >= min_shared and (optionally) top 'top_per_node' per node.
    """
    df = transitions_df.copy()
    df.columns = [c.strip() for c in df.columns]
    if not {"prefix","Opening","count"} <= set(df.columns):
        raise ValueError("expected columns ['prefix','Opening','count']")
    # pivot: rows=prefix, cols=Opening
    M = df.pivot_table(index="prefix", columns="Opening", values="count", fill_value=0)
    # co-occurrence matrix across openings
    C = M.T.dot(M)  # Opening x Opening
    # build graph
    G = nx.Graph()
    opens = list(C.index)
    for o in opens:
        G.add_node(o)
    # add edges with weights
    for i, oi in enumerate(opens):
        row = C.iloc[i]
        # keep top edges from this node
        top = row.sort_values(ascending=False)
        # drop self
        top = top[top.index != oi]
        if top_per_node:
            top = top.head(top_per_node)
        for oj, w in top.items():
            if w >= min_shared:
                G.add_edge(oi, oj, weight=float(w))
    return G

def communities_and_centrality(G: nx.Graph):
    """
    Greedy modularity communities + centralities (weighted degree, PageRank).
    """
    if G.number_of_nodes() == 0:
        return [], {}, {}, {}
    # communities
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        comms = [set(G.nodes())]
    # membership dict
    community_of = {}
    for cid, s in enumerate(comms):
        for n in s:
            community_of[n] = cid
    # centrality
    strength = {n: sum(d.get("weight",1.0) for _,_,d in G.edges(n, data=True)) for n in G.nodes()}
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}
    return comms, community_of, strength, pr
# ========= /OPENING PROJECTION, COMMUNITIES, CENTRALITY =========
