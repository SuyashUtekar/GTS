Implementation
Step 5.5 — Neighborhood Expansion

# ═══════════════════════════════════════════════════════════════
# STEP 5.5: TOP-K NEIGHBORHOOD EXPANSION
# ═══════════════════════════════════════════════════════════════

def find_neighborhood(
    selected_nodes,
    embeddings,
    sentences,
    edge_weight_matrix,
    n_neighbors     = 3,
    source          = 'edge_weight',  # 'edge_weight' or 'embedding'
    exclude_selected= True,
) -> dict:
    """
    For each selected sentence, find its top-N closest
    sentences from the FULL original document.

    Parameters:
      selected_nodes   : list of selected sentence ids
      embeddings       : SBERT embeddings (n × dim)
      sentences        : article['sentences']
      edge_weight_matrix: precomputed edge weights (n × n)
      n_neighbors      : how many neighbors per sentence
      source           : use edge_weight or raw embedding sim
      exclude_selected : don't add already-selected sentences

    Returns:
      neighborhood dict:
      {
        selected_node: {
          "neighbors"   : [node_id, ...],
          "similarities": [score, ...],
          "texts"       : [text, ...]
        }
      }
    """
    selected_set = set(selected_nodes)
    n_total      = len(sentences)
    neighborhood = {}

    print(f"Finding top-{n_neighbors} neighbors "
          f"for {len(selected_nodes)} selected sentences...")
    print(f"  Source          : {source}")
    print(f"  Exclude selected: {exclude_selected}")
    print("-" * 55)

    for node in selected_nodes:
        scores = []

        for j in range(n_total):
            if j == node:
                continue
            if exclude_selected and j in selected_set:
                continue

            if source == 'edge_weight':
                # use precomputed combined edge weight
                sim = float(edge_weight_matrix[node][j])
            else:
                # use raw cosine similarity from embeddings
                sim = float(
                    cosine_similarity(
                        embeddings[node].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
                )

            scores.append((sim, j))

        # top-N by similarity
        top_n = sorted(scores, key=lambda x: x[0],
                        reverse=True)[:n_neighbors]

        neighbors   = [j   for _, j in top_n]
        sims        = [s   for s, _ in top_n]
        texts       = [sentences[j]['text'] for j in neighbors]

        neighborhood[node] = {
            "neighbors"    : neighbors,
            "similarities" : sims,
            "texts"        : texts,
        }

        # print top neighbors for inspection
        node_txt = sentences[node]['text'][:60]
        print(f"\n  S{node:03d}: \"{node_txt}...\"")
        for rank, (j, s) in enumerate(zip(neighbors, sims), 1):
            n_txt = sentences[j]['text'][:60]
            print(f"    [{rank}] S{j:03d} "
                  f"(sim={s:.4f}): \"{n_txt}...\"")

    return neighborhood


# ── Run neighborhood expansion ────────────────────────────────
print("=" * 65)
print("STEP 5.5: NEIGHBORHOOD EXPANSION")
print("=" * 65)

neighborhood = find_neighborhood(
    selected_nodes     = final_selection_ordered,
    embeddings         = article['embeddings'],
    sentences          = article['sentences'],
    edge_weight_matrix = article['edge_weight_matrix'],
    n_neighbors        = 3,
    source             = 'edge_weight',   # uses α·S + β·T + γ·E
    exclude_selected   = True,
)

article['neighborhood'] = neighborhood





def build_context_windows(
    final_selection_ordered,
    neighborhood,
    sentences,
    partition,
    topic_importance,
    window_order = 'context_first',  # or 'sentence_first'
) -> dict:
    """
    For each selected sentence, build a context window:
      [neighbor_before, selected_sentence, neighbor_after]

    OR simply:
      [all neighbors + selected sentence]

    Parameters:
      window_order:
        'context_first'   → neighbors before sentence
        'sentence_first'  → sentence first then neighbors
        'interleaved'     → rank-ordered by document position

    Returns:
      context_windows dict:
      {
        node: {
          "window_nodes": [n1, n2, selected, n3],
          "window_text" : "full context paragraph",
          "community"   : comm_id,
          "importance"  : topic_importance score
        }
      }
    """
    context_windows = {}

    for node in final_selection_ordered:
        neighbors = neighborhood[node]['neighbors']
        comm_id   = partition.get(node, -1)
        ti        = topic_importance.get(comm_id, 0.0)

        # combine selected + neighbors
        all_nodes = [node] + neighbors

        if window_order == 'interleaved':
            # sort by original document position
            # most natural reading order
            all_nodes = sorted(set(all_nodes))

        elif window_order == 'context_first':
            # neighbors that come BEFORE selected node first
            before = sorted([n for n in neighbors if n < node])
            after  = sorted([n for n in neighbors if n > node])
            all_nodes = before + [node] + after

        elif window_order == 'sentence_first':
            # selected sentence first for emphasis
            all_nodes = [node] + sorted(neighbors)

        # build window text
        window_texts = [
            sentences[n]['text'].strip()
            for n in all_nodes
        ]
        window_text = " ".join(window_texts)

        context_windows[node] = {
            "window_nodes": all_nodes,
            "window_text" : window_text,
            "community"   : comm_id,
            "importance"  : ti,
            "n_words"     : len(window_text.split()),
        }

    return context_windows


# ── Build context windows ─────────────────────────────────────
context_windows = build_context_windows(
    final_selection_ordered = final_selection_ordered,
    neighborhood            = neighborhood,
    sentences               = article['sentences'],
    partition               = partition,
    topic_importance        = topic_importance,
    window_order            = 'context_first',
)

article['context_windows'] = context_windows

# ── Stats ─────────────────────────────────────────────────────
total_window_words = sum(
    cw['n_words'] for cw in context_windows.values()
)
print(f"\n✓ Context windows built")
print(f"  Selected sentences  : {len(final_selection_ordered)}")
print(f"  Neighbors per sent  : 3")
print(f"  Total window nodes  : "
      f"{sum(len(cw['window_nodes']) for cw in context_windows.values())}")
print(f"  Total window words  : {total_window_words}")
print(f"\nSample context window:")
print("-" * 55)
sample_node = final_selection_ordered[0]
cw          = context_windows[sample_node]
print(f"  Selected: S{sample_node:03d}")
print(f"  Window nodes: {cw['window_nodes']}")
print(f"  Window text: \"{cw['window_text'][:200]}...\"")





def build_neighborhood_summary(
    final_selection_ordered,
    context_windows,
    partition,
    topic_importance,
    bridge_nodes,
) -> str:
    """
    Build final extractive summary using
    context windows instead of bare sentences.

    Deduplicates: if a neighbor node appears in
    multiple windows, it is included only once
    at its first occurrence.
    """
    seen_nodes     = set()
    summary_parts  = []
    prev_community = None
    transition_idx = 0

    transitions = [
        "Furthermore,",
        "Additionally,",
        "In this regard,",
        "With respect to this,",
        "Moreover,",
    ]

    for node in final_selection_ordered:
        cw        = context_windows[node]
        curr_comm = partition.get(node, -1)
        is_bridge = node in bridge_nodes

        # community transition
        if (prev_community is not None
                and curr_comm != prev_community
                and not is_bridge):
            transition = transitions[
                transition_idx % len(transitions)
            ]
            transition_idx += 1
        else:
            transition = None

        # add window sentences in order
        # skipping already seen nodes
        window_texts = []
        for w_node in cw['window_nodes']:
            if w_node not in seen_nodes:
                window_texts.append(
                    article['sentences'][w_node]['text'].strip()
                )
                seen_nodes.add(w_node)

        if not window_texts:
            continue

        # apply transition to first sentence of window
        if transition:
            starters = (
                "however", "additionally", "furthermore",
                "moreover", "also", "but", "yet", "in",
                "on", "while", "with", "regarding",
                "notwithstanding", "subject to"
            )
            if not window_texts[0].lower().startswith(starters):
                window_texts[0] = f"{transition} {window_texts[0]}"

        summary_parts.extend(window_texts)
        prev_community = curr_comm

    return " ".join(summary_parts)


# ── Build neighborhood-expanded summary ───────────────────────
neighborhood_summary = build_neighborhood_summary(
    final_selection_ordered = final_selection_ordered,
    context_windows         = context_windows,
    partition               = partition,
    topic_importance        = topic_importance,
    bridge_nodes            = bridge_nodes,
)

article['neighborhood_summary'] = neighborhood_summary

print("\n" + "=" * 65)
print("NEIGHBORHOOD-EXPANDED EXTRACTIVE SUMMARY:")
print("=" * 65)
print(neighborhood_summary)
print(f"\nWords     : {len(neighborhood_summary.split())}")
print(f"Sentences : {len(seen_nodes)}")



# ── Compare bare vs neighborhood summary ──────────────────────
print("\n" + "=" * 65)
print("COMPARISON: BARE vs NEIGHBORHOOD SUMMARY")
print("=" * 65)
print(f"\n  {'Metric':<25} {'Bare Extractive':>16} "
      f"{'Neighborhood':>14}")
print(f"  {'─'*57}")
print(f"  {'Word count':<25} "
      f"{len(extractive_summary.split()):>16} "
      f"{len(neighborhood_summary.split()):>14}")
print(f"  {'Sentence count':<25} "
      f"{len(final_selection_ordered):>16} "
      f"{len(seen_nodes):>14}")

# ROUGE comparison
rouge_bare  = scorer.score(
    article['reference'], extractive_summary
)
rouge_nbhd  = scorer.score(
    article['reference'], neighborhood_summary
)

for metric in ['rouge1', 'rouge2', 'rougeL']:
    b = rouge_bare[metric].fmeasure
    n = rouge_nbhd[metric].fmeasure
    w = 'Bare  ' if b >= n else 'Nbhd  '
    print(f"  {metric:<25} {b:>16.4f} {n:>14.4f}  ← {w}")

# now use neighborhood_summary as input to BART
# instead of bare extractive_summary
print(f"\n✓ neighborhood_summary ready for Step 7 BART input")



# ── Feed into Step 7 BART ─────────────────────────────────────
# ONE LINE CHANGE in Step 7:

# BEFORE (bare extractive):
# abstractive_result = abstractive_single_pass(
#     extractive_text = extractive_summary, ← change this
#     ...
# )

# AFTER (neighborhood expanded):
abstractive_result = abstractive_single_pass(
    extractive_text = neighborhood_summary,  # ← neighborhood
    summarizer      = summarizer,
    tokenizer       = abs_tokenizer,
    min_length      = 80,
    max_length      = 220,                   # slightly larger
)                                            # more input = more output

# OR for chunked strategy:
abstractive_result = abstractive_chunked_pass(
    final_selection_ordered = final_selection_ordered,
    sentences               = article['sentences'],
    partition               = partition,
    topic_importance        = topic_importance,
    summarizer              = summarizer,
    tokenizer               = abs_tokenizer,
    # neighborhood passed here ↓
    context_windows         = context_windows,
    max_chunk_tokens        = 900,
    min_length              = 40,
    max_length              = 100,
)



What Changes in Chunked Strategy
If your token count exceeds 1024 after adding neighbors, update abstractive_chunked_pass to use context windows per community:

# inside abstractive_chunked_pass()
# replace:
#   chunk_text = " ".join(comm_sentences[comm_id])

# with:
for comm_id in sorted_comms:
    # get context windows for this community's sentences
    comm_nodes = [
        n for n in final_selection_ordered
        if partition.get(n) == comm_id
    ]

    # build chunk from context windows
    chunk_parts = []
    seen        = set()
    for node in comm_nodes:
        if node in context_windows:
            for w_node in context_windows[node]['window_nodes']:
                if w_node not in seen:
                    chunk_parts.append(
                        sentences[w_node]['text']
                    )
                    seen.add(w_node)

    chunk_text = " ".join(chunk_parts)
    # rest unchanged...





