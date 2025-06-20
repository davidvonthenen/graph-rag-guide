## Executive Summary

Retrieval-Augmented Generation (RAG) is the go-to pattern for grounding large-language-model answers in real data. Most teams reach for a vector database first—then wonder why the model still hallucinates and can't explain itself. A graph-based RAG agent (long-term memory on durable storage and short-term memory on high-speed cache), solves those pain points in one shot.

**Why graphs beat pure vectors**

- **Fewer hallucinations.** Queries resolve to concrete nodes and relationships instead of fuzzy embedding neighbours, so the model can't invent facts that aren't in the graph.
- **Transparent explainability.** Every answer is traceable through a Cypher path that humans can read and auditors can log. No more "the embedding said so."
- **Governance baked in.** The schema exposes bias hot-spots, records provenance, and satisfies regulatory audits with clear lineage.

**Dual-memory design**

| Layer                         | Where it lives                                        | What it stores                               | Why it matters                                                |
| ----------------------------- | ----------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| **Long-term memory**          | Dedicated Graph instance on traditional disk          | Curated, non-volatile knowledge graph        | Authoritative source; supports compliance and audit trails    |
| **Short-term memory / cache** | Separate Graph on NVMe, RAM disk, or NetApp FlexCache | The working set for the current conversation | Millisecond look-ups; keeps latency low even under heavy load |

Promotion is metadata-only: remove an `expiration` timestamp and a short-term fact becomes long-term. The same mechanism enables reinforcement learning—validated insights flow naturally into the permanent graph.

**Business impact**

- **Transparency & accountability** - every answer can be walked back to its origin.
- **Fairness & bias mitigation** - explicit edges make biased correlations easy to spot and fix.
- **Risk reduction** - grounding answers in a strict knowledge graph slashes the odds of rogue outputs.
- **Performance at scale** - FlexCache-powered short-term memory keeps hot data near compute while SnapMirror protects it.

The sections that follow walk through ingesting data (`ingest.py`), caching queries (`cache_cypher_query.py`), and promoting knowledge (`short_to_long_transfer.py`). Together they form an open-source reference implementation any team can drop into a GitHub repo and run today.

Welcome to graph-based RAG: faster, clearer, and finally governable.

## 2. Ingesting Your Data Into Long-Term Memory

### Why we start with clean knowledge

Long-term memory is the system's "source of truth." Anything that lands here must be authoritative, fully traceable, and ready for governance audits. The ingestion pipeline does four things in one pass:

1. **Parses raw content** (articles, manuals, support tickets—pick your poison).
2. **Slices it into paragraphs** so retrieval can be as granular as a single sentence when needed.
3. **Extracts named entities** with a lightweight NER model (spaCy in the reference code).
4. **Persists entities, paragraphs, and documents** as first-class graph objects linked by explicit relationships.

Do this well once, and every downstream RAG prompt inherits the same audit-friendly provenance.

### Pipeline at a Glance

The reference `ingest.py` file spells it out clearly: documents are split into paragraphs, entities are lower-cased for stable matching, and everything is stitched together with `PART_OF` and `MENTIONS` relationships .

```
Document  <- PART_OF -  Paragraph
Entity    - MENTIONS -> Paragraph | Document
```

Each object carries a UUID plus helpful metadata (category, index, expiration flag). Because relationships store the `expiration` property too, you can later "age out" bad data without touching the nodes.

### Step-by-step Walkthrough

| Stage                           | What happens                                                                                                         | Code cue |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------- |
| **1. Load a spaCy model**       | `nlp = spacy.load("en_core_web_sm")` boots a fast NER pipeline.                                                      |          |
| **2. Walk the dataset**         | For every text file, the first line is the title, the rest is body.                                                  |          |
| **3. Create a `Document` node** | One call to `create_document` with a fresh UUID stores title, body, category.                                        |          |
| **4. Split into paragraphs**    | Blank lines mark paragraph breaks; each paragraph gets its own node plus an index for ordering.                      |          |
| **5. Extract entities**         | The NER loop filters labels via an environment variable so you can ingest only what matters (e.g., `PERSON`, `ORG`). |          |
| **6. Deduplicate entities**     | `MERGE (e:Entity {name:$name})` guarantees one node per unique lower-cased name—no duplicates, ever.                 |          |
| **7. Link everything**          | Two `MENTIONS` edges connect the entity to both paragraph and document for multi-hop reasoning.                      |          |
| **8. Idempotent reruns**        | Rerunning the script won't create duplicates thanks to `MERGE`; useful for batch or streaming loads.                 |          |

### Operational Knobs You Control

| Environment variable            | Purpose                                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `DATA_DIR`                      | Where the raw files live. Point it at a shared volume or object store mount.                                      |
| `NER_TYPES`                     | Comma-separated list of labels to keep. Empty means "accept all."                                                 |
| `*_URI`, `*_USER`, `*_PASSWORD` | Connection details for your chosen graph database. Swap drivers as needed—nothing in the logic is brand-specific. |
| `ALLOWED_LABELS` set            | Acts as a coarse throttle so your graph doesn't bloat with every `DATE` or `TIME` entity the NER model finds.     |

Because the script is pure client code, migrating to another graph engine is as easy as swapping out the driver in the `with GraphDatabase.driver(...)` block. Everything above the driver layer stays the same.

### Implementation Considerations

The reference implementation is a model for a recommendation. It's highly recommended and required to adapt your Ontology (ie, Graph structure, keywords, etc) based on your problem domain.

The use of Named-Entity-Recognition, more specifically the use of `spaCy`, is used for demonstration purposes only and to move the discussion along. Ideally, how you anchor your keywords to the associations on source data will be determined by your problem domain. In other words, it is probably more advantageous to implementation a Named Entity Recognition model based on the keywords you want to identity and act upon. This could be an off-the-shelf implementation or this could be a model that you train based on the data in your dataset.

## 3. Promotion of Long-Term Memory into Short-Term Cache

### Why bother with promotion?

Queries rarely roam the entire knowledge graph. Most conversations cling to a handful of entities for a while—think of them as today's "hot topics." Shuttling those entities (and the paragraphs that mention them) into a high-speed cache keeps latency down and keeps your GPU batch queue humming instead of twiddling its thumbs.

### How the promotion cycle works

1. **Detect entities in the user's question.**
   The demo loop runs a spaCy pass over every prompt and collects `(name, label)` pairs. If an entity hasn't been seen in the current session, it becomes a promotion candidate .

2. **Fetch the full context from long-term memory.**
   A Cypher statement (shown in the reference code as `PROMOTION_QUERY`) grabs the entity, its parent document, and every paragraph that references it in a single round-trip .

3. **Copy the sub-graph into short-term memory.**
   The function `promote_entity` merges nodes and relationships into the cache. It also writes an expiration timestamp (`TTL_MS`, default = 1 hour) so stale data quietly disappears tomorrow morning  .

4. **Optionally include the whole document.**
   Flip `PROMOTE_DOCUMENT_NODES=0` if you only need paragraphs; leave it on to keep the top-level document for full-text answers .

5. **Serve the user from cache first.**
   After promotion, subsequent queries pull paragraphs straight from short-term memory via `FETCH_PARAS_QUERY`, ordered by "how many entities match" and paragraph index .

### Key design decisions

| Design choice                            | Rationale                                                                                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Expiration on the edge, not the node** | Relationships get an `expiration` timestamp. When it lapses, the edge becomes invisible without deleting the node—perfect for governance audits.                    |
| **Idempotent merges**                    | Every `MERGE` uses a stable UUID, so re-running promotion can't create duplicates .                                                                                 |
| **Session-level "seen" cache**           | A tiny in-memory set blocks redundant promotions inside the same chat. CPU-cheap and embarrassingly effective .                                                     |
| **Storage-agnostic layers**              | The promotion logic is plain Cypher plus a driver call. Swap the driver and connection strings, and you're on a different graph engine—no code gymnastics required. |

### Operational knobs you can tweak

| Env var / constant         | What it controls                                                              |
| -------------------------- | ----------------------------------------------------------------------------- |
| `TTL_MS`                   | Lifetime of cached facts (default: one day).                                  |
| `PROMOTE_DOCUMENT_NODES`   | Promote whole documents (`1`) or paragraphs only (`0`).                       |
| `INTERESTING_ENTITY_TYPES` | Filter which entity labels trigger promotion (PERSON, ORG, etc.).             |
| High-speed backing store   | NVMe, RAM disk, NetApp **FlexCache**—pick your weapon for micro-second reads. |

### Tips from the field

- **Keep TTL realistic.** A 24-hour cache is gold for human chat; an hour is plenty for server monitoring alerts.
- **Batch promotion when latency matters.** Detect all new entities first, then promote in one transaction to avoid chatty round-trips.
- **Monitor cache hit rate.** If hits fall below .70 %, widen `TTL_MS` or revisit entity filtering.

### The payoff

Promotion turns your short-term memory into a turbo-charged working set: most answers come straight from fast storage, while long-term memory idles happily on cheaper disks. The result? Faster responses, lower I/O thrash, and an audit trail you can actually read.

## 4. Provides Reinforcement Learning and Data Promotion

### Why a feedback loop matters

If short-term memory is the "working set," reinforcement learning decides what deserves a permanent spot in long-term memory. We watch how often a fact is reused, validated, or corrected. When its score crosses a threshold, we promote it. Anything below the line expires quietly.

### Enter MCP: today's missing context

Large language models freeze at their last training cut-off. **Managed Content Pipelines (MCP)**—news feeds, regulatory bulletins, live sensor logs, you name it—fill that gap. They pour *new* events into short-term memory minutes after they happen. From there:

1. **Immediate recall.** Users can question the fresh data right away.
2. **Scoring starts.** Every cache hit or explicit "yes, that's correct" bumps the item's `confidence_score`.
3. **Automatic promotion.** Once `confidence_score ≥ PROMOTE_THRESHOLD`, the `short_to_long_transfer.py` job copies the sub-graph into long-term memory and marks it `promoted:true`.

### Promotion workflow

| Stage                              | Trigger                                       | Action                                                                                              |
| ---------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **1. Ingest MCP event**            | New item arrives                              | Create nodes/edges in short-term memory with `confidence_score=1`, `expiration=TTL`.                |
| **2. Increment score**             | Paragraph or entity is retrieved in an answer | `confidence_score += HIT_WEIGHT`.                                                                   |
| **3. Human validation (optional)** | SME flags item as correct                     | `confidence_score += VALIDATION_WEIGHT`.                                                            |
| **4. Cross threshold**             | `confidence_score ≥ PROMOTE_THRESHOLD`        | `promotion_job` copies the sub-graph and relationships into long-term memory; clears `expiration`.  |
| **5. Audit trail**                 | After copy                                    | Write a promotion log record with timestamp, user/session id, and source (e.g., "MCP-regulations"). |

All scoring fields live on the **relationship**, not the node, so you can retract a single claim without touching the underlying entity.

### Code cues in the reference transfer script

```python
# promotion_job.py  (excerpt)
MATCH (d:Document {promoted:false})<-[:PART_OF]-(p:Paragraph)
WHERE p.confidence_score >= $PROMOTE_THRESHOLD
WITH d, COLLECT(p) AS paras
CALL {
  WITH d, paras
  // create-or-merge in long-term store
  UNWIND paras AS para
  MERGE (ld:Document {uuid:d.uuid})          // idempotent
  MERGE (lp:Paragraph {uuid:para.uuid})
  MERGE (lp)-[:PART_OF]->(ld)
  FOREACH (rel IN relationships(para) |
           MERGE (e:Entity {uuid:rel.other_uuid})
           MERGE (e)-[:MENTIONS]->(lp))
}
SET d.promoted = true
```

*No database names in sight—swap the driver and it runs on any graph engine.*

### Tunables you should expose

| Variable            | Default | Why you might change it                                                   |
| ------------------- | ------- | ------------------------------------------------------------------------- |
| `HIT_WEIGHT`        | 1       | Raise it for longer chat sessions, lower it for high-volume API calls.    |
| `VALIDATION_WEIGHT` | 10      | Higher when humans are the gatekeepers; lower for purely automated flows. |
| `PROMOTE_THRESHOLD` | 25      | Controls strictness—set high for regulated domains.                       |
| `TTL`               | 1 h     | Shorten for bursty data (social media), extend for weekly digests.        |

### Governance and safety wins

- **Traceable updates** - Every promotion record shows *who* and *why* a fact became permanent.
- **Bias checks** - Because MCP sources are tagged, you can audit whether one feed dominates the graph.
- **Rollback-ready** - If a promoted fact proves wrong, flip `promoted:false`, run the demotion script, and all edges evaporate without a delete.

### Field notes

*Start narrow.* Begin with one MCP feed and watch the hit rate. You'll quickly see which entity types (e.g., `ORG`, `LAW`, `VULNERABILITY`) drive the most promotions. Then tune `HIT_WEIGHT` and `PROMOTE_THRESHOLD` instead of rewriting logic.

With reinforcement learning and MCP in play, your graph stays fresh, relevant, and governed—all without nightly retraining runs. Next, we'll wrap up with final thoughts and how you can contribute.

## 5. Implementation Guide**

For a reference implementation, please check out the following: [community_implementation/README.md](./community_implementation/README.md)

## 6. Conclusion

Graph-based RAG gives you more than quick answers—it gives you answers you can **trust**.

- **Transparency and explainability** come baked-in: every node and relationship is a breadcrumb you can trace end-to-end.
- **Fairness, accountability, and risk control** improve because biases and data lineage sit in plain sight.
- **Compliance** stops feeling like red tape and starts feeling like a query away.

Pair that with the dual-memory design—lightning-fast short-term cache plus durable long-term memory—and you get a system that's both **responsive today** and **auditable tomorrow**.

### Your next steps

1. **Clone the repo.** All reference code and docs live in the public GitHub project; fork it, run it, break it, make it your own.
2. **Swap in your graph engine.** The Cypher examples are driver-agnostic, so plug in whatever backend fits your stack.
3. **Augment the LLM using MCP for a Data Feed.** Let fresh facts flow, score them, and watch reinforcement learning push the best ones upstream.
4. **Share what you learn.** Open an issue, file a PR, or write a case study—this project gets better when the community kicks the tires.

Graph-based RAG isn't a speculative idea; it's running code with clear governance wins. Dig in, experiment, and help raise the bar for responsible AI.
