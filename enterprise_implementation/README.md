# Enterprise Implementation for Graph RAG with Cache and Reinforcement Learning

This README.md provides an **enterprise-oriented reference implementation** for a Graph-based Retrieval-Augmented Generation (RAG) system. It builds upon the [open-source community version](../community_implementation/README.md) by adding production-grade features: **Kafka Connect** pipelines for automated data sync, a **dual-memory model** with high-speed cache ([FlexCache](https://www.netapp.com/data-storage/what-is-flex-cache/)) and durable storage (SnapMirror replication), and a **reinforcement learning loop** that promotes facts via Kafka-triggered events.

Graph-based RAG explicitly models knowledge as nodes and relationships, mitigating issues like hallucinations and opaque reasoning that plague vector databases. The integration of Apache Kafka’s change-data-capture (CDC) ensures seamless synchronization between short-term and long-term memory stores, while technologies such as NetApp [FlexCache](https://www.netapp.com/data-storage/what-is-flex-cache/) and [SnapMirror](https://docs.netapp.com/us-en/ontap/concepts/snapmirror-disaster-recovery-data-transfer-concept.html) provide low-latency access and multi-site resilience. The result is a faster, more transparent, and governable RAG system ready for enterprise workloads.

## Prerequisites

- A Linux or Mac-based development machine with sufficient memory to run two Neo4j instances and an LLM (≈8B parameters).
  - *Windows users:* use a Linux VM or cloud instance if possible.
- **Python 3.10+** installed (with [venv](https://docs.python.org/3/library/venv.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) for isolation).
- **Docker** installed (for running Neo4j, Kafka, etc.).
- **Apache Kafka** with Kafka Connect available (e.g. via Confluent Platform Docker images) for streaming data between Neo4j instances.
- (Optional) Access to a **NetApp ONTAP** environment if you plan to use [FlexCache](https://www.netapp.com/data-storage/what-is-flex-cache/) for short-term storage and [SnapMirror](https://docs.netapp.com/us-en/ontap/concepts/snapmirror-disaster-recovery-data-transfer-concept.html) for replication, though the example can be run without it.
- Basic familiarity with shell and Docker commands.

**Docker images to pre-pull:**

- `neo4j:5.26-enterprise` (Neo4j database for both long-term and short-term instances)
- Kafka and Zookeeper images (e.g. `confluentinc/cp-kafka:7.4.0` and `confluentinc/cp-zookeeper:7.4.0`, or another Kafka distribution)
- Kafka Connect image (e.g. `confluentinc/cp-kafka-connect:7.4.0` with the Neo4j Connector plugin installed)

> **Note:** Ensure the Neo4j Kafka Connector plugin (Source and Sink) is available to your Kafka Connect worker. You can obtain the connector JAR from Neo4j’s official distribution or Confluent Hub, and configure the Connect container to load it.

### LLM to pre-download:

For example, you can use the following 7-8B parameter models that run locally (CPU-friendly via [llama.cpp](https://github.com/ggerganov/llama.cpp)):

- Intel's **neural-chat-7B-v3-3-GGUF** - *(tested model)* available on HuggingFace
- OR [bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) - an 8B instruction-tuned Llama variant.
- OR use an API/manager like [Ollama](https://ollama.com/), e.g. `ollama pull llama3:8b` for an 8B Llama model.

## Setting Up the Environment

To get started, we will set up two Neo4j graph database instances (one for **long-term memory** and one for **short-term cache**), an Apache Kafka environment (broker, Zookeeper, and Kafka Connect with Neo4j connectors), and a local LLM for query processing. The instructions below assume a single-machine Docker-based setup for demonstration, but the components can be deployed to your infrastructure of choice.

### Launch Neo4j and Kafka with Docker

We will now launch two Neo4j instances (one for **long-term memory** and one for **short-term memory**), along with Apache Kafka and Kafka Connect. For convenience, a Docker CLI commands configurations are provided (bringing up Neo4j, Zookeeper, Kafka broker, and Connect). You can use Docker CLI commands to start everything, or run the services manually as shown below. All services should join a common Docker network so they can communicate (the compose file handles this automatically).

Below are example Docker CLI commands for the Neo4j instances. They configure each Neo4j container with the necessary ports, credentials, and plugins (APOC is enabled for import/export). The short-term instance’s data directory is pointed at the high-speed storage (`$HOME/neo4j-short`). Both instances are given distinct ports and passwords:

```bash
# Long-Term Memory Instance (neo4j-long-term)
# Admin Panel: http://127.0.0.1:7475
# Note the password: neo4j/neo4jneo4j1
docker run -d \
    --name neo4j-long-term \
    -p 7475:7474 -p 7688:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j1 \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_server_http_advertised__address="localhost:7475" \
    -e NEO4J_server_bolt_advertised__address="localhost:7688" \
    -v "$HOME/neo4j-long/data":/data \
    -v "$HOME/neo4j-long/import":/import \
    -v "$HOME/neo4j-long/plugins":/plugins \
    neo4j:5.26

# Short-Term Memory/Cache Instance (neo4j-short-term)
# Admin Panel: http://127.0.0.1:7476
# Note the password: neo4j/neo4jneo4j2
docker run -d \
    --name neo4j-short-term \
    -p 7476:7474 -p 7689:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j2 \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_server_http_advertised__address="localhost:7476" \
    -e NEO4J_server_bolt_advertised__address="localhost:7689" \
    -v "$HOME/neo4j-short/data":/data \
    -v "$HOME/neo4j-short/import":/import \
    -v "$HOME/neo4j-short/plugins":/plugins \
    neo4j:5.26
```

> **Note:** If running manually, first create a Docker network (e.g. `docker network create graph-rag-net`) and add `--network graph-rag-net` to each `docker run`. Ensure the Kafka and Connect containers use the same network, so that the connectors can reach the Neo4j instances by name.

At this point, also launch your Kafka services (Zookeeper, Kafka broker, and the Kafka Connect worker). If using Docker CLI commands, these will start automatically. If launching manually, run the Zookeeper container, then the Kafka broker (linking to Zookeeper and configuring `KAFKA_ADVERTISED_LISTENERS`), and finally the Kafka Connect worker (ensuring it can reach both Kafka and Neo4j). Refer to Kafka’s documentation for the specific commands or use the provided compose file.

Once all containers are up, the Neo4j databases will be empty initially. You can verify the Neo4j instances are running by opening the Neo4j Browser web UI at **[http://localhost:7475](http://localhost:7475)** (long-term) and **[http://localhost:7476](http://localhost:7476)** (short-term). Log in with username `neo4j` and the password you set (as per the above commands). If you see the Neo4j browser interface, the databases are up and ready.

> **IMPORTANT:** The password for the **Long-term Memory** Neo4j instance is `neo4jneo4j1` and the password for the **Short-term Memory** instance is `neo4jneo4j2` (as configured above).

### Configuring Kafka Connectors

Next, set up the Kafka Connect **Source** and **Sink** connectors for Neo4j. These connectors will automate the movement of graph data between the two Neo4j instances in both directions (long-term → short-term for caching, and short-term → long-term for promotions). If you are using the provided Docker commands, the Kafka Connect service should already have the Neo4j connector plugin available. You can configure the connectors by using Kafka Connect’s REST API (POST the connector configurations as JSON) or through the Confluent Control Center if available.

**Neo4j Source Connector (Long-term → Short-term):** This connector monitors the long-term Neo4j database for relevant changes or query results and publishes them to Kafka topics. For on-demand caching, we configure the source to execute a parameterized Cypher query for entities of interest and stream the results. For example, a source connector configuration might look like this (connecting to the long-term DB and outputting to topics with a `promoted` prefix):

```json
{
  "name": "ShortTermNeo4jSource",
  "connector.class": "Neo4jSourceConnector",
  "neo4j.server.uri": "bolt://neo4j-long-term:7688",
  "topic.prefix": "promoted",
  "neo4j.streaming.from": "now"
}
```

This tells Kafka Connect to start streaming from the **neo4j-long-term** instance (using the Bolt port 7688) and write any captured events to topics named `promoted.*`. In practice, this source would be set to capture either updates or query results for the subgraph that needs to be cached.

**Neo4j Sink Connector (Short-term → Long-term):** This connector consumes Kafka topics and writes data into a Neo4j database. In our setup, we use a sink on the **long-term** database to receive validated facts from short-term memory. We embed Cypher templates in the sink configuration to define *how* the incoming events should be merged into Neo4j. For example, the sink config below listens on topics for promoted nodes and relationships and upserts them into the long-term store:

```json
{
  "name": "LongTermNeo4jSink",
  "connector.class": "Neo4jSinkConnector",
  "topics": "validated.nodes,validated.rels",
  "neo4j.topic.cypher.validated.nodes": "MERGE (n:Entity {uuid:event.id}) SET n += event.properties REMOVE n.expiration SET n.promoted=true",
  "neo4j.topic.cypher.validated.rels": "MERGE (a {uuid:event.start.id}) MERGE (b {uuid:event.end.id}) MERGE (a)-[r:MENTIONS]->(b) SET r += event.properties REMOVE r.expiration SET r.promoted=true"
}
```

Here, the sink connector is configured to merge incoming **node** events and **edge/relationship** events into the long-term Neo4j. The Cypher templates ensure that if an Entity or relationship already exists it will be updated (`MERGE ... SET += event.properties`), and critically, they **remove the `expiration` property** and set a flag `promoted=true` on the data. Removing the `expiration` is the *magic moment* when a fact graduates from short-term cache to long-term memory (i.e., it will no longer be evicted). The `promoted=true` property and any transaction metadata serve as an audit trail of the promotion event.

> **IMPORTNT:** (Similarly, you would configure a mirror-image **sink connector** on the short-term Neo4j side to ingest data from `promoted.*` topics for the caching flow, using Cypher that **adds** an `expiration` timestamp to incoming relationships. In our architecture, the **Source** on long-term and a corresponding **Sink** on short-term work together to handle cache population. Conversely, a **Source** on short-term and the above **Sink** on long-term handle permanent promotions when facts are validated.)*

With these connectors in place, the system achieves **exactly-once** delivery of graph updates via Kafka buffering, decoupling the two databases. Kafka will queue any changes if a database is down and replay them when it’s back up. This robust pipeline removes the need for custom scripting to move data and ensures consistency between the caches and the source of truth.

### Python Environment and Dependencies

With the databases and Kafka pipeline running, set up the Python environment for running the provided code. You should have Python 3.10+ available. It’s highly recommended to use a virtual environment (using [venv](https://docs.python.org/3/library/venv.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)) for isolation.

Install the required Python libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install necessary packages such as Neo4j Python driver, Py2Neo, spaCy, etc. After installing spaCy, download the small English model for named entity recognition (NER) if you haven't already:

```bash
python -m spacy download en_core_web_sm
```

### Set Up the Local LLM (using llama.cpp)

For the question-answering component, you’ll need a local LLM. In this guide, we use a 7B parameter model called **neural-chat-7B-v3-3-GGUF** (a quantized GGUF model) as it can run on CPU with llama.cpp and provides a good balance of performance and size. Using this known model ensures the setup works out-of-the-box.

However, you can substitute a different model if desired. For example, the [bartowski/Meta-Llama-3-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) model (8B parameters) or an Ollama-managed model could be used — the system is not tied to a specific LLM. The key is that the model is accessible via the Python code (in our case, through llama.cpp bindings) and can respond to prompts.

## Background on the Data

Our example knowledge source is a collection of BBC news articles in text format, provided in the archive **bbc-example.zip**. This zip file contains a subset of 300 BBC news articles (technology category) out of the 2,225 articles in the [BBC Full Text Document Classification dataset](https://bit.ly/4hBKNjp). After unzipping the archive, the directory structure will look like:

```
bbc/
├── tech/
    ├── 001.txt
    ├── 002.txt
    ├── 003.txt
    ├── 004.txt
    ├── 005.txt
    └── ...
```

Each file is a news article related to technology. In an enterprise scenario, this would be replaced with your domain-specific documents (internal wikis, knowledge bases, incident reports, etc.), but the BBC dataset serves as a clean example for this demonstration.

If you have not already, unzip the `bbc-example.zip` file (located in this repository) to prepare the data directory:

```bash
unzip bbc-example.zip
```

This will extract the articles into the `bbc/` folder as shown above.

## Example Workflows

The following example workflows demonstrate how the short-term cache and long-term graph work together in this enterprise setup to improve AI governance and performance. Each workflow shows a different aspect of the system in action, from basic query caching to reinforcement learning-driven promotions.

### 1. Simple Query Example

In this scenario, we will ingest data into the long-term graph, then execute a query to see how the system automatically pulls relevant data into the short-term cache (via Kafka connectors) to speed up subsequent queries.

1. **Perform the Ingest**: Run `python ingest.py`.
   **WARNING:** This will erase any existing nodes/edges in your Neo4j databases (both long-term and short-term) and then load the BBC dataset afresh. The ingest process will parse the documents, split them into paragraphs, perform NER to extract entities, and merge everything into the **long-term** Neo4j instance. Each relationship is annotated with metadata such as its source document, ingestion timestamp, and a schema version tag for governance. (In an enterprise setting, the ingest script wraps the entire batch in a single transaction and tags it with a batch ID for easier rollback, ensuring an authoritative, idempotent load of knowledge.)

2. **Perform a Query (Cache Miss → Cache Populate)**: Run `python nocache_cypher_query.py`.
   This script poses a sample question to the system. On the first run, the query will not find any relevant data in the short-term cache (a *cache miss*). The enterprise pipeline will then kick in: the **Kafka Source Connector** on the long-term database executes a predefined Cypher query to fetch the subgraph (entities and paragraphs) related to the query topics, and publishes the results to Kafka. The **Kafka Sink Connector** on the short-term database consumes these results and MERGEs the nodes and relationships into the short-term Neo4j (the cache), adding an `expiration` timestamp to each new relationship to enforce TTL (time-to-live). After this behind-the-scenes transfer, the relevant data now resides in the high-speed cache. The script will automatically re-run the query against the short-term store, which now yields results with sub-50ms latency (a cache hit). The user question is answered using only the cached subset of the graph, improving response time dramatically.

3. **Clean Up Short-Term Memory/Cache**: Run `python helper/wipe_short_memory.py`.
   This is a maintenance step for the demo - it clears the short-term Neo4j instance of all cached data. In a real deployment, cached entries would naturally expire and be evicted. Each cached relationship has an `expiration` property set (e.g. 1 hour) and a background job or scheduled process can periodically remove those expired edges (while leaving the nodes for audit purposes). For now, we manually wipe the short-term database to simulate that the cache has been cleared of stale data.

### 2. Reinforcement Learning Example

This workflow demonstrates how new facts can be introduced and evaluated in short-term memory, and how the system uses a reinforcement learning (RL) style feedback loop to decide which facts get promoted to long-term memory. We will simulate adding new information and then "teaching" the system through usage and validation, triggering an automatic promotion via Kafka when a fact proves its value.

1. **Perform the Ingest**: Run `python ingest.py` again.
   **WARNING:** This will erase all existing data in both Neo4j instances (long-term and short-term) to reset the state. We start with a clean long-term graph loaded with the base dataset (BBC articles), and an empty short-term cache.

2. **Enter New Facts into Short-Term Memory**: Run `python example/reinforcement_learning.py`.
   This example script will prompt you to introduce 5 new "facts" into the short-term memory (cache). These could be thought of as insights or data points not present in the original dataset. For example, facts #1 and #3 are about **OpenAI** (which are not in the BBC tech articles by default). You can choose to inject all or some of these facts; we recommend selecting **fact #1 and fact #3** for this demo. The script will add these as nodes/relationships in the short-term Neo4j (with an initial `confidence_score = 1` and an `expiration` time, since they start as unvalidated, short-term knowledge). After insertion, the script may run a few test queries or "challenge questions" about these facts to simulate user usage. Each time a new fact is used to answer a question (i.e., results in a successful retrieval), its `confidence_score` may be incremented (this simulates the system learning from usage: e.g., a cache hit might add **HIT_WEIGHT = 1** to the score). If a subject-matter expert or an external signal validates a fact, a larger increment (e.g., **VALIDATION_WEIGHT = 10**) would be applied. In this simplified script, usage of the facts through the queries serves as the reinforcement signal.

3. **Mark a Fact for Promotion to Long-Term**: Run `python example/promote_short_term_facts.py`.
   This script allows us to flag certain short-term facts as *validated* (worthy of long-term preservation) and others as *expired*. Following the recommendation, select the fact you added as #1 (e.g., the first OpenAI fact) to **promote**, and select fact #3 to let it expire. Under the hood, the script will update the graph: it flags the chosen fact (#1) by adding a special label (for example, `:Validated`) or property, and possibly sets fact #3’s state to indicate it should expire without promotion. At this moment, fact #1’s `confidence_score` has effectively reached the promotion threshold (e.g., **PROMOTE_THRESHOLD = 25** points, considering all the usage increments) and we mark it as such. No data is transferred yet, but the **presence of the `Validated` marker on fact #1** generates a CDC event in the short-term Neo4j instance.

4. **Automatic Transfer via Kafka (Promotion)**: *No manual step is required here.*
   The Neo4j **Source Connector** on the short-term cache has been watching for promotion events. It sees the fact labeled as `Validated` and streams this event to the Kafka topics (e.g., `validated.nodes`/`validated.rels`). The **Sink Connector** on the long-term Neo4j consumes the event and automatically **MERGEs the fact into the long-term Neo4j instance**, effectively **transferring the node/relationship from short-term to long-term storage**. As part of this merge, the connector’s Cypher template removes the `expiration` property and sets `promoted=true` on that fact, making it a permanent part of long-term memory. The transfer is logged by Kafka (offsets and transaction IDs), creating an audit trail for the promotion. At this point, the fact #1 has been **persisted** to long-term memory without any direct human intervention in the transfer process - the promotion was triggered entirely by the system’s RL feedback loop.

5. **Expire Short-Term Memory/Cache**: Run `python evict_expired_short_term.py`.
   This will remove any remaining facts in the short-term database that have passed their expiration or were marked to expire (in our case, the fact #3 that we did not promote). This step simulates the cache eviction process for facts that did not prove useful. In a production environment, such expiration could also be handled by Kafka Connect emitting a *tombstone* event or by a scheduled job that prunes expired data. (For instance, if a fact is not promoted and its time-to-live lapses, it will simply disappear from the cache, leaving only long-term facts stored permanently.)

6. **Verify Fact in Long-Term Memory**: Run `python example/cypher_query.py`.
   This will query the long-term Neo4j instance for the promoted fact (fact #1) to confirm that it now resides in long-term memory. The query should retrieve the fact from the long-term database, demonstrating that the fact was successfully carried over. In contrast, if you query for fact #3 (the one we let expire), it will not be found in either database (having been purged from short-term and never added to long-term).

7. **Reset the System (Optional)**: Finally, you can run `python helper/wipe_all_memory.py` to clean up both Neo4j instances completely.
   This stops and/or clears the data in both the long-term and short-term databases, allowing you to repeat the workflows from a clean state if desired.

**Note:** The above reinforcement loop showcases how the system "learns" which information to keep. Over time, frequently accessed or explicitly validated facts accumulate a higher confidence score. Once they cross the preset threshold, the system auto-promotes them, ensuring your long-term knowledge base stays up-to-date with proven information while transient or low-value data ages out. You can tune the parameters of this process to suit your needs - for example, adjust the `HIT_WEIGHT`, `VALIDATION_WEIGHT`, or `PROMOTE_THRESHOLD` constants in the configuration to control how quickly facts get promoted. Also adjust `TTL_MS` (the time-to-live for cache entries) to balance cache freshness vs. performance. All these **operational knobs** are exposed so you can calibrate the system’s behavior for your domain.

## Conclusion

By following these workflows, you have deployed a dual-memory Graph RAG system with enterprise enhancements. We used Docker CLI commands to orchestrate a Neo4j-based **long-term memory** and a high-speed **short-term cache**, with Kafka Connect bridging the two in real time. The **graph schema** (documents, paragraphs, entities, relationships) and the Cypher queries remain the same as in the community edition, but now all data movement and life-cycle management are automated via Kafka and storage technology. The result is a retrieval-augmented generation pipeline that is **fast, transparent, and robust**: sub-second query responses with full traceability of which facts were used and how they moved through the system.

For further exploration, consider pointing the ingestion pipeline at your own data sources (e.g. streaming data from internal feeds into the short-term store). The system is designed to be modular - for instance, you can swap in a different graph database that supports Cypher by changing the driver configuration (should work with minimal changes). You can also integrate a larger language model or an API-based model for the LLM component if needed.

> **IMPORTANT:** This is modeled after a specific example, so your implementation might differ materially from the specifics here; however, the high-level concepts are the same.

We hope this reference implementation provides a solid foundation for building **production-ready, enterprise-scale Graph RAG** solutions. By combining graph databases with Kafka-based CDC and advanced storage like FlexCache, you can achieve AI systems that are **faster, clearer, safer, and compliant by design**. Happy building!
