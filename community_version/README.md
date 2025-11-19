# Pure Open Source Community Implementation for Graph RAG with Cache and Reinforcement Learning

This README.md provides an **open-source/community-oriented reference implementation** for a Graph-based Retrieval-Augmented Generation (RAG) system, which supports a cache layer for faster access and reinforcement learning to consume new facts. This is modeled after a specific example, so your implementation might differ materially from the specifics here; however, the high-level concepts are the same.

## Prerequisites

- A Linux or Mac-based development machine with sufficient memory to run two Neo4j instances and an LLM (≈8B parameters).
  - *Windows users:* use a Linux VM or cloud instance if possible.
- **Python 3.10+** installed (with [venv](https://docs.python.org/3/library/venv.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) for isolation).
- **Docker** installed (for running Neo4j, etc.).
- Basic familiarity with shell and Docker commands.

**Docker images to pre-pull:**

- `neo4j:5.26` (Neo4j database for both long-term and short-term instances)

### LLM to pre-download:

For example, you can use the following 7-8B parameter models that run locally (CPU-friendly via [llama.cpp](https://github.com/ggerganov/llama.cpp)):

- Intel's **neural-chat-7B-v3-3-GGUF** - *(tested model)* available on HuggingFace
- OR [bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) - an 8B instruction-tuned Llama variant.
- OR use an API/manager like [Ollama](https://ollama.com/), e.g. `ollama pull llama3:8b` for an 8B Llama model.

## Setting Up the Environment

To get started, we need to set up two main components of our environment: a Neo4j graph database and a local LLM for question-answering. We'll use **Docker** to run Neo4j. You'll need to download an LLM (we'll provide some recommendations) and set up a Python environment for our code.

### Demonstration Purposes

For demonstration purposes, we will create a RAM disk and mount it to `$HOME/neo4j-short`. To create and mount the RAM disk, run the following commands below.

> **NOTE:** This RAM disk will automatically disappear on reboot.

### Linux RAM disk

```bash
# 1. Create the mount point
mkdir -p $HOME/neo4j-short

# 2. Mount a tmpfs of the desired size (e.g. 2 GB)
sudo mount -t tmpfs -o size=2G tmpfs $HOME/neo4j-short

# 3. (Optional) Adjust ownership so your user can read/write
sudo chown $USER:$USER $HOME/neo4j-short
```

To clean up the RAM disk, run the following commands:

```bash
sudo umount $HOME/neo4j-short
rmdir $HOME/neo4j-short
```

### MacOS RAM disk

```bash
# 1. Create the mount point
mkdir -p $HOME/neo4j-short

# 2. Attach a RAM-backed device (prints something like /dev/disk4)
DEVICE=$(hdiutil attach -nomount ram://4194304)

# 3. Format it as HFS+ with a volume name
newfs_hfs -v "neo4j-short" $DEVICE

# 4. Mount it to your directory
sudo mount -t hfs $DEVICE $HOME/neo4j-short
```

To clean up the RAM disk, run the following commands:

```bash
sudo umount $HOME/neo4j-short
hdiutil detach $DEVICE
```

### Launch Neo4j with Docker

First, spin up two instances of Neo4j using Docker; an instance for `long-term memory` and one for `short-term memory and caching`. Each instance will expose different ports for bolt and the UI. Below is a **Docker command** that starts a Neo4j instance:

```bash
# Long-Term Memory Instance
# Admin Panel: http://127.0.0.1:7475
# Note the password difference: neo4jneo4j1
docker run -d \
    --name neo4j-long \
    -p 7475:7474  -p 7688:7687 \
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

# short-Term Memory/Cache Instance
# Admin Panel: http://127.0.0.1:7476
# Note the password difference: neo4jneo4j2
docker run -d \
    --name neo4j-short \
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

This will download the Neo4j image (if not already present) and start two Neo4j servers in the background. The Neo4j database will be empty initially. You can verify it's running by opening the Neo4j Browser at **[http://localhost:7475](http://localhost:7475)** and **[http://localhost:7476](http://localhost:7476)** in your browser. Log in with the username `neo4j` and use the corresponding password for your instance. If you see the Neo4j UI, your database is up and ready.

> **IMPORTANT:** The password for the `Long-term Memory` instance is `neo4jneo4j1` and the password for the `Short-term Memory` instance is `neo4jneo4j2`.

### Python Environment and Dependencies

With Neo4j running and the model file ready, set up a Python environment for running the provided code. You should have Python 3.10+ available. It's recommended to use a virtual environment or a Conda environment for the lab.

Install the required Python libraries using pip. A convenient `requirements.txt` file has been provided for you. 

```bash
pip install -r requirements.txt
```

After installing spaCy, download the small English model for NER:

```bash
python -m spacy download en_core_web_sm
```

### Start the shared NER + promotion service

The community workflows now rely on a single Flask service that both performs
spaCy NER and copies supporting facts from the **long-term** graph into the
**short-term** cache when promotion is requested. Launch it in a separate
terminal before running the demos:

```bash
python ner_service.py
```

Environment variables such as `LONG_NEO4J_URI`, `SHORT_NEO4J_URI`,
`PROMOTION_TTL_MS`, and `PROMOTE_DOCUMENT_NODES` are read by the service. They
control which databases it talks to, how long promoted relationships stay
visible, and whether full documents accompany each paragraph in the cache.
Because the promotion runs entirely inside this API, client scripts only need
to send a single HTTP request per query and let the service enforce TTLs and
handle retries/error logging for cache writes.

### Set Up the Local LLM (using llama.cpp)

For the question-answering component, you'll need a local LLM. In this guide, we use a 7B parameter model called **neural-chat-7B-v3-3-GGUF** (a quantized GGUF model) as it can run on CPU with llama.cpp and provides a good balance of performance and size. Using this known model ensures the setup works out of the box.

However, you can substitute a different model if desired. For example, the [bartowski/Meta-Llama-3-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) model (8B parameters) or an Ollama-managed model could be used — the system is not tied to a specific LLM. The key is that the model is accessible via the Python code (in our case, through llama.cpp bindings) and can respond to prompts.

## Background on the Data

Our knowledge source is a collection of BBC news articles in text format, which can be found in the zip file [bbc-example.zip](./bbc-example.zip). This zip file contains a subset of 300 BBC news articles from the 2225 articles in the [BBC Full Text Document Classification](https://bit.ly/4hBKNjp) dataset. After unzipping the archive, the directory structure will look like:

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

Each file is a news article relating to technology in the world today.

You may need to unzip the `bbc-example.zip` file, which you can do by running this script:

```bash
unzip bbc-example.zip
```

## Example Workflows

Here are different workflows that demonstrate how a Graph RAG solution can lead to better AI Governance.

### 1. Simple Query Example

In this scenario, we will ingest data into the long-term graph and then execute a query to see how the system automatically pulls relevant data into the short-term cache (via explicit copying from one database instance to another) to speed up subsequent queries.

1. **Perform the Ingest**: Run `python ingest.py`
   **WARNING:** This will erase any existing nodes and edges in your Neo4j databases (both long-term and short-term) and then reload the BBC dataset afresh. The ingest process will parse the documents, split them into paragraphs, perform NER to extract entities, and merge everything into the **long-term** Neo4j instance. Each relationship is annotated with metadata, including its source document, ingestion timestamp, and a schema version tag for governance purposes. (In an enterprise setting, the ingest script wraps the entire batch in a single transaction and tags it with a batch ID for easier rollback, ensuring an authoritative, idempotent load of knowledge.)

2. **Perform a Simple Query**: Run `python cache_cypher_query.py`
 This script poses a sample question to the system. On the first run, the query will not find any relevant data in the short-term cache (a *cache miss*). The script sends the question to the shared `ner_service.py`, which both extracts entities and copies the long-term subgraph into the short-term database. Every relationship written by the service receives an `expiration` timestamp (driven by `PROMOTION_TTL_MS`) so cached facts naturally age out. Once the service finishes the promotion call, the relevant data resides in the high-speed cache and the question is answered using only the cached subset of the graph, dramatically improving response time. Before handing the context to the LLM, the script now pipes the retrieved paragraphs through a BM25 ranker so the most textually relevant snippets are surfaced first.

3. **Clean Up Short-Term Memory/Cache**: Run `python helper/wipe_short_memory.py`
 This is a maintenance step for the demo, which clears the short-term Neo4j instance of all cached data. In a real deployment, cached entries would naturally expire and be evicted. Each cached relationship has an `expiration` property set (e.g., 1 hour), and a background job or scheduled process can periodically remove those expired edges (while leaving the nodes for audit purposes). For now, we manually wipe the short-term database to simulate that the cache has been cleared of stale data.

### 2. Reinforcement Learning

This workflow illustrates how new facts can be introduced and evaluated in short-term memory, and how the system utilizes a reinforcement learning (RL)- style feedback loop to determine which facts are promoted to long-term memory. We will simulate adding new information and then "teaching" the system through usage and validation. Not depicted in this example is the `CRON` job implementation, which invokes the following scripts: `evict_expired_short_term.py` and `python short_to_long_transfer.py` to handle promotion and expiration of facts from short-term memory.

1. **Perform the Ingest**: Run `python ingest.py`
   **WARNING:** This will erase any existing nodes and edges in your Neo4j databases (both long-term and short-term) and then reload the BBC dataset afresh. The ingest process will parse the documents, split them into paragraphs, perform NER to extract entities, and merge everything into the **long-term** Neo4j instance. Each relationship is annotated with metadata, including its source document, ingestion timestamp, and a schema version tag for governance purposes. (In an enterprise setting, the ingest script wraps the entire batch in a single transaction and tags it with a batch ID for easier rollback, ensuring an authoritative, idempotent load of knowledge.)

2. **Enter New Facts Into Short-Term Memory/Cache**: Run `python example/reinforcement_learning.py`
 This example script will prompt you to introduce five new "facts" into the short-term memory (cache). These could be considered insights or data points not present in the original dataset. For example, facts #1 and #3 are about **OpenAI** (which are not in the BBC tech articles by default). You can choose to inject all or some of these facts; we recommend selecting **fact #1 and fact #3** for this demo. In this simplified script, the use of facts through the queries serves as a reinforcement signal.

3. **Mark One Fact Into Long-Term Memory**: Run `python example/promote_short_term_facts.py`
 This script allows us to flag specific short-term facts as *validated* (worthy of long-term preservation) and others as *expired*. Following the recommendation, select the fact you added as #1 (e.g., the first OpenAI fact) to **promote**, and select fact #3 to let it *expire*.

4. **Tranfer Marked Fact Into Long-Term Memory**: Run `python short_to_long_transfer.py`
 Executing this script transfers the selected facts from the short-term database instance to the long-term database instance for long-term storage.

5. **Expire Short-Term Memory/Cache**: Run `python evict_expired_short_term.py`
 This will remove any remaining facts in the short-term database that have expired or were marked to expire (in our case, fact #3, which we did not promote). This step simulates the cache eviction process for facts that prove not to be useful. In a production environment, such expiration could also be handled by Kafka Connect emitting a *tombstone* event or by a scheduled job that prunes expired data. (For instance, if a fact is not promoted and its time-to-live lapses, it will simply disappear from the cache, leaving only long-term facts stored permanently.)

6. **Verify Fact Can Be Queried**: Run `python example/cypher_query.py`
 This will query the long-term Neo4j instance for the promoted fact (fact #1) to confirm that it now resides in long-term memory. The query should retrieve the fact from the long-term database, demonstrating that the fact was successfully carried over. In contrast, if you query for fact #3 (the one we let expire), it will not be found in either database (having been purged from short-term and never added to long-term).

7. **Reset Both Long-Term and Short-Term Instances**: Run `python helper/wipe_all_memory.py`.
 This stops and/or clears the data in both the long-term and short-term databases, allowing you to repeat the workflows from a clean state if desired.

## Conclusion

By following these workflows, you have deployed a dual-memory Graph RAG Agent. We utilized Docker CLI commands to orchestrate a Neo4j-based **long-term memory** and a high-speed **short-term cache** to build out this solution. The result is a retrieval-augmented generation pipeline that is **fast, transparent, and robust**: delivering sub-second query responses with full traceability of which facts were used and how they were processed through the system.

For further exploration, consider pointing the ingestion pipeline at your data sources (e.g., streaming data from internal feeds into the short-term store). The system is designed to be modular; for instance, you can swap in a different graph database that supports Cypher by changing the driver configuration (which should work with minimal changes). You can also integrate a larger language model or an API-based model for the LLM component if needed.

> **IMPORTANT:** This is modeled after a specific example, so your implementation might differ materially from the specifics here; however, the high-level concepts are the same.

We hope this reference implementation provides a solid foundation for building **production-ready, enterprise-scale Graph RAG** solutions. By combining graph databases with Kafka-based CDC and advanced storage like FlexCache, you can achieve AI systems that are **faster, clearer, safer, and compliant by design**. Happy building!

