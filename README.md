# Graph‑Based RAG: Better Option for AI Governance

Welcome to the **Graph RAG Guide** – a dual‑memory, knowledge‑graph approach to Retrieval‑Augmented Generation (RAG) that delivers answers that are faster, clearer, and governed by design. The repo houses two fully‑working paths:

* **Community / Open‑Source implementation** – a self‑contained demo you can run on a laptop.
* **Enterprise implementation** – a production‑grade variant that layers in Kafka CDC pipelines, NetApp FlexCache, SnapMirror, and other operational muscle.

By storing knowledge as **nodes & relationships** instead of opaque vectors, the agent gains traceability, reduces hallucinations, and meets demanding audit requirements.

## Project Purpose

* Provide a reference architecture for **graph‑based RAG** with explicit short‑term (cache) and long‑term (authoritative) memory.
* Demonstrate how reinforcement‑learning signals can promote proven facts from cache to durable storage.
* Show upgrade paths – from a minimal Python/RAM‑disk demo to an enterprise pipeline with exactly‑once Kafka connectors and enterprise storage.

## Community vs Enterprise – What Changes?

| Capability                 | Community Edition                                                              | Enterprise Edition                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Data movement**          | Python helper scripts copy sub‑graphs on demand                                | **Kafka Connect** Source + Sink connectors stream CDC events between Neo4j tiers                              |
| **Cache backing store**    | `tmpfs` / RAM‑disk on the dev box                                              | **NetApp FlexCache** for NVMe‑speed reads; resiliency via **SnapMirror** replicas                             |
| **Operational guarantees** | At‑least‑once copies; manual scripts                                           | Exactly‑once semantics, replay, and audit via Kafka offsets                                                   |
| **Governance hooks**       | Basic provenance tags                                                          | Promotion events logged, schema‑versioned, and instantly traceable                                            |
| **Performance**            | Millisecond‑level on warm RAM cache                                            | Sub‑50 ms at global scale; hot blocks auto‑evicted by FlexCache                                               |

> **TL;DR –** start with the community guide for laptops; switch to the enterprise path when you need 24×7, multi‑site, or compliance.

## Where To Dive Deeper

| Document                                      | What it covers                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Knowledge Graphs for Better AI Governance** | Vision & business case for graph‑based RAG [link](./Knowledge_Graphs_for_Better_AI_Governance.md) |
| **Community Implementation Guide**            | Step‑by‑step setup for the open‑source flavour [link](./OSS_Community_Implementation.md)          |
| **Community README**                          | Hands‑on commands & scripts [link](./community_implementation/README.md)                          |
| **Enterprise Implementation Guide**           | Deep dive on Kafka CDC, FlexCache, SnapMirror [link](./Enterprise_Implementation.md)              |
| **Enterprise README**                         | Production deployment notes [link](./enterprise_implementation/README.md)                         |

## Quick Start

```bash
# 1. Clone the repo
$ git clone https://github.com/<your‑org>/graph‑rag‑guide.git

# 2. Pick your path
$ cd graph‑rag‑guide/community_implementation   # laptop demo
# or
$ cd graph‑rag‑guide/enterprise_implementation  # prod‑ready setup

# 3. Follow the README in that folder
```

Questions? Open an issue or start a discussion – contributions are welcome!
