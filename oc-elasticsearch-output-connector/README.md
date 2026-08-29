# OpenCrawling - Elasticsearch Output Connector

Elasticsearch 8.x output connector for OpenCrawling. Ingests crawled documents as chunked
`dense_vector` documents into an Elasticsearch index, with automatic index provisioning,
zero-trust ACL mapping, and idempotent writes.

## Features

- **Automatic index provisioning**: the target index is created on first write with a
  `dense_vector` HNSW mapping plus keyword/text/date fields — no manual setup required.
- **kNN vector search**: Lucene HNSW approximate nearest neighbor search with configurable
  similarity (`cosine` default, `dot_product`, `l2_norm`).
- **Quantization options**: choose `hnsw` (default, full precision) or `int8_hnsw`
  (int8 scalar quantization, ~75% less memory with minor recall loss) via `index-type`.
- **Zero-trust ACL mapping**: `security_allowed_read` / `security_denied_read` keyword
  fields are indexed for every chunk so retrieval adapters can pre-filter by identity.
- **Idempotent writes**: every chunk is written with a deterministic `_id`
  (`docId_chunkId`) through the Bulk API, so re-ingesting the same document overwrites
  instead of duplicating.

## Configuration

Activate the connector with `spring.opencrawling.output.type=elasticsearch` and configure it under `spring.opencrawling.output.elasticsearch.*`:

| Property | Default | Description |
|---|---|---|
| `uris` | `http://localhost:9200` | Single node or comma-separated list |
| `username` | *(empty)* | Basic auth username |
| `password` | *(empty)* | Basic auth password |
| `api-key` | *(empty)* | API Key authentication (takes precedence over basic auth) |
| `index-name` | `enterprise_kb` | Target index (auto-provisioned) |
| `dimensions` | `1024` | Embedding vector dimensions |
| `similarity` | `cosine` | `cosine`, `dot_product`, or `l2_norm` |
| `index-type` | `hnsw` | `hnsw`, `int8_hnsw`, or `flat` (exact brute-force) |
| `ssl.trust-store-path` | *(empty)* | Trust store for HTTPS clusters with private CAs |
| `ssl.trust-store-password` | *(empty)* | Trust store password |

Example:

```yaml
spring:
  opencrawling:
    output:
      type: elasticsearch
      elasticsearch:
        uris: "https://localhost:9200"
        api-key: "your-api-key"
        index-name: "enterprise_kb"
        dimensions: 1024
        similarity: "cosine"
        index-type: "hnsw"
```

## Local Testing

Start a single-node Elasticsearch 8.x instance with security disabled:

```bash
docker compose -f docker-compose-elasticsearch.yml up -d
```

## Tests

```bash
# Unit tests (mocked client)
mvn test -pl oc-elasticsearch-output-connector

# Integration tests (Testcontainers: real Elasticsearch 8.17 container)
mvn verify -pl oc-elasticsearch-output-connector -DskipITs=false
```

The integration suite covers ACL-enforced kNN search, deterministic-id idempotent
rewrites, and vector recall ranking.
