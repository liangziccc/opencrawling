/*
 * Copyright © ${year} the original author or authors (piergiorgio@apache.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.opencrawling.runtime.service;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencrawling.elasticsearch.ElasticsearchConstants;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsAclFilter;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridHit;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridQueryResult;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.KnnNotSupportedException;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.RetrieverUnsupportedException;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.testcontainers.containers.wait.strategy.LogMessageWaitStrategy;
import org.testcontainers.elasticsearch.ElasticsearchContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.io.StringReader;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Integration tests for {@link ElasticsearchInsightsService#runHybridQuery} against a real
 * Elasticsearch cluster (Testcontainers). Covers the hybrid recall behaviour (BM25 + kNN + RRF),
 * the ACL filtering and both risk points:
 * <ul>
 *   <li>risk point 1 - an index whose embeddings field is mapped flat ({@code index:false}) must
 *       fail with the actionable {@link KnnNotSupportedException}, not the raw ES error;</li>
 *   <li>risk point 2 - a cluster older than 8.14 (no retriever/RRF DSL) must fail with the
 *       actionable {@link RetrieverUnsupportedException}; if the low-version image cannot be
 *       pulled (network), the test is skipped without blocking the other verifications.</li>
 * </ul>
 * Vectors come from a deterministic seed (same approach as ElasticsearchOutputConnectorIT) so no
 * real embedding model is required.
 */
@Testcontainers(disabledWithoutDocker = true)
class ElasticsearchInsightsServiceIT {

    private static final String RECALL_INDEX = "it_hybrid_recall";
    private static final String ACL_INDEX = "it_hybrid_acl";
    private static final String FLAT_INDEX = "it_hybrid_flat";
    private static final int DIMENSIONS = 384;

    private static final String RECALL_QUERY = "maintenance of the flux capacitor";
    private static final String ACL_QUERY = "handbook about vector search";

    /** Same wait strategy the testcontainers Elasticsearch module uses, with a longer budget. */
    private static LogMessageWaitStrategy startedLogWait(Duration timeout) {
        return (LogMessageWaitStrategy) new LogMessageWaitStrategy()
                .withRegEx(".*(\"message\":\\s?\"started[\\s?|\"].*|] started\n$)")
                .withStartupTimeout(timeout);
    }

    /**
     * Deterministic per-text vector: same text always maps to the same normalized direction,
     * different texts map to (near-)orthogonal directions. Enables recall assertions
     * without a real embedding model.
     */
    private static float[] seededVector(String text) {
        java.util.Random random = new java.util.Random(text.hashCode());
        float[] vector = new float[DIMENSIONS];
        double norm = 0;
        for (int i = 0; i < DIMENSIONS; i++) {
            vector[i] = (float) random.nextGaussian();
            norm += vector[i] * vector[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < DIMENSIONS; i++) {
            vector[i] = (float) (vector[i] / norm);
        }
        return vector;
    }

    @Container
    private static final ElasticsearchContainer es = new ElasticsearchContainer(
            DockerImageName.parse("docker.elastic.co/elasticsearch/elasticsearch:8.17.0"))
            .withEnv("xpack.security.enabled", "false")
            .withEnv("xpack.security.http.ssl.enabled", "false")
            // The VM also runs the long-lived dev containers; the default auto-sized heap
            // (50% of VM RAM) gets OOM-killed, so pin a small heap. Give ES time to boot too.
            .withEnv("ES_JAVA_OPTS", "-Xms256m -Xmx512m")
            // RRF is license-gated on ES 8.14-8.18 (basic license returns 403); self-generate
            // a trial license so the retriever API is available on the test cluster
            .withEnv("xpack.license.self_generated.type", "trial")
            .waitingFor(startedLogWait(Duration.ofMinutes(5)));

    private static ElasticsearchClient client;
    private static ElasticsearchInsightsService service;
    private static String baseUri;

    @BeforeAll
    static void setUpAll() throws Exception {
        // Force IPv4 loopback: plain http against the mapped container port
        baseUri = "http://127.0.0.1:" + es.getMappedPort(9200);

        RestClient restClient = RestClient.builder(HttpHost.create(es.getHttpHostAddress())).build();
        client = new ElasticsearchClient(new RestClientTransport(restClient, new JacksonJsonpMapper()));

        EmbeddingModel dummyModel = new EmbeddingModel() {
            @Override
            public float[] embed(Document document) {
                return seededVector(document.getText());
            }

            @Override
            public float[] embed(String text) {
                return seededVector(text);
            }

            @Override
            public int dimensions() {
                return DIMENSIONS;
            }

            @Override
            public EmbeddingResponse call(EmbeddingRequest request) {
                List<Embedding> embeddings = request.getInstructions().stream()
                        .map(i -> new Embedding(seededVector(i), null))
                        .collect(Collectors.toList());
                return new EmbeddingResponse(embeddings, null);
            }
        };
        service = new ElasticsearchInsightsService(dummyModel);

        createIndex(RECALL_INDEX, indexedEmbeddingsMapping());
        createIndex(ACL_INDEX, indexedEmbeddingsMapping());
        createIndex(FLAT_INDEX, flatEmbeddingsMapping());
        seedRecallDocs();
        seedAclDocs();
        seedFlatDoc();
    }

    /** Mapping mirroring the connector's hnsw index, used by the happy-path scenarios. */
    private static String indexedEmbeddingsMapping() {
        return "\"type\":\"dense_vector\",\"dims\":" + DIMENSIONS + ",\"index\":true,"
                + "\"similarity\":\"cosine\",\"index_options\":{\"type\":\"hnsw\"}";
    }

    /** Risk point 1: flat (index:false) embeddings - kNN must be rejected by Elasticsearch. */
    private static String flatEmbeddingsMapping() {
        return "\"type\":\"dense_vector\",\"dims\":" + DIMENSIONS + ",\"index\":false";
    }

    private static void createIndex(String index, String embeddingsProperty) throws Exception {
        String mapping = """
                {
                  "mappings": {
                    "properties": {
                      "id": {"type":"keyword"},
                      "text": {"type":"text"},
                      "uri": {"type":"keyword"},
                      "lastModified": {"type":"date"},
                      "acl": {"type":"keyword"},
                      "security_inheritance_enabled": {"type":"boolean"},
                      "security_allowed_read": {"type":"keyword"},
                      "security_denied_read": {"type":"keyword"},
                      "embeddings": {%s}
                    }
                  }
                }
                """.formatted(embeddingsProperty);
        client.indices().create(c -> c.index(index).withJson(new StringReader(mapping)));
    }

    private static void indexDoc(String index, String id, String text, String acl,
                                 List<String> allowedRead, List<String> deniedRead,
                                 float[] vector) throws Exception {
        Map<String, Object> source = new HashMap<>();
        source.put(ElasticsearchConstants.FIELD_ID, id);
        source.put(ElasticsearchConstants.FIELD_TEXT, text);
        source.put(ElasticsearchConstants.FIELD_URI, "file:///it/" + id + ".txt");
        source.put(ElasticsearchConstants.FIELD_LAST_MODIFIED, "2026-01-01T00:00:00Z");
        source.put(ElasticsearchConstants.FIELD_ACL, acl);
        source.put(ElasticsearchConstants.FIELD_SECURITY_INHERITANCE,
                !allowedRead.isEmpty() || !deniedRead.isEmpty());
        source.put(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ, allowedRead);
        source.put(ElasticsearchConstants.FIELD_SECURITY_DENIED_READ, deniedRead);
        List<Float> embeddings = new ArrayList<>();
        for (float v : vector) {
            embeddings.add(v);
        }
        source.put(ElasticsearchConstants.FIELD_EMBEDDINGS, embeddings);
        client.index(i -> i.index(index).id(id).document(source));
    }

    /**
     * Recall corpus: docKeyword matches only the BM25 leg (exact query tokens, unrelated vector),
     * docSemantic matches only the kNN leg (no shared tokens, exact query vector), docNoise
     * matches neither leg.
     */
    private static void seedRecallDocs() throws Exception {
        indexDoc(RECALL_INDEX, "doc-keyword", "Flux Capacitor Maintenance Guide",
                "public", List.of(), List.of(), seededVector("unrelated-sourdough-bread-recipe"));
        indexDoc(RECALL_INDEX, "doc-semantic", "keeping time machine power source healthy",
                "public", List.of(), List.of(), seededVector(RECALL_QUERY));
        indexDoc(RECALL_INDEX, "doc-noise", "quarterly earnings report summary",
                "public", List.of(), List.of(), seededVector("another-unrelated-direction"));
        client.indices().refresh(r -> r.index(RECALL_INDEX));
    }

    private static void seedAclDocs() throws Exception {
        indexDoc(ACL_INDEX, "doc-open", "shared handbook about vector search",
                "public", List.of(), List.of(), seededVector(ACL_QUERY));
        indexDoc(ACL_INDEX, "doc-restricted", "restricted handbook about vector search",
                "private", List.of("user1"), List.of(), seededVector(ACL_QUERY));
        // allowed and denied for the same identity: deny must win
        indexDoc(ACL_INDEX, "doc-denied", "sealed handbook about vector search",
                "private", List.of("user1"), List.of("user1"), seededVector(ACL_QUERY));
        client.indices().refresh(r -> r.index(ACL_INDEX));
    }

    private static void seedFlatDoc() throws Exception {
        indexDoc(FLAT_INDEX, "doc-flat", "flat index content about vector search",
                "public", List.of(), List.of(), seededVector("flat-index-content"));
        client.indices().refresh(r -> r.index(FLAT_INDEX));
    }

    @Test
    void hybridSearchCoversKeywordAndSemanticMatches() {
        EsHybridQueryResult result = service.runHybridQuery(baseUri, RECALL_INDEX, RECALL_QUERY,
                ElasticsearchInsightsService.MODE_HYBRID, 10, null, null, null);

        List<String> uris = result.hits().stream().map(EsHybridHit::uri).toList();
        // The BM25-only document and the kNN-only document must both be recalled by the fusion
        assertThat(uris).contains("file:///it/doc-keyword.txt", "file:///it/doc-semantic.txt");
        // ...and both must outrank the document matching neither leg
        int noiseRank = uris.indexOf("file:///it/doc-noise.txt");
        if (noiseRank >= 0) {
            assertThat(uris.indexOf("file:///it/doc-keyword.txt")).isLessThan(noiseRank);
            assertThat(uris.indexOf("file:///it/doc-semantic.txt")).isLessThan(noiseRank);
        }
        // RRF fusion scores come back ordered, descending
        for (int i = 1; i < result.hits().size(); i++) {
            assertThat(result.hits().get(i).score()).isLessThanOrEqualTo(result.hits().get(i - 1).score());
        }
        assertThat(result.mode()).isEqualTo(ElasticsearchInsightsService.MODE_HYBRID);
        assertThat(result.index()).isEqualTo(RECALL_INDEX);
    }

    @Test
    void keywordModeMatchesOnlyTextualHits() {
        EsHybridQueryResult result = service.runHybridQuery(baseUri, RECALL_INDEX, RECALL_QUERY,
                ElasticsearchInsightsService.MODE_KEYWORD, 10, null, null, null);

        List<String> uris = result.hits().stream().map(EsHybridHit::uri).toList();
        assertThat(uris).contains("file:///it/doc-keyword.txt");
        // The semantic-only document shares no tokens with the query: BM25 cannot recall it
        assertThat(uris).doesNotContain("file:///it/doc-semantic.txt");
        assertThat(result.mode()).isEqualTo(ElasticsearchInsightsService.MODE_KEYWORD);
    }

    @Test
    void vectorModeRanksExactVectorFirst() {
        EsHybridQueryResult result = service.runHybridQuery(baseUri, RECALL_INDEX, RECALL_QUERY,
                ElasticsearchInsightsService.MODE_VECTOR, 10, null, null, null);

        assertThat(result.hits()).isNotEmpty();
        // doc-semantic carries the exact query vector: cosine 1.0 puts it first
        assertThat(result.hits().get(0).uri()).isEqualTo("file:///it/doc-semantic.txt");
        assertThat(result.mode()).isEqualTo(ElasticsearchInsightsService.MODE_VECTOR);
    }

    @Test
    void aclFilterAppliesAllowedAndDeniedRules() {
        // As user1: public docs, docs allowing user1, but NOT docs explicitly denying user1
        EsHybridQueryResult asUser1 = service.runHybridQuery(baseUri, ACL_INDEX, ACL_QUERY,
                ElasticsearchInsightsService.MODE_HYBRID, 10, null, null,
                new EsAclFilter(List.of("user1"), List.of("public", "user1"), List.of("user1")));
        List<String> urisUser1 = asUser1.hits().stream().map(EsHybridHit::uri).toList();
        assertThat(urisUser1).contains("file:///it/doc-open.txt", "file:///it/doc-restricted.txt");
        assertThat(urisUser1).doesNotContain("file:///it/doc-denied.txt");

        // As someone-else: only the public doc remains visible
        EsHybridQueryResult asOther = service.runHybridQuery(baseUri, ACL_INDEX, ACL_QUERY,
                ElasticsearchInsightsService.MODE_HYBRID, 10, null, null,
                new EsAclFilter(List.of("someone-else"), List.of("public", "someone-else"), List.of("someone-else")));
        List<String> urisOther = asOther.hits().stream().map(EsHybridHit::uri).toList();
        assertThat(urisOther).contains("file:///it/doc-open.txt");
        assertThat(urisOther).doesNotContain("file:///it/doc-restricted.txt", "file:///it/doc-denied.txt");

        // The per-hit security fields must be exposed so the MCP layer can re-check ACLs
        EsHybridHit restricted = asUser1.hits().stream()
                .filter(h -> "file:///it/doc-restricted.txt".equals(h.uri()))
                .findFirst().orElseThrow();
        assertThat(restricted.allowedRead()).containsExactly("user1");
    }

    /** Risk point 1: flat (index:false) embeddings must surface as KnnNotSupportedException. */
    @Test
    void flatIndexedEmbeddingsThrowsActionableKnnError() {
        KnnNotSupportedException hybridFailure = assertThrows(KnnNotSupportedException.class,
                () -> service.runHybridQuery(baseUri, FLAT_INDEX, "vector search",
                        ElasticsearchInsightsService.MODE_HYBRID, 5, null, null, null));
        assertThat(hybridFailure.getMessage())
                .contains(FLAT_INDEX)
                .contains(ElasticsearchConstants.FIELD_EMBEDDINGS)
                .contains("该索引不支持 kNN，需重建索引");

        KnnNotSupportedException vectorFailure = assertThrows(KnnNotSupportedException.class,
                () -> service.runHybridQuery(baseUri, FLAT_INDEX, "vector search",
                        ElasticsearchInsightsService.MODE_VECTOR, 5, null, null, null));
        assertThat(vectorFailure.getMessage()).contains("该索引不支持 kNN，需重建索引");

        // Keyword search must keep working on a flat index (BM25 does not need the vector index)
        EsHybridQueryResult keywordResult = service.runHybridQuery(baseUri, FLAT_INDEX, "flat index content",
                ElasticsearchInsightsService.MODE_KEYWORD, 5, null, null, null);
        assertThat(keywordResult.hits()).isNotEmpty();
    }

    /**
     * Risk point 2: a cluster below 8.14 cannot parse the retriever/RRF DSL and must surface as
     * RetrieverUnsupportedException (never silently downgraded). The low-version image is pulled
     * on demand; if that fails (e.g. network restrictions) the verification is skipped with the
     * failure reason recorded, without blocking the other tests.
     */
    @Test
    void oldClusterThrowsActionableVersionError() {
        ElasticsearchContainer oldEs = new ElasticsearchContainer(
                DockerImageName.parse("docker.elastic.co/elasticsearch/elasticsearch:8.7.1"))
                .withEnv("xpack.security.enabled", "false")
                .withEnv("xpack.security.http.ssl.enabled", "false")
                .withEnv("ES_JAVA_OPTS", "-Xms256m -Xmx512m")
                .waitingFor(startedLogWait(Duration.ofMinutes(5)));
        try {
            oldEs.start();
        } catch (Exception e) {
            Assumptions.abort("Skipping low-version cluster risk point verification: could not start "
                    + "elasticsearch:8.7.1 container (likely image pull failure): " + e);
        }
        try {
            String oldUri = "http://127.0.0.1:" + oldEs.getMappedPort(9200);
            RetrieverUnsupportedException failure = assertThrows(RetrieverUnsupportedException.class,
                    () -> service.runHybridQuery(oldUri, "it_old_cluster", "anything",
                            ElasticsearchInsightsService.MODE_HYBRID, 5, null, null, null));
            assertThat(failure.getMessage()).contains("集群版本过低");
            assertThat(failure.getMessage()).contains("8.14");
        } finally {
            oldEs.stop();
        }
    }
}
