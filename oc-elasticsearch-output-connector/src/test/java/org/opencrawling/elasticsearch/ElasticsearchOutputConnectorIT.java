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
package org.opencrawling.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencrawling.core.document.RepositoryDocument;
import org.opencrawling.core.security.PermissionRule;
import org.opencrawling.core.security.SecurityConfig;
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

import java.io.ByteArrayInputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers(disabledWithoutDocker = true)
class ElasticsearchOutputConnectorIT {

    private static final String INDEX = "it_kb";
    private static final int DIMENSIONS = 1024;

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

    private static float[] QUERY_VECTOR = seededVector("acl-probe");

    /** Same wait strategy the testcontainers Elasticsearch module uses, with a longer budget. */
    private static LogMessageWaitStrategy startedLogWait(Duration timeout) {
        return (LogMessageWaitStrategy) new LogMessageWaitStrategy()
                .withRegEx(".*(\"message\":\\s?\"started[\\s?|\"].*|] started\n$)")
                .withStartupTimeout(timeout);
    }

    @Container
    private static final ElasticsearchContainer es = new ElasticsearchContainer(
            DockerImageName.parse("docker.elastic.co/elasticsearch/elasticsearch:8.17.0"))
            .withEnv("xpack.security.enabled", "false")
            .withEnv("xpack.security.http.ssl.enabled", "false")
            // The local test VM has tight resources; the default auto-sized heap gets OOM-killed,
            // so pin a small heap and give Elasticsearch a longer startup budget.
            .withEnv("ES_JAVA_OPTS", "-Xms256m -Xmx512m")
            .waitingFor(startedLogWait(Duration.ofMinutes(5)));

    private static ElasticsearchClient searchClient;
    private static ElasticsearchOutputConnector connector;

    @BeforeAll
    static void setUpAll() {
        RestClient restClient = RestClient.builder(HttpHost.create(es.getHttpHostAddress())).build();
        searchClient = new ElasticsearchClient(new RestClientTransport(restClient, new JacksonJsonpMapper()));

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

        connector = new ElasticsearchOutputConnector(null, dummyModel);
        setField("indexName", INDEX);
        // Force IPv4 loopback: plain http against the mapped container port
        setField("uris", "http://127.0.0.1:" + es.getMappedPort(9200));
    }

    private static void setField(String name, Object value) {
        try {
            java.lang.reflect.Field field = ElasticsearchOutputConnector.class.getDeclaredField(name);
            field.setAccessible(true);
            field.set(connector, value);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to set field " + name, e);
        }
    }

    private RepositoryDocument document(String id, String text, SecurityConfig security) {
        return new RepositoryDocument(
                id,
                "file:///it/" + id + ".txt",
                new ByteArrayInputStream(text.getBytes(java.nio.charset.StandardCharsets.UTF_8)),
                Map.of("mimeType", List.of("text/plain"), "name", List.of(id + ".txt")),
                "public",
                security,
                Instant.parse("2026-01-01T00:00:00Z"));
    }

    @Test
    void ingestsDocumentsAndEnforcesAclOnSearch() throws Exception {
        SecurityConfig restricted = new SecurityConfig(true, List.of(
                new PermissionRule("user1", "user", "User One", "read"),
                new PermissionRule("group1", "group", "Group One", "deny")));

        connector.send(document("acl-doc", "restricted content about vector search", restricted)).block();
        connector.send(document("public-doc", "public content about vector search", null)).block();

        searchClient.indices().refresh(r -> r.index(INDEX));

        long total = searchClient.count(c -> c.index(INDEX)).count();
        assertThat(total).isGreaterThanOrEqualTo(2);

        // kNN search with ACL allow-filter: restricted doc must be visible to user1
        SearchResponse<Void> allowed = knnSearch("user1");
        assertThat(allowed.hits().hits())
                .extracting(Hit::id)
                .anySatisfy(id -> assertThat(id).startsWith("acl-doc"));

        // kNN search as someone denied: restricted doc must NOT be returned
        // (public doc remains visible)
        SearchResponse<Void> denied = knnSearch("someone-else");
        assertThat(denied.hits().hits())
                .extracting(Hit::id)
                .noneSatisfy(id -> assertThat(id).startsWith("acl-doc"));
    }

    @Test
    void rewriteWithDeterministicIdIsIdempotent() throws Exception {
        RepositoryDocument doc = document("idem-doc", "idempotent rewrite content", null);

        connector.send(doc).block();
        searchClient.indices().refresh(r -> r.index(INDEX));
        long countAfterFirst = countByIdemDocUri();

        connector.send(doc).block();
        searchClient.indices().refresh(r -> r.index(INDEX));
        long countAfterSecond = countByIdemDocUri();

        // Deterministic ids (docId + chunkId) make the second write an overwrite, not a duplicate
        assertThat(countAfterFirst).isGreaterThanOrEqualTo(1);
        assertThat(countAfterSecond).isEqualTo(countAfterFirst);
    }

    private long countByIdemDocUri() throws Exception {
        return searchClient.search(s -> s.index(INDEX)
                .trackTotalHits(t -> t.enabled(true))
                .query(q -> q.term(t -> t.field(ElasticsearchConstants.FIELD_URI)
                        .value("file:///it/idem-doc.txt"))), Void.class)
                .hits().total().value();
    }

    @Test
    void vectorRecallReturnsNearestNeighbors() throws Exception {
        String textA = "content about medicine and healthcare recommendations";
        String textB = "content about automotive engine repair and maintenance";

        connector.send(document("doc-recall-a", textA, null)).block();
        connector.send(document("doc-recall-b", textB, null)).block();
        searchClient.indices().refresh(r -> r.index(INDEX));

        // Query with the exact vector of textA: its chunk must rank first (similarity recall)
        List<String> topByA = knnSearchRaw(seededVector(textA));
        assertThat(topByA).isNotEmpty();
        assertThat(topByA.get(0)).startsWith("doc-recall-a");

        // Query with the exact vector of textB: its chunk must rank first
        List<String> topByB = knnSearchRaw(seededVector(textB));
        assertThat(topByB).isNotEmpty();
        assertThat(topByB.get(0)).startsWith("doc-recall-b");
    }

    private List<String> knnSearchRaw(float[] queryVector) throws Exception {
        List<Float> vector = new ArrayList<>();
        for (float v : queryVector) {
            vector.add(v);
        }
        SearchResponse<Void> response = searchClient.search(s -> s.index(INDEX)
                .knn(k -> k
                        .field(ElasticsearchConstants.FIELD_EMBEDDINGS)
                        .queryVector(vector)
                        .k(2)
                        .numCandidates(100)), Void.class);
        List<String> ids = new ArrayList<>();
        response.hits().hits().forEach(h -> ids.add(h.id()));
        return ids;
    }

    private SearchResponse<Void> knnSearch(String allowedIdentity) throws Exception {
        List<Float> vector = new ArrayList<>();
        for (float v : QUERY_VECTOR) {
            vector.add(v);
        }
        return searchClient.search(s -> s.index(INDEX)
                .knn(k -> k
                        .field(ElasticsearchConstants.FIELD_EMBEDDINGS)
                        .queryVector(vector)
                        .k(10)
                        .numCandidates(100)
                        .filter(f -> f.term(t -> t.field(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ)
                                .value(allowedIdentity))))
                , Void.class);
    }
}
