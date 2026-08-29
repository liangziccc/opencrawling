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
import co.elastic.clients.elasticsearch.core.BulkRequest;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.message.BasicHeader;
import org.apache.http.ssl.SSLContexts;
import org.apache.tika.Tika;
import org.opencrawling.core.connector.OutputConnector;
import org.opencrawling.core.document.RepositoryDocument;
import org.opencrawling.core.security.PermissionRule;
import org.elasticsearch.client.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.io.File;
import java.io.InputStream;
import java.io.StringReader;
import java.util.*;

@Component
@ConditionalOnProperty(name = "spring.opencrawling.output.type", havingValue = "elasticsearch")
public class ElasticsearchOutputConnector implements OutputConnector {

    private static final Logger log = LoggerFactory.getLogger(ElasticsearchOutputConnector.class);

    private final TokenTextSplitter textSplitter;
    private final Tika tika;

    @Value("${spring.opencrawling.output.elasticsearch.uris:http://localhost:9200}")
    private String uris = "http://localhost:9200";

    @Value("${spring.opencrawling.output.elasticsearch.username:}")
    private String username = "";

    @Value("${spring.opencrawling.output.elasticsearch.password:}")
    private String password = "";

    @Value("${spring.opencrawling.output.elasticsearch.api-key:}")
    private String apiKey = "";

    @Value("${spring.opencrawling.output.elasticsearch.index-name:enterprise_kb}")
    private String indexName = "enterprise_kb";

    @Value("${spring.opencrawling.output.elasticsearch.dimensions:1024}")
    private int dimensions = 1024;

    @Value("${spring.opencrawling.output.elasticsearch.similarity:cosine}")
    private String similarity = "cosine";

    @Value("${spring.opencrawling.output.elasticsearch.index-type:hnsw}")
    private String indexType = "hnsw";

    @Value("${spring.opencrawling.output.elasticsearch.ssl.trust-store-path:}")
    private String trustStorePath = "";

    @Value("${spring.opencrawling.output.elasticsearch.ssl.trust-store-password:}")
    private String trustStorePassword = "";

    private ElasticsearchClient client;
    private volatile boolean indexEnsured = false;
    private final EmbeddingModel embeddingModel;

    @Autowired
    public ElasticsearchOutputConnector(
            @Autowired(required = false) ElasticsearchClient client,
            @Autowired(required = false) @org.springframework.beans.factory.annotation.Qualifier("ollamaEmbeddingModel") EmbeddingModel embeddingModel) {
        this(client, embeddingModel, null, null, null, null, null, 0, null, null);
    }

    /**
     * Convenience constructor for programmatic (per-job) instantiation, e.g. the dynamic
     * connector resolution in JobController: configuration values passed here override the
     * defaults (null / non-positive values keep the defaults).
     */
    public ElasticsearchOutputConnector(
            ElasticsearchClient client,
            EmbeddingModel embeddingModel,
            String uris,
            String indexName,
            String username,
            String password,
            String apiKey,
            int dimensions,
            String similarity,
            String indexType) {
        this.client = client;
        this.embeddingModel = embeddingModel;
        if (uris != null && !uris.isBlank()) {
            this.uris = uris;
        }
        if (indexName != null && !indexName.isBlank()) {
            this.indexName = indexName;
        }
        if (username != null) {
            this.username = username;
        }
        if (password != null) {
            this.password = password;
        }
        if (apiKey != null) {
            this.apiKey = apiKey;
        }
        if (dimensions > 0) {
            this.dimensions = dimensions;
        }
        if (similarity != null && !similarity.isBlank()) {
            this.similarity = similarity;
        }
        if (indexType != null && !indexType.isBlank()) {
            this.indexType = indexType;
        }
        this.textSplitter = TokenTextSplitter.builder().build();
        this.tika = new Tika();
    }

    private synchronized ElasticsearchClient getOrInitClient() {
        if (client == null) {
            log.info("Initializing lazy ElasticsearchClient in connector. URIs: {}", uris);
            try {
                RestClient restClient = RestClient.builder(parseHosts()).setHttpClientConfigCallback(httpClientBuilder -> {
                    try {
                        if (!trustStorePath.isBlank()) {
                            char[] storePassword = trustStorePassword.toCharArray();
                            httpClientBuilder.setSSLContext(SSLContexts.custom()
                                    .loadTrustMaterial(new File(trustStorePath), storePassword)
                                    .build());
                        }
                    } catch (Exception e) {
                        throw new IllegalStateException("Failed to load SSL trust store: " + trustStorePath, e);
                    }
                    if (apiKey != null && !apiKey.isBlank()) {
                        // API Key authentication takes precedence when provided
                        return httpClientBuilder.setDefaultHeaders(List.of(
                                new BasicHeader("Authorization", "ApiKey " + apiKey)));
                    }
                    if (username != null && !username.isBlank()) {
                        BasicCredentialsProvider credentialsProvider = new BasicCredentialsProvider();
                        credentialsProvider.setCredentials(AuthScope.ANY,
                                new UsernamePasswordCredentials(username, password));
                        return httpClientBuilder.setDefaultCredentialsProvider(credentialsProvider);
                    }
                    return httpClientBuilder;
                }).build();

                client = new ElasticsearchClient(new RestClientTransport(restClient, new JacksonJsonpMapper()));
            } catch (Exception e) {
                log.error("Failed to initialize ElasticsearchClient", e);
                throw new RuntimeException("Elasticsearch client initialization error", e);
            }
        }
        return client;
    }

    private HttpHost[] parseHosts() {
        String[] raw = uris.split(",");
        List<HttpHost> hosts = new ArrayList<>();
        for (String uri : raw) {
            String trimmed = uri.trim();
            if (!trimmed.isEmpty()) {
                hosts.add(HttpHost.create(trimmed));
            }
        }
        if (hosts.isEmpty()) {
            throw new IllegalArgumentException("No valid Elasticsearch URI provided: " + uris);
        }
        return hosts.toArray(new HttpHost[0]);
    }

    private String buildIndexMapping() {
        String vectorProperty;
        if ("flat".equalsIgnoreCase(indexType)) {
            // No ANN graph: exact (brute-force) vector search semantics
            vectorProperty = "\"type\":\"dense_vector\",\"dims\":" + dimensions + ",\"index\":false";
        } else {
            // "hnsw" (default) or "int8_hnsw" quantized HNSW
            vectorProperty = "\"type\":\"dense_vector\",\"dims\":" + dimensions
                    + ",\"index\":true,\"similarity\":\"" + similarity + "\""
                    + ",\"index_options\":{\"type\":\"" + indexType + "\"}";
        }
        return """
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
                """.formatted(vectorProperty);
    }

    private void ensureIndexExists() throws Exception {
        if (indexEnsured) {
            return;
        }
        synchronized (this) {
            if (indexEnsured) {
                return;
            }
            boolean exists = getOrInitClient().indices().exists(e -> e.index(indexName)).value();
            if (!exists) {
                getOrInitClient().indices().create(c -> c
                        .index(indexName)
                        .withJson(new StringReader(buildIndexMapping())));
                log.info("Created Elasticsearch index '{}' (dims={}, similarity={}, indexType={})",
                        indexName, dimensions, similarity, indexType);
            }
            indexEnsured = true;
        }
    }

    @Override
    public String getName() {
        return "ElasticsearchOutputConnector";
    }

    @Override
    public void connect() throws Exception {
        ensureIndexExists();
    }

    @Override
    public void disconnect() throws Exception {
        client = null;
        indexEnsured = false;
    }

    @Override
    public Mono<Void> send(RepositoryDocument document) {
        return Mono.fromRunnable(() -> {
            try (InputStream is = document.contentStream()) {
                byte[] contentBytes = is.readAllBytes();

                if (contentBytes.length == 0) {
                    log.warn("Document {} content is empty, skipping Elasticsearch ingestion.", document.id());
                    return;
                }

                // Parse document text using Tika
                String text = "";
                try {
                    text = tika.parseToString(new java.io.ByteArrayInputStream(contentBytes));
                } catch (Exception e) {
                    log.warn("Tika failed to parse document {}: {}. Falling back to plain text check.", document.id(), e.getMessage());
                }

                // Fallback for plain text
                if (text.isBlank() && contentBytes.length > 0) {
                    String mimeType = String.valueOf(document.metadata().getOrDefault("mimeType", List.of("text/plain")));
                    if (mimeType.contains("text") || mimeType.contains("json") || mimeType.contains("xml") || mimeType.contains("csv")) {
                        text = new String(contentBytes, java.nio.charset.StandardCharsets.UTF_8);
                    }
                }

                // Remove null characters
                text = text.replace("\u0000", "");

                if (text.isBlank()) {
                    log.warn("Document {} extracted text is empty, skipping Elasticsearch ingestion.", document.id());
                    return;
                }

                log.info("Extracted {} characters from document: {}", text.length(), document.id());

                // Build metadata map
                Map<String, Object> metadata = new HashMap<>();
                document.metadata().forEach((key, val) -> {
                    if (val != null) {
                        List<String> cleanedList = new ArrayList<>();
                        for (String s : val) {
                            if (s != null) {
                                cleanedList.add(s.replace("\u0000", ""));
                            }
                        }
                        metadata.put(key, cleanedList);
                    }
                });
                metadata.put(ElasticsearchConstants.FIELD_URI, document.uri());
                metadata.put(ElasticsearchConstants.FIELD_ACL, document.acl());
                metadata.put(ElasticsearchConstants.FIELD_LAST_MODIFIED, document.lastModified().toString());

                // Construct Spring AI Document for chunking
                Document aiDoc = new Document(document.id(), text, metadata);
                List<Document> chunks = textSplitter.apply(List.of(aiDoc));
                log.info("Split document into {} chunks for Elasticsearch.", chunks.size());

                List<Map<String, Object>> rows = new ArrayList<>();
                int chunkIndex = 0;
                for (Document chunk : chunks) {
                    // Deterministic id (docId + chunk index) makes re-ingestion an idempotent
                    // overwrite instead of a duplicate: the splitter assigns random UUIDs to
                    // chunks, so chunk.getId() must NOT be used here.
                    String chunkId = document.id() + "_" + chunkIndex++;

                    // Get or compute embedding
                    float[] embedding = null;
                    if (embeddingModel != null) {
                        try {
                            embedding = embeddingModel.embed(chunk);
                        } catch (Exception e) {
                            log.debug("Failed embedding from document metadata, trying to embed text directly.", e);
                            embedding = embeddingModel.embed(chunk.getText());
                        }
                    } else {
                        // Fallback/Simulated vector if no model is present (e.g. tests)
                        embedding = new float[dimensions];
                        embedding[0] = 1.0f;
                    }

                    // Map ACL lists
                    List<String> allowedRead = new ArrayList<>();
                    List<String> deniedRead = new ArrayList<>();
                    boolean inheritanceEnabled = true;

                    if (document.security() != null) {
                        inheritanceEnabled = document.security().inheritanceEnabled();
                        for (PermissionRule rule : document.security().permissions()) {
                            if ("read".equalsIgnoreCase(rule.access()) || "write".equalsIgnoreCase(rule.access())) {
                                allowedRead.add(rule.identity());
                            } else if ("deny".equalsIgnoreCase(rule.access())) {
                                deniedRead.add(rule.identity());
                            }
                        }
                    }

                    Map<String, Object> row = new HashMap<>();
                    row.put(ElasticsearchConstants.FIELD_ID, chunkId);
                    row.put(ElasticsearchConstants.FIELD_TEXT, chunk.getText());
                    row.put(ElasticsearchConstants.FIELD_URI, document.uri());
                    row.put(ElasticsearchConstants.FIELD_ACL, document.acl());
                    row.put(ElasticsearchConstants.FIELD_LAST_MODIFIED, document.lastModified().toString());
                    row.put(ElasticsearchConstants.FIELD_SECURITY_INHERITANCE, inheritanceEnabled);
                    row.put(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ, allowedRead);
                    row.put(ElasticsearchConstants.FIELD_SECURITY_DENIED_READ, deniedRead);
                    row.put(ElasticsearchConstants.FIELD_EMBEDDINGS, embedding);

                    // Add other dynamic metadata properties to row
                    metadata.forEach((key, val) -> {
                        if (!ElasticsearchConstants.FIELD_URI.equals(key) &&
                            !ElasticsearchConstants.FIELD_ACL.equals(key) &&
                            !ElasticsearchConstants.FIELD_LAST_MODIFIED.equals(key)) {
                            row.put(key, val);
                        }
                    });

                    rows.add(row);
                }

                // Insert to Elasticsearch
                if (!rows.isEmpty()) {
                    ensureIndexExists();
                    BulkRequest.Builder br = new BulkRequest.Builder();
                    for (Map<String, Object> row : rows) {
                        String id = (String) row.get(ElasticsearchConstants.FIELD_ID);
                        br.operations(op -> op
                            .index(idx -> idx
                                .index(indexName)
                                .id(id)
                                .document(row)
                            )
                        );
                    }
                    getOrInitClient().bulk(br.build());
                    log.info("Successfully added {} chunks for document {} to Elasticsearch index '{}' via Bulk API.",
                            rows.size(), document.id(), indexName);
                }

            } catch (Exception e) {
                log.error("Error processing document {} for Elasticsearch: {}", document.id(), e.getMessage());
                throw new RuntimeException("Failed to process document for Elasticsearch: " + document.id(), e);
            }
        });
    }
}
