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
import co.elastic.clients.elasticsearch.core.BulkResponse;
import co.elastic.clients.elasticsearch.indices.ElasticsearchIndicesClient;
import co.elastic.clients.transport.endpoints.BooleanResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.opencrawling.core.document.RepositoryDocument;
import org.opencrawling.core.security.PermissionRule;
import org.opencrawling.core.security.SecurityConfig;
import org.springframework.ai.embedding.EmbeddingModel;

import java.io.ByteArrayInputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ElasticsearchOutputConnectorTest {

    private ElasticsearchOutputConnector connector;

    private ElasticsearchClient client;
    private ElasticsearchIndicesClient indicesClient;
    private EmbeddingModel embeddingModel;

    @BeforeEach
    void setUp() {
        client = Mockito.mock(ElasticsearchClient.class);
        indicesClient = Mockito.mock(ElasticsearchIndicesClient.class);
        embeddingModel = Mockito.mock(EmbeddingModel.class);
        connector = new ElasticsearchOutputConnector(client, embeddingModel);
    }

    @Test
    void testGetName() {
        assertThat(connector.getName()).isEqualTo("ElasticsearchOutputConnector");
    }

    @Test
    void testDefaultMappingUsesHnsw() throws Exception {
        String mapping = buildIndexMapping();
        assertThat(mapping)
                .contains("\"type\":\"dense_vector\"")
                .contains("\"dims\":1024")
                .contains("\"similarity\":\"cosine\"")
                .contains("\"index_options\":{\"type\":\"hnsw\"}")
                .contains("security_allowed_read")
                .contains("security_denied_read");
    }

    @Test
    void testInt8HnswMapping() throws Exception {
        setField("indexType", "int8_hnsw");
        String mapping = buildIndexMapping();
        assertThat(mapping).contains("\"index_options\":{\"type\":\"int8_hnsw\"}");
    }

    @Test
    void testFlatMappingDisablesIndex() throws Exception {
        setField("indexType", "flat");
        String mapping = buildIndexMapping();
        assertThat(mapping).contains("\"index\":false");
        assertThat(mapping).doesNotContain("index_options");
    }

    @Test
    void testSendDocumentBulksChunks() throws Exception {
        String contentText = "This is a test content that is long enough to be indexed in the vector store.";
        Map<String, List<String>> metadata = new HashMap<>();
        metadata.put("mimeType", List.of("text/plain"));
        metadata.put("title", List.of("Test Title"));

        SecurityConfig securityConfig = new SecurityConfig(true, List.of(
                new PermissionRule("user1", "user", "User One", "read"),
                new PermissionRule("baduser", "user", "Bad User", "deny")));

        RepositoryDocument doc = new RepositoryDocument(
                "doc-1",
                "file:///test/doc.txt",
                new ByteArrayInputStream(contentText.getBytes(StandardCharsets.UTF_8)),
                metadata,
                "public",
                securityConfig,
                Instant.parse("2026-01-01T00:00:00Z"));

        when(client.indices()).thenReturn(indicesClient);
        when(indicesClient.exists(any(java.util.function.Function.class))).thenReturn(new BooleanResponse(true));
        when(client.bulk(any(BulkRequest.class))).thenReturn(BulkResponse.of(b -> b.took(1).errors(false).items(List.of())));
        when(embeddingModel.embed(any(org.springframework.ai.document.Document.class)))
                .thenReturn(new float[1024]);

        connector.send(doc).block();

        verify(client).bulk(any(BulkRequest.class));
    }

    private String buildIndexMapping() throws Exception {
        Method method = ElasticsearchOutputConnector.class.getDeclaredMethod("buildIndexMapping");
        method.setAccessible(true);
        return (String) method.invoke(connector);
    }

    private void setField(String name, Object value) throws Exception {
        Field field = ElasticsearchOutputConnector.class.getDeclaredField(name);
        field.setAccessible(true);
        field.set(connector, value);
    }
}
