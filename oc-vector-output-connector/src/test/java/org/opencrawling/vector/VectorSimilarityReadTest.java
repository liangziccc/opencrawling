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
package org.opencrawling.vector;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaEmbeddingModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaEmbeddingOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.jdbc.core.JdbcTemplate;

import java.net.Socket;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Smoke test that reads back vectors from a locally running PostgreSQL (pgvector)
 * using the real Spring AI client stack: query -> Ollama embedding -> kNN search.
 * Skips automatically when local postgres (5432) or ollama (11434) is unreachable.
 */
class VectorSimilarityReadTest {

    private static boolean reachable(String host, int port) {
        try (Socket ignored = new Socket(host, port)) {
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Test
    void similaritySearchAgainstLocalPgVector() {
        if (!reachable("127.0.0.1", 5432) || !reachable("127.0.0.1", 11434)) {
            System.out.println("[SKIP] local postgres/ollama not reachable, skipping read test");
            return;
        }

        HikariConfig hikari = new HikariConfig();
        hikari.setJdbcUrl("jdbc:postgresql://127.0.0.1:5432/opencrawling");
        hikari.setUsername("opencrawling");
        hikari.setPassword("opencrawling_password");
        hikari.setMaximumPoolSize(2);
        JdbcTemplate jdbcTemplate = new JdbcTemplate(new HikariDataSource(hikari));

        OllamaApi ollamaApi = OllamaApi.builder().baseUrl("http://127.0.0.1:11434").build();
        OllamaEmbeddingModel embeddingModel = OllamaEmbeddingModel.builder()
                .ollamaApi(ollamaApi)
                .options(OllamaEmbeddingOptions.builder().model("mxbai-embed-large").build())
                .build();

        PgVectorStore store = PgVectorStore.builder(jdbcTemplate, embeddingModel)
                .vectorTableName("vector_store")
                .dimensions(1024)
                .initializeSchema(false)
                .build();

        List<Document> hits = store.similaritySearch(
                SearchRequest.builder().query("OpenCrawling 能做什么？").topK(5).build());

        System.out.println("=== similaritySearch top-5 结果 ===");
        assertThat(hits).isNotNull();
        for (Document d : hits) {
            String uri = String.valueOf(d.getMetadata().getOrDefault("uri", "?"));
            String preview = d.getText() == null ? "" : d.getText().substring(0, Math.min(60, d.getText().length())).replace('\n', ' ');
            System.out.printf("score=%.4f | uri=%s | %s%n", d.getScore(), uri, preview);
        }
        assertThat(hits).isNotEmpty();
    }
}
