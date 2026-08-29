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
package org.opencrawling.elasticsearch.messaging;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import org.opencrawling.core.messaging.DocumentEmbeddedMessage;
import org.opencrawling.elasticsearch.ElasticsearchConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.*;

@Component
// Mirrors VectorStoreWriterConsumer (pgvector) / MilvusStoreWriterConsumer / QdrantStoreWriterConsumer:
// enabled by default so a single-process deployment writes with no extra config, and decoupled
// non-writer services opt out with opencrawling.consumer.writer.enabled=false. Only active when
// the Elasticsearch output is selected.
@ConditionalOnProperty(name = "opencrawling.consumer.writer.enabled", havingValue = "true", matchIfMissing = true)
@ConditionalOnExpression("'${spring.opencrawling.output.type:pgvector}' == 'elasticsearch'")
public class ElasticsearchStoreWriterConsumer {

    private static final Logger log = LoggerFactory.getLogger(ElasticsearchStoreWriterConsumer.class);

    private final ElasticsearchClient client;

    @Value("${spring.opencrawling.output.elasticsearch.index-name:enterprise_kb}")
    private String indexName;

    public ElasticsearchStoreWriterConsumer(ElasticsearchClient client) {
        this.client = client;
    }

    /**
     * Resolves the target index for a message: the per-job outputConfig carried through the
     * Kafka pipeline overrides the static indexName, so different jobs can write to different
     * indices through the same consumer.
     */
    private String resolveTargetIndex(Map<String, String> outputConfig) {
        if (outputConfig == null || outputConfig.isEmpty()) {
            return indexName;
        }
        return outputConfig.getOrDefault("elasticsearchIndexName", indexName);
    }

    @jakarta.annotation.PostConstruct
    public void init() {
        log.info("ElasticsearchStoreWriterConsumer initialized successfully!");
    }

    @KafkaListener(topics = "opencrawling-embedded")
    public void consume(DocumentEmbeddedMessage message) {
        log.info("Received embedded chunk for Elasticsearch storage: {} (Dimensions: {})", message.chunkId(),
            message.embedding() != null ? message.embedding().length : 0);
        try {
            // Map Zero-Trust security ACLs
            List<String> allowedRead = new ArrayList<>();
            List<String> deniedRead = new ArrayList<>();
            boolean inheritanceEnabled = true;

            Object securityObj = message.metadata().get("security");
            if (securityObj instanceof Map) {
                Map<?, ?> securityMap = (Map<?, ?>) securityObj;
                if (securityMap.containsKey("inheritanceEnabled")) {
                    inheritanceEnabled = Boolean.TRUE.equals(securityMap.get("inheritanceEnabled"));
                }
                Object permsObj = securityMap.get("permissions");
                if (permsObj instanceof List) {
                    List<?> permsList = (List<?>) permsObj;
                    for (Object permObj : permsList) {
                        if (permObj instanceof Map) {
                            Map<?, ?> permMap = (Map<?, ?>) permObj;
                            String identity = String.valueOf(permMap.get("identity"));
                            String access = String.valueOf(permMap.get("access"));
                            if ("read".equalsIgnoreCase(access) || "write".equalsIgnoreCase(access)) {
                                allowedRead.add(identity);
                            } else if ("deny".equalsIgnoreCase(access)) {
                                deniedRead.add(identity);
                            }
                        }
                    }
                }
            } else if (securityObj instanceof org.opencrawling.core.security.SecurityConfig) {
                org.opencrawling.core.security.SecurityConfig sc = (org.opencrawling.core.security.SecurityConfig) securityObj;
                inheritanceEnabled = sc.inheritanceEnabled();
                for (org.opencrawling.core.security.PermissionRule rule : sc.permissions()) {
                    if ("read".equalsIgnoreCase(rule.access()) || "write".equalsIgnoreCase(rule.access())) {
                        allowedRead.add(rule.identity());
                    } else if ("deny".equalsIgnoreCase(rule.access())) {
                        deniedRead.add(rule.identity());
                    }
                }
            }

            Map<String, Object> row = new HashMap<>();
            row.put(ElasticsearchConstants.FIELD_ID, message.chunkId());
            row.put(ElasticsearchConstants.FIELD_TEXT, message.text());

            Object uriVal = message.metadata().get("uri");
            row.put(ElasticsearchConstants.FIELD_URI, uriVal != null ? String.valueOf(uriVal) : "");

            Object aclVal = message.metadata().get("acl");
            row.put(ElasticsearchConstants.FIELD_ACL, aclVal != null ? String.valueOf(aclVal) : "");

            Object lastModifiedVal = message.metadata().get("lastModified");
            row.put(ElasticsearchConstants.FIELD_LAST_MODIFIED, lastModifiedVal != null ? String.valueOf(lastModifiedVal) : "");

            row.put(ElasticsearchConstants.FIELD_SECURITY_INHERITANCE, inheritanceEnabled);
            row.put(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ, allowedRead);
            row.put(ElasticsearchConstants.FIELD_SECURITY_DENIED_READ, deniedRead);
            row.put(ElasticsearchConstants.FIELD_EMBEDDINGS, message.embedding());

            // Put other dynamic metadata properties to row
            message.metadata().forEach((key, val) -> {
                if (!"uri".equals(key) && !"acl".equals(key) && !"lastModified".equals(key) && !"security".equals(key)) {
                    row.put(key, val);
                }
            });

            // Resolve target index from the per-job outputConfig when available (falls back
            // to the static configuration for messages from older producers that carry no config)
            String targetIndex = resolveTargetIndex(message.outputConfig());

            // Writing with the chunkId as _id makes re-delivery an idempotent overwrite
            client.index(i -> i
                    .index(targetIndex)
                    .id(message.chunkId())
                    .document(row)
            );
            log.info("Successfully saved chunk {} to Elasticsearch index '{}'.", message.chunkId(), targetIndex);
        } catch (Exception e) {
            log.error("Failed to store embedded chunk in Elasticsearch: {}", message.chunkId(), e);
        }
    }
}
