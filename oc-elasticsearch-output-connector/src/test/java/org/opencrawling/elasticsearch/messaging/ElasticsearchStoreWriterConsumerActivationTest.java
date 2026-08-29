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
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

/**
 * Verifies the activation gate of {@link ElasticsearchStoreWriterConsumer}. The Kafka writer that
 * persists embedded chunks must register only when the Elasticsearch output is selected, be enabled
 * by default (so a single-process deployment writes with no extra config), and be able to opt out
 * via {@code opencrawling.consumer.writer.enabled=false} — mirroring the pgvector/Milvus/Qdrant
 * writer consumers.
 */
class ElasticsearchStoreWriterConsumerActivationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withBean(ElasticsearchClient.class, () -> mock(ElasticsearchClient.class))
            .withUserConfiguration(ElasticsearchStoreWriterConsumer.class);

    @Test
    void registersByDefaultWhenElasticsearchOutputSelected() {
        contextRunner
                .withPropertyValues("spring.opencrawling.output.type=elasticsearch")
                .run(context -> assertThat(context).hasSingleBean(ElasticsearchStoreWriterConsumer.class));
    }

    @Test
    void registersWhenWriterExplicitlyEnabled() {
        contextRunner
                .withPropertyValues(
                        "spring.opencrawling.output.type=elasticsearch",
                        "opencrawling.consumer.writer.enabled=true")
                .run(context -> assertThat(context).hasSingleBean(ElasticsearchStoreWriterConsumer.class));
    }

    @Test
    void doesNotRegisterWhenWriterDisabled() {
        contextRunner
                .withPropertyValues(
                        "spring.opencrawling.output.type=elasticsearch",
                        "opencrawling.consumer.writer.enabled=false")
                .run(context -> assertThat(context).doesNotHaveBean(ElasticsearchStoreWriterConsumer.class));
    }

    @Test
    void doesNotRegisterWhenOutputIsPgvector() {
        contextRunner
                .withPropertyValues("spring.opencrawling.output.type=pgvector")
                .run(context -> assertThat(context).doesNotHaveBean(ElasticsearchStoreWriterConsumer.class));
    }

    @Test
    void doesNotRegisterWhenOutputTypeUnset() {
        contextRunner
                .run(context -> assertThat(context).doesNotHaveBean(ElasticsearchStoreWriterConsumer.class));
    }
}
