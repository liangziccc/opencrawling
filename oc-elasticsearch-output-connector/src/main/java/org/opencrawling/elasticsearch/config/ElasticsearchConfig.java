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
package org.opencrawling.elasticsearch.config;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.message.BasicHeader;
import org.apache.http.ssl.SSLContexts;
import org.elasticsearch.client.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Configuration
@ConditionalOnProperty(name = "spring.opencrawling.output.type", havingValue = "elasticsearch")
public class ElasticsearchConfig {

    private static final Logger log = LoggerFactory.getLogger(ElasticsearchConfig.class);

    @Value("${spring.opencrawling.output.elasticsearch.uris:http://localhost:9200}")
    private String uris;

    @Value("${spring.opencrawling.output.elasticsearch.username:}")
    private String username;

    @Value("${spring.opencrawling.output.elasticsearch.password:}")
    private String password;

    @Value("${spring.opencrawling.output.elasticsearch.api-key:}")
    private String apiKey;

    @Value("${spring.opencrawling.output.elasticsearch.ssl.trust-store-path:}")
    private String trustStorePath;

    @Value("${spring.opencrawling.output.elasticsearch.ssl.trust-store-password:}")
    private String trustStorePassword;

    @Bean
    public ElasticsearchClient elasticsearchClient() {
        log.info("Initializing ElasticsearchClient connected to: {}", uris);
        try {
            String[] raw = uris.split(",");
            List<HttpHost> hosts = new ArrayList<>();
            for (String uri : raw) {
                String trimmed = uri.trim();
                if (!trimmed.isEmpty()) {
                    hosts.add(HttpHost.create(trimmed));
                }
            }

            RestClient restClient = RestClient.builder(hosts.toArray(new HttpHost[0]))
                    .setHttpClientConfigCallback(httpClientBuilder -> {
                        try {
                            if (trustStorePath != null && !trustStorePath.isBlank()) {
                                httpClientBuilder.setSSLContext(SSLContexts.custom()
                                        .loadTrustMaterial(new File(trustStorePath), trustStorePassword.toCharArray())
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
                    })
                    .build();

            return new ElasticsearchClient(new RestClientTransport(restClient, new JacksonJsonpMapper()));
        } catch (Exception e) {
            log.error("Failed to initialize ElasticsearchClient", e);
            throw new RuntimeException("Elasticsearch client initialization error", e);
        }
    }
}
