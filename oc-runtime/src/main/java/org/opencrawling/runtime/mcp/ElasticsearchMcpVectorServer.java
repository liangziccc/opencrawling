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
package org.opencrawling.runtime.mcp;

import org.opencrawling.runtime.service.ElasticsearchInsightsService;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsAclFilter;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridHit;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridQueryResult;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.KnnNotSupportedException;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.RetrieverUnsupportedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.mcp.annotation.McpTool;
import org.springframework.ai.mcp.annotation.McpToolParam;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Elasticsearch-native counterpart to {@link VespaMcpVectorServer}, backed by the hybrid search
 * (BM25 + kNN + RRF) implementation in {@link ElasticsearchInsightsService}. Only active when this
 * instance runs with Elasticsearch as its output store (mirrors
 * {@link org.opencrawling.elasticsearch.config.ElasticsearchConfig}'s condition). Defaults to the
 * "hybrid" mode and enforces the same authoritative per-hit ACL re-check the other MCP servers
 * already perform: the in-index terms filter only narrows candidates, {@link #isAccessible} remains
 * the sole authority on what reaches the LLM.
 */
@Component
@ConditionalOnProperty(name = "opencrawling.mcp.server.enabled", havingValue = "true", matchIfMissing = true)
@ConditionalOnExpression("'${spring.opencrawling.output.type:pgvector}' == 'elasticsearch'")
public class ElasticsearchMcpVectorServer {

    private static final Logger log = LoggerFactory.getLogger(ElasticsearchMcpVectorServer.class);

    private final ElasticsearchInsightsService insightsService;
    private final String uris;
    private final String indexName;

    public ElasticsearchMcpVectorServer(
            ElasticsearchInsightsService insightsService,
            @Value("${spring.opencrawling.output.elasticsearch.uris:http://localhost:9200}") String uris,
            @Value("${spring.opencrawling.output.elasticsearch.index-name:enterprise_kb}") String indexName) {
        this.insightsService = insightsService;
        this.uris = uris;
        this.indexName = indexName;
        log.info("Initialized Elasticsearch-backed Secure MCP tool surface (index '{}' at {}).", indexName, uris);
    }

    public record EsSearchHit(String chunkId, String uri, String text, String acl, double score) {}

    public record EsSecureSearchResult(List<EsSearchHit> hits, boolean degraded, String message) {}

    @McpTool(description = "Perform a secure search on vectorized documents stored in Elasticsearch. Defaults to hybrid search, fusing BM25 keyword scoring with kNN vector similarity through RRF rank fusion in a single query. Results are filtered on the server side using the caller's identity/groups against each document's security rules before anything is returned to the LLM.")
    public EsSecureSearchResult esHybridSearch(
            @McpToolParam(description = "The natural language query or keywords to search for", required = true) String query,
            @McpToolParam(description = "The user principal identity or email of the caller to enforce ACL check", required = true) String userPrincipal,
            @McpToolParam(description = "Comma-separated list of groups/roles the caller belongs to (e.g. 'finance,engineering')", required = false) String userRoles,
            @McpToolParam(description = "Maximum number of results to return (default 5)", required = false) Integer maxResults,
            @McpToolParam(description = "Search mode: 'hybrid' (default, BM25+kNN fused via RRF), 'keyword' (BM25 only), or 'vector' (kNN only)", required = false) String mode
    ) {
        int limit = (maxResults != null && maxResults > 0) ? maxResults : 5;
        List<String> roles = splitRoles(userRoles);
        String effectiveMode = (mode == null || mode.isBlank()) ? ElasticsearchInsightsService.MODE_HYBRID : mode;

        log.info("Elasticsearch MCP secure search. Query: '{}', Principal: '{}', Mode: '{}', Index: '{}'",
                query, userPrincipal, effectiveMode, indexName);

        try {
            // Over-fetch candidates (like VespaMcpVectorServer) so per-hit ACL filtering still
            // yields up to 'limit' accessible results
            int candidateK = Math.max(limit * 3, 30);
            EsAclFilter aclFilter = buildCallerAclFilter(userPrincipal, roles);
            EsHybridQueryResult result = insightsService.runHybridQuery(uris, indexName, query, effectiveMode,
                    candidateK, Math.max(ElasticsearchInsightsService.DEFAULT_RANK_WINDOW_SIZE, candidateK),
                    null, aclFilter);

            List<EsSearchHit> accessible = result.hits().stream()
                    .filter(hit -> isAccessible(hit, userPrincipal, roles))
                    .limit(limit)
                    .map(hit -> new EsSearchHit(hit.chunkId(), hit.uri(), hit.text(), hit.acl(), hit.score()))
                    .toList();
            return new EsSecureSearchResult(accessible, result.message() != null, result.message());
        } catch (KnnNotSupportedException | RetrieverUnsupportedException | IllegalArgumentException e) {
            // Actionable, already clearly worded errors - propagate verbatim to the LLM
            throw e;
        } catch (Exception e) {
            log.error("Elasticsearch MCP secure search failed: {}", e.getMessage(), e);
            throw new RuntimeException("Error executing secure Elasticsearch search", e);
        }
    }

    /**
     * Superset pre-filter pushed into the Elasticsearch query: keeps explicit allows, flat acl
     * matches (incl. "public") and excludes explicit denials for the caller identities. It is
     * deliberately broader than the authoritative check - {@link #isAccessible} re-filters every
     * hit afterwards, e.g. a document with a non-empty allow list that misses the caller is still
     * dropped there even when its flat acl says "public".
     */
    static EsAclFilter buildCallerAclFilter(String userPrincipal, List<String> userGroups) {
        List<String> identities = new ArrayList<>();
        if (userPrincipal != null && !userPrincipal.isBlank()) {
            identities.add(userPrincipal.trim());
        }
        if (userGroups != null) {
            identities.addAll(userGroups);
        }
        if (identities.isEmpty()) {
            // Anonymous callers may only see explicitly public documents
            return new EsAclFilter(null, List.of("public"), null);
        }
        List<String> aclValues = new ArrayList<>();
        aclValues.add("public");
        aclValues.addAll(identities);
        return new EsAclFilter(identities, aclValues, identities);
    }

    // --- ACL enforcement ---------------------------------------------------------------------

    static boolean isAccessible(EsHybridHit hit, String userPrincipal, List<String> userGroups) {
        if (matchesAny(hit.deniedRead(), userPrincipal, userGroups)) {
            return false;
        }
        if (hit.allowedRead() != null && !hit.allowedRead().isEmpty()) {
            return matchesAny(hit.allowedRead(), userPrincipal, userGroups);
        }
        String acl = (hit.acl() == null || hit.acl().isBlank()) ? "public" : hit.acl();
        return matchesAny(List.of(acl), userPrincipal, userGroups);
    }

    private static boolean matchesAny(List<String> identities, String userPrincipal, List<String> userGroups) {
        if (identities == null || identities.isEmpty()) return false;
        for (String identity : identities) {
            if ("public".equalsIgnoreCase(identity)) return true;
            if (userPrincipal != null && userPrincipal.equalsIgnoreCase(identity)) return true;
            if (userGroups != null && userGroups.stream().anyMatch(g -> g.equalsIgnoreCase(identity))) return true;
        }
        return false;
    }

    private static List<String> splitRoles(String userRoles) {
        if (userRoles == null || userRoles.trim().isBlank()) {
            return List.of();
        }
        return Arrays.stream(userRoles.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
    }
}
