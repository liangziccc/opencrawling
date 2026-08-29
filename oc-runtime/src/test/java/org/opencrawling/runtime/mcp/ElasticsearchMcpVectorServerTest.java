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

import org.junit.jupiter.api.Test;
import org.opencrawling.runtime.service.ElasticsearchInsightsService;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsAclFilter;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridHit;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridQueryResult;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.KnnNotSupportedException;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Mirrors VespaMcpVectorServerTest for the Elasticsearch-backed MCP tool: per-hit ACL semantics,
 * caller filter construction and error propagation.
 */
class ElasticsearchMcpVectorServerTest {

    private static final String UNREACHABLE = "http://127.0.0.1:59999";

    private final ElasticsearchMcpVectorServer mcpServer = new ElasticsearchMcpVectorServer(
            new ElasticsearchInsightsService(null), UNREACHABLE, "it_kb");

    private static EsHybridHit hit(String acl, List<String> allowedRead, List<String> deniedRead) {
        return new EsHybridHit("c1", 1.0, "file:///a.txt", "text", acl, allowedRead, deniedRead);
    }

    @Test
    void testSecureSearchUnreachableEndpointThrows() {
        assertThrows(RuntimeException.class, () ->
                mcpServer.esHybridSearch("kubernetes", "user@enterprise.com", "engineering", 5, "hybrid"));
    }

    @Test
    void testSecureSearchPropagatesKnnNotSupportedVerbatim() {
        ElasticsearchMcpVectorServer server = new ElasticsearchMcpVectorServer(
                new ElasticsearchInsightsService(null) {
                    @Override
                    public EsHybridQueryResult runHybridQuery(String uris, String index, String queryText, String mode,
                                                              Integer k, Integer rankWindowSize, Integer rankConstant,
                                                              EsAclFilter aclFilter) {
                        throw new KnnNotSupportedException("该索引不支持 kNN，需重建索引");
                    }
                }, UNREACHABLE, "flat_kb");
        KnnNotSupportedException e = assertThrows(KnnNotSupportedException.class, () ->
                server.esHybridSearch("anything", "user@enterprise.com", null, null, null));
        assertTrue(e.getMessage().contains("该索引不支持 kNN"));
    }

    @Test
    void testIsAccessiblePublicLegacyAclAllowsAnyone() {
        assertTrue(ElasticsearchMcpVectorServer.isAccessible(
                hit("public", List.of(), List.of()), "anyone@enterprise.com", List.of()));
    }

    @Test
    void testIsAccessibleDenyOverridesAllow() {
        EsHybridHit denied = hit("", List.of("finance"), List.of("user@enterprise.com"));
        assertFalse(ElasticsearchMcpVectorServer.isAccessible(denied, "user@enterprise.com", List.of("finance")));
    }

    @Test
    void testIsAccessibleAllowedGroupMatches() {
        EsHybridHit allowed = hit("", List.of("finance"), List.of());
        assertTrue(ElasticsearchMcpVectorServer.isAccessible(allowed, "user@enterprise.com", List.of("finance")));
        assertFalse(ElasticsearchMcpVectorServer.isAccessible(allowed, "user@enterprise.com", List.of("engineering")));
    }

    @Test
    void testIsAccessibleNonEmptyAllowListBeatsPublicAcl() {
        // Document restricted to user1 only: even a "public" flat acl must not grant access
        EsHybridHit restricted = hit("public", List.of("user1"), List.of());
        assertTrue(ElasticsearchMcpVectorServer.isAccessible(restricted, "user1", List.of()));
        assertFalse(ElasticsearchMcpVectorServer.isAccessible(restricted, "someone-else", List.of()));
    }

    @Test
    void testIsAccessibleEmptyRulesFallBackToLegacyAcl() {
        EsHybridHit legacy = hit("finance-manager@enterprise.com", List.of(), List.of());
        assertTrue(ElasticsearchMcpVectorServer.isAccessible(legacy, "finance-manager@enterprise.com", List.of()));
        assertFalse(ElasticsearchMcpVectorServer.isAccessible(legacy, "someone-else@enterprise.com", List.of()));
    }

    @Test
    void testBuildCallerAclFilterCarriesIdentitiesAndPublic() {
        EsAclFilter filter = ElasticsearchMcpVectorServer.buildCallerAclFilter("user1", List.of("finance"));
        assertEquals(List.of("user1", "finance"), filter.allowedRead());
        assertEquals(List.of("user1", "finance"), filter.deniedRead());
        assertTrue(filter.acl().contains("public"));
        assertTrue(filter.acl().contains("user1"));
        assertTrue(filter.acl().contains("finance"));
    }

    @Test
    void testBuildCallerAclFilterAnonymousOnlySeesPublic() {
        EsAclFilter filter = ElasticsearchMcpVectorServer.buildCallerAclFilter(null, null);
        assertNull(filter.allowedRead());
        assertNull(filter.deniedRead());
        assertEquals(List.of("public"), filter.acl());
    }
}
