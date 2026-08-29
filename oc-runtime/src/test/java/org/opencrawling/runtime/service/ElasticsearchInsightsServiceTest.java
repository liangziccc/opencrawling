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

import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch._types.ErrorResponse;
import co.elastic.clients.elasticsearch._types.query_dsl.Query;
import co.elastic.clients.elasticsearch.core.SearchRequest;
import co.elastic.clients.json.JsonpMapper;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.json.stream.JsonGenerator;
import org.junit.jupiter.api.Test;
import org.opencrawling.elasticsearch.ElasticsearchConstants;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsAclFilter;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.KnnNotSupportedException;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.RetrieverUnsupportedException;

import java.io.StringWriter;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline unit tests for the hybrid search request assembly, parameter defaults, ACL filter
 * construction and error classification (risk points 1 and 2) in ElasticsearchInsightsService.
 */
class ElasticsearchInsightsServiceTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final List<Float> VECTOR = List.of(0.1f, 0.2f, 0.3f);

    private static String toJson(SearchRequest request) throws Exception {
        StringWriter writer = new StringWriter();
        JsonpMapper mapper = new JacksonJsonpMapper();
        try (JsonGenerator generator = mapper.jsonProvider().createGenerator(writer)) {
            request.serialize(generator, mapper);
        }
        return writer.toString();
    }

    private static String toJson(Query query) throws Exception {
        StringWriter writer = new StringWriter();
        JsonpMapper mapper = new JacksonJsonpMapper();
        try (JsonGenerator generator = mapper.jsonProvider().createGenerator(writer)) {
            query.serialize(generator, mapper);
        }
        return writer.toString();
    }

    // --- mode and default resolution --------------------------------------------------------

    @Test
    void normalizeModeDefaultsToHybridAndAcceptsKnownModes() {
        assertEquals(ElasticsearchInsightsService.MODE_HYBRID, ElasticsearchInsightsService.normalizeMode(null));
        assertEquals(ElasticsearchInsightsService.MODE_HYBRID, ElasticsearchInsightsService.normalizeMode("  "));
        assertEquals(ElasticsearchInsightsService.MODE_HYBRID, ElasticsearchInsightsService.normalizeMode("HYBRID"));
        assertEquals(ElasticsearchInsightsService.MODE_KEYWORD, ElasticsearchInsightsService.normalizeMode("keyword"));
        assertEquals(ElasticsearchInsightsService.MODE_VECTOR, ElasticsearchInsightsService.normalizeMode("Vector"));
    }

    @Test
    void normalizeModeRejectsUnknownMode() {
        assertThrows(IllegalArgumentException.class,
                () -> ElasticsearchInsightsService.normalizeMode("semantic"));
    }

    @Test
    void resolveDefaultKeepsPositiveValuesAndFallsBackOtherwise() {
        assertEquals(10, ElasticsearchInsightsService.resolveDefault(null, 10));
        assertEquals(10, ElasticsearchInsightsService.resolveDefault(0, 10));
        assertEquals(10, ElasticsearchInsightsService.resolveDefault(-3, 10));
        assertEquals(7, ElasticsearchInsightsService.resolveDefault(7, 10));
    }

    @Test
    void defaultsMatchTheSpecification() {
        assertEquals(10, ElasticsearchInsightsService.DEFAULT_K);
        assertEquals(100, ElasticsearchInsightsService.DEFAULT_RANK_WINDOW_SIZE);
        assertEquals(60, ElasticsearchInsightsService.DEFAULT_RANK_CONSTANT);
    }

    // --- ACL filter construction --------------------------------------------------------------

    @Test
    void buildAclQueryReturnsNullWhenNothingProvided() throws Exception {
        assertNull(ElasticsearchInsightsService.buildAclQuery(null));
        assertNull(ElasticsearchInsightsService.buildAclQuery(EsAclFilter.NONE));
        assertNull(ElasticsearchInsightsService.buildAclQuery(new EsAclFilter(List.of(), List.of(), List.of())));
    }

    @Test
    void buildAclQueryWithAllowedReadUsesTermsOnSecurityAllowedRead() throws Exception {
        Query query = ElasticsearchInsightsService.buildAclQuery(new EsAclFilter(List.of("user1", "group1"), null, null));
        JsonNode root = MAPPER.readTree(toJson(query));
        JsonNode should = root.path("bool").path("should");
        assertEquals(1, should.size());
        JsonNode values = should.get(0).path("terms").path(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ);
        assertEquals("user1", values.get(0).asText());
        assertEquals("group1", values.get(1).asText());
        assertEquals("1", root.path("bool").path("minimum_should_match").asText());
        assertFalse(root.path("bool").has("must_not"));
    }

    @Test
    void buildAclQueryWithAclUsesTermsOnAclField() throws Exception {
        Query query = ElasticsearchInsightsService.buildAclQuery(new EsAclFilter(null, List.of("public"), null));
        JsonNode root = MAPPER.readTree(toJson(query));
        assertTrue(root.path("bool").path("should").get(0).path("terms").has(ElasticsearchConstants.FIELD_ACL));
    }

    @Test
    void buildAclQueryCombinesAllowedReadAndAclAsShouldClauses() throws Exception {
        Query query = ElasticsearchInsightsService.buildAclQuery(
                new EsAclFilter(List.of("user1"), List.of("public"), null));
        JsonNode should = MAPPER.readTree(toJson(query)).path("bool").path("should");
        assertEquals(2, should.size());
    }

    @Test
    void buildAclQueryWithDeniedReadAddsMustNotClause() throws Exception {
        Query query = ElasticsearchInsightsService.buildAclQuery(
                new EsAclFilter(List.of("user1"), null, List.of("group-denied")));
        JsonNode root = MAPPER.readTree(toJson(query));
        JsonNode mustNot = root.path("bool").path("must_not");
        assertEquals(1, mustNot.size());
        assertTrue(mustNot.get(0).path("terms").has(ElasticsearchConstants.FIELD_SECURITY_DENIED_READ));
    }

    // --- request assembly per mode ------------------------------------------------------------

    @Test
    void hybridModeAssemblesRrfRetrieverWithStandardAndKnnLegs() throws Exception {
        SearchRequest request = ElasticsearchInsightsService.buildHybridSearchRequest(
                "it_kb", ElasticsearchInsightsService.MODE_HYBRID, "zebrafish research",
                VECTOR, 10, 100, 60, null);
        JsonNode root = MAPPER.readTree(toJson(request));

        // "index" travels in the request path, not the body - assert on the model instead
        assertEquals(List.of("it_kb"), request.index());
        assertEquals(10, root.path("size").asInt());

        JsonNode rrf = root.path("retriever").path("rrf");
        assertFalse(rrf.isMissingNode(), "hybrid mode must use the rrf retriever");
        assertEquals(60, rrf.path("rank_constant").asInt());
        assertEquals(100, rrf.path("rank_window_size").asInt());

        JsonNode retrievers = rrf.path("retrievers");
        assertEquals(2, retrievers.size());

        // BM25 leg: plain match on the text field when no ACL filter applies
        JsonNode standardQuery = retrievers.get(0).path("standard").path("query");
        assertEquals("zebrafish research",
                standardQuery.path("match").path(ElasticsearchConstants.FIELD_TEXT).path("query").asText());

        // kNN leg
        JsonNode knn = retrievers.get(1).path("knn");
        assertEquals(ElasticsearchConstants.FIELD_EMBEDDINGS, knn.path("field").asText());
        assertEquals(10, knn.path("k").asInt());
        assertEquals(100, knn.path("num_candidates").asInt());
        assertEquals(3, knn.path("query_vector").size());
        assertFalse(knn.has("filter"));
    }

    @Test
    void hybridModeAppliesAclFilterToBothLegs() throws Exception {
        Query aclQuery = ElasticsearchInsightsService.buildAclQuery(new EsAclFilter(List.of("user1"), null, null));
        SearchRequest request = ElasticsearchInsightsService.buildHybridSearchRequest(
                "it_kb", ElasticsearchInsightsService.MODE_HYBRID, "zebrafish",
                VECTOR, 5, 100, 60, aclQuery);
        JsonNode retrievers = MAPPER.readTree(toJson(request)).path("retriever").path("rrf").path("retrievers");

        // BM25 leg wraps match + filter in a bool query
        JsonNode standardBool = retrievers.get(0).path("standard").path("query").path("bool");
        assertEquals("zebrafish",
                standardBool.path("must").get(0).path("match").path(ElasticsearchConstants.FIELD_TEXT).path("query").asText());
        assertTrue(standardBool.path("filter").get(0).path("bool").path("should").get(0).path("terms")
                .has(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ));

        // kNN leg carries the same filter
        assertTrue(retrievers.get(1).path("knn").has("filter"));
    }

    @Test
    void keywordModeUsesOnlyTheStandardRetriever() throws Exception {
        SearchRequest request = ElasticsearchInsightsService.buildHybridSearchRequest(
                "it_kb", ElasticsearchInsightsService.MODE_KEYWORD, "zebrafish",
                null, 10, 100, 60, null);
        JsonNode root = MAPPER.readTree(toJson(request));

        JsonNode standard = root.path("retriever").path("standard");
        assertFalse(standard.isMissingNode());
        assertEquals("zebrafish",
                standard.path("query").path("match").path(ElasticsearchConstants.FIELD_TEXT).path("query").asText());
        assertFalse(root.path("retriever").has("rrf"));
        assertFalse(root.has("knn"));
    }

    @Test
    void vectorModeReusesTopLevelKnnWithNumCandidatesFloor() throws Exception {
        SearchRequest request = ElasticsearchInsightsService.buildHybridSearchRequest(
                "it_kb", ElasticsearchInsightsService.MODE_VECTOR, "zebrafish",
                VECTOR, 5, 100, 60, null);
        JsonNode root = MAPPER.readTree(toJson(request));

        JsonNode knn = root.path("knn");
        assertTrue(knn.isArray() && !knn.isEmpty(), "vector mode must use the top-level knn clause");
        assertEquals(ElasticsearchConstants.FIELD_EMBEDDINGS, knn.get(0).path("field").asText());
        assertEquals(5, knn.get(0).path("k").asInt());
        // Math.max(k * 10, 100) as in the original runQuery
        assertEquals(100, knn.get(0).path("num_candidates").asInt());
        assertFalse(root.has("retriever"));
    }

    // --- error classification (risk points 1 & 2) ----------------------------------------------

    private static ElasticsearchException esException(int status, String type, String reason) {
        return new ElasticsearchException("search", new ErrorResponse.Builder()
                .error(e -> e.type(type).reason(reason))
                .status(status)
                .build());
    }

    @Test
    void classifierMapsRetrieverParseErrorToVersionHint() {
        ElasticsearchException e = esException(400, "x_content_parse_exception",
                "[1:42] [search] unknown field [retriever]");
        RuntimeException classified = ElasticsearchInsightsService.classifyElasticsearchError(
                e.status(), ElasticsearchInsightsService.collectErrorText(e.error()),
                "it_kb", ElasticsearchInsightsService.MODE_HYBRID, e);

        assertInstanceOf(RetrieverUnsupportedException.class, classified);
        assertTrue(classified.getMessage().contains("集群版本过低"));
        assertTrue(classified.getMessage().contains("8.14"));
    }

    @Test
    void classifierMapsKnnMappingErrorToRebuildHint() {
        ElasticsearchException e = esException(400, "illegal_argument_exception",
                "field [embeddings] of mapping [dense_vector] does not support kNN search");
        RuntimeException classified = ElasticsearchInsightsService.classifyElasticsearchError(
                e.status(), ElasticsearchInsightsService.collectErrorText(e.error()),
                "flat_kb", ElasticsearchInsightsService.MODE_HYBRID, e);

        assertInstanceOf(KnnNotSupportedException.class, classified);
        assertTrue(classified.getMessage().contains("flat_kb"));
        assertTrue(classified.getMessage().contains(ElasticsearchConstants.FIELD_EMBEDDINGS));
        assertTrue(classified.getMessage().contains("该索引不支持 kNN"));
        assertTrue(classified.getMessage().contains("重建索引"));
    }

    @Test
    void classifierIgnoresKnnErrorsForKeywordMode() {
        ElasticsearchException e = esException(400, "illegal_argument_exception",
                "something about knn that does not support anything");
        RuntimeException classified = ElasticsearchInsightsService.classifyElasticsearchError(
                e.status(), ElasticsearchInsightsService.collectErrorText(e.error()),
                "it_kb", ElasticsearchInsightsService.MODE_KEYWORD, e);

        assertSame(e, classified);
    }

    @Test
    void classifierLeavesUnrelatedErrorsUntouched() {
        ElasticsearchException e = esException(500, "search_phase_execution_exception",
                "all shards failed");
        RuntimeException classified = ElasticsearchInsightsService.classifyElasticsearchError(
                e.status(), ElasticsearchInsightsService.collectErrorText(e.error()),
                "it_kb", ElasticsearchInsightsService.MODE_HYBRID, e);

        assertSame(e, classified);
    }

    @Test
    void extractErrorReasonsPullsShardLevelReasonsFromRawBody() {
        // Shape of the real ES 8.17 response when the RRF wrapper wraps a knn-on-flat shard failure
        String raw = "{\"error\":{\"type\":\"status_exception\","
                + "\"reason\":\"[rrf] search failed - retrievers '[knn]' returned errors.\","
                + "\"suppressed\":[{\"type\":\"search_phase_execution_exception\","
                + "\"reason\":\"all shards failed\","
                + "\"failed_shards\":[{\"shard\":0,\"reason\":{\"type\":\"query_shard_exception\","
                + "\"reason\":\"failed to create query: to perform knn search on field [embeddings], "
                + "its mapping must have [index] set to [true]\"}}]}]},\"status\":400}";

        String reasons = ElasticsearchInsightsService.extractErrorReasons(raw);

        assertTrue(reasons.contains("[rrf] search failed"));
        assertTrue(reasons.contains("all shards failed"));
        assertTrue(reasons.contains("its mapping must have [index] set to [true]"));
    }

    @Test
    void classifierPrefersKnnMappingFailureOverRetrieverWrapper() {
        // The RRF wrapper text mentions retrievers, but the actionable cause is the knn mapping:
        // classification must land on KnnNotSupportedException, not the version hint
        String detail = "status_exception: [rrf] search failed - retrievers '[knn]' returned errors. "
                + "search_phase_execution_exception: all shards failed "
                + "failed to create query: to perform knn search on field [embeddings], "
                + "its mapping must have [index] set to [true] ";
        ElasticsearchException e = esException(400, "status_exception",
                "[rrf] search failed - retrievers '[knn]' returned errors.");

        RuntimeException classified = ElasticsearchInsightsService.classifyElasticsearchError(
                400, detail, "flat_kb", ElasticsearchInsightsService.MODE_HYBRID, e);

        assertInstanceOf(KnnNotSupportedException.class, classified);
        assertTrue(classified.getMessage().contains("该索引不支持 kNN"));
    }
}
