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

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch._types.ErrorCause;
import co.elastic.clients.elasticsearch._types.FieldValue;
import co.elastic.clients.elasticsearch._types.query_dsl.Query;
import co.elastic.clients.elasticsearch.core.SearchRequest;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.TransportException;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.opencrawling.elasticsearch.ElasticsearchConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Inspection and query-test helpers for the Elasticsearch output connector,
 * mirroring the insights UX of the Vespa connector (health, document counts,
 * and an end-to-end kNN query test from the Admin UI).
 */
@Service
public class ElasticsearchInsightsService {

    private static final Logger log = LoggerFactory.getLogger(ElasticsearchInsightsService.class);

    /** Hybrid search modes supported by {@link #runHybridQuery}. */
    public static final String MODE_HYBRID = "hybrid";
    public static final String MODE_KEYWORD = "keyword";
    public static final String MODE_VECTOR = "vector";

    /** Defaults mandated for the hybrid query endpoint (k=10, rankWindowSize=100, rankConstant=60). */
    public static final int DEFAULT_K = 10;
    public static final int DEFAULT_RANK_WINDOW_SIZE = 100;
    public static final int DEFAULT_RANK_CONSTANT = 60;

    /** Same fallback index name the Elasticsearch output connector defaults to. */
    public static final String DEFAULT_INDEX = "enterprise_kb";

    /** Minimum Elasticsearch cluster version able to parse the retriever/RRF search DSL. */
    static final String MIN_RETRIEVER_VERSION_HINT = "8.14";

    private final EmbeddingModel embeddingModel;

    public ElasticsearchInsightsService(
            @Autowired(required = false) @Qualifier("ollamaEmbeddingModel") EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    public record EsHealthResult(boolean reachable, String clusterName, String version, String message) {}

    public record EsIndexCount(String index, long documentCount, String message) {}

    public record EsQueryHit(double score, String uri, String text) {}

    public record EsQueryResult(String index, String queryText, List<EsQueryHit> hits, String message) {}

    /**
     * One hit of a hybrid query result. Besides score/uri/text (like {@link EsQueryHit}) it carries
     * the chunk id and the security fields so server-side callers (e.g. the MCP tool) can re-check
     * ACLs per hit - the in-index filter alone cannot express the full allow/deny precedence.
     */
    public record EsHybridHit(String chunkId, double score, String uri, String text,
                              String acl, List<String> allowedRead, List<String> deniedRead) {}

    public record EsHybridQueryResult(String index, String queryText, String mode,
                                      List<EsHybridHit> hits, String message) {}

    /**
     * Optional ACL filtering for hybrid queries. All three lists are optional; a document passes
     * when it has no explicit denial for the caller identities AND (it explicitly allows one of
     * them OR its flat "acl" label matches). Mirrors the kNN .filter(...) semantics used by the
     * connector's own integration tests.
     */
    public record EsAclFilter(List<String> allowedRead, List<String> acl, List<String> deniedRead) {
        public static final EsAclFilter NONE = new EsAclFilter(null, null, null);
    }

    /**
     * The target index cannot run kNN at all - e.g. its "embeddings" dense_vector field was mapped
     * with index:false ("flat" index type). The only remedy is rebuilding the index with an indexed
     * dense_vector (hnsw / int8_hnsw), hence the explicit hint instead of a raw Elasticsearch error.
     */
    public static class KnnNotSupportedException extends RuntimeException {
        public KnnNotSupportedException(String message) {
            super(message);
        }
    }

    /**
     * The Elasticsearch cluster is too old to understand the retriever/RRF search DSL (needs 8.14+).
     * Deliberately NOT silently degraded to a plain query: callers must see the version mismatch.
     */
    public static class RetrieverUnsupportedException extends RuntimeException {
        public RetrieverUnsupportedException(String message) {
            super(message);
        }
    }

    public EsHealthResult checkHealth(String uris) {
        try (RestClient restClient = buildRestClient(uris)) {
            ElasticsearchClient client = new ElasticsearchClient(
                    new RestClientTransport(restClient, new JacksonJsonpMapper()));
            var info = client.info();
            return new EsHealthResult(true, info.clusterName(), info.version().number(), null);
        } catch (Exception e) {
            log.warn("Elasticsearch health check failed for {}: {}", uris, e.getMessage());
            return new EsHealthResult(false, null, null, e.getMessage());
        }
    }

    public EsIndexCount getDocumentCounts(String uris, String index) {
        try (RestClient restClient = buildRestClient(uris)) {
            ElasticsearchClient client = new ElasticsearchClient(
                    new RestClientTransport(restClient, new JacksonJsonpMapper()));
            boolean exists = client.indices().exists(e -> e.index(index)).value();
            if (!exists) {
                return new EsIndexCount(index, 0, "index does not exist yet (it is created on first ingestion)");
            }
            long count = client.count(c -> c.index(index)).count();
            return new EsIndexCount(index, count, null);
        } catch (Exception e) {
            log.warn("Elasticsearch document count failed for {}/{}: {}", uris, index, e.getMessage());
            return new EsIndexCount(index, 0, e.getMessage());
        }
    }

    public EsQueryResult runQuery(String uris, String index, String queryText, int topK) {
        if (queryText == null || queryText.isBlank()) {
            return new EsQueryResult(index, queryText, List.of(), "queryText must not be blank");
        }
        if (embeddingModel == null) {
            return new EsQueryResult(index, queryText, List.of(),
                    "no embedding model available (configure spring.ai.ollama.* to enable query tests)");
        }
        int k = topK > 0 ? topK : 5;
        try (RestClient restClient = buildRestClient(uris)) {
            ElasticsearchClient client = new ElasticsearchClient(
                    new RestClientTransport(restClient, new JacksonJsonpMapper()));

            float[] vector = embeddingModel.embed(queryText);
            List<Float> queryVector = new ArrayList<>(vector.length);
            for (float v : vector) {
                queryVector.add(v);
            }

            SearchResponse<java.util.Map> response = client.search(s -> s.index(index)
                    .knn(knn -> knn
                            .field(ElasticsearchConstants.FIELD_EMBEDDINGS)
                            .queryVector(queryVector)
                            .k(k)
                            .numCandidates(Math.max(k * 10, 100))), java.util.Map.class);

            List<EsQueryHit> hits = new ArrayList<>();
            for (Hit<java.util.Map> hit : response.hits().hits()) {
                java.util.Map<?, ?> source = hit.source();
                hits.add(new EsQueryHit(
                        hit.score() != null ? hit.score() : 0.0,
                        source != null ? String.valueOf(source.get(ElasticsearchConstants.FIELD_URI)) : null,
                        source != null ? String.valueOf(source.get(ElasticsearchConstants.FIELD_TEXT)) : null));
            }
            return new EsQueryResult(index, queryText, hits, null);
        } catch (Exception e) {
            log.warn("Elasticsearch query test failed for {}/{}: {}", uris, index, e.getMessage());
            return new EsQueryResult(index, queryText, List.of(), e.getMessage());
        }
    }

    /**
     * Hybrid search over the Elasticsearch output connector's index, fusing BM25 keyword scoring and
     * kNN vector similarity through Elasticsearch's Retriever API with RRF rank fusion.
     *
     * @param uris            comma separated Elasticsearch http(s) URIs
     * @param index           target index; blank falls back to {@link #DEFAULT_INDEX}
     * @param queryText       natural language query (used for BM25 and to build the query embedding)
     * @param mode            "hybrid" (default), "keyword" (standard/match retriever only) or "vector" (kNN only)
     * @param k               number of hits to return; defaults to {@value #DEFAULT_K}
     * @param rankWindowSize  RRF rank window size; defaults to {@value #DEFAULT_RANK_WINDOW_SIZE}
     * @param rankConstant    RRF rank constant; defaults to {@value #DEFAULT_RANK_CONSTANT}
     * @param aclFilter       optional ACL filter; {@code null} or {@link EsAclFilter#NONE} disables filtering
     * @throws IllegalArgumentException        on blank query text or unknown mode
     * @throws KnnNotSupportedException        when the index's embeddings field cannot run kNN (e.g. flat mapping)
     * @throws RetrieverUnsupportedException   when the cluster is older than 8.14 and cannot parse retrievers
     */
    public EsHybridQueryResult runHybridQuery(String uris, String index, String queryText, String mode,
                                              Integer k, Integer rankWindowSize, Integer rankConstant,
                                              EsAclFilter aclFilter) {
        if (queryText == null || queryText.isBlank()) {
            throw new IllegalArgumentException("queryText must not be blank");
        }
        String effectiveMode = normalizeMode(mode);
        String effectiveIndex = (index == null || index.isBlank()) ? DEFAULT_INDEX : index;
        int effectiveK = resolveDefault(k, DEFAULT_K);
        // The rank window must cover k, otherwise Elasticsearch rejects the RRF request
        int effectiveWindow = Math.max(resolveDefault(rankWindowSize, DEFAULT_RANK_WINDOW_SIZE), effectiveK);
        int effectiveConstant = resolveDefault(rankConstant, DEFAULT_RANK_CONSTANT);
        EsAclFilter effectiveFilter = aclFilter == null ? EsAclFilter.NONE : aclFilter;

        String degradedMessage = null;
        boolean needsEmbedding = !MODE_KEYWORD.equals(effectiveMode);
        if (needsEmbedding && embeddingModel == null) {
            // Explicit (never silent) degradation, mirroring VespaInsightsService behaviour
            effectiveMode = MODE_KEYWORD;
            needsEmbedding = false;
            degradedMessage = "no embedding model available (configure spring.ai.ollama.* to enable "
                    + "vector search); ran keyword-only search instead";
        }

        List<Float> queryVector = null;
        if (needsEmbedding) {
            float[] vector;
            try {
                vector = embeddingModel.embed(queryText);
            } catch (Exception e) {
                throw new RuntimeException("Failed to compute the query embedding for hybrid search: "
                        + e.getMessage(), e);
            }
            queryVector = new ArrayList<>(vector.length);
            for (float v : vector) {
                queryVector.add(v);
            }
        }

        Query aclQuery = buildAclQuery(effectiveFilter);
        try (RestClient restClient = buildRestClient(uris)) {
            ElasticsearchClient client = new ElasticsearchClient(
                    new RestClientTransport(restClient, new JacksonJsonpMapper()));

            SearchRequest request = buildHybridSearchRequest(effectiveIndex, effectiveMode, queryText,
                    queryVector, effectiveK, effectiveWindow, effectiveConstant, aclQuery);
            SearchResponse<java.util.Map> response = client.search(request, java.util.Map.class);

            List<EsHybridHit> hits = new ArrayList<>();
            for (Hit<java.util.Map> hit : response.hits().hits()) {
                java.util.Map<?, ?> source = hit.source();
                String chunkId = source != null && source.get(ElasticsearchConstants.FIELD_ID) != null
                        ? String.valueOf(source.get(ElasticsearchConstants.FIELD_ID))
                        : hit.id();
                hits.add(new EsHybridHit(
                        chunkId,
                        hit.score() != null ? hit.score() : 0.0,
                        source != null ? String.valueOf(source.get(ElasticsearchConstants.FIELD_URI)) : null,
                        source != null ? String.valueOf(source.get(ElasticsearchConstants.FIELD_TEXT)) : null,
                        source != null ? asString(source.get(ElasticsearchConstants.FIELD_ACL)) : null,
                        source != null ? asStringList(source.get(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ)) : List.of(),
                        source != null ? asStringList(source.get(ElasticsearchConstants.FIELD_SECURITY_DENIED_READ)) : List.of()));
            }
            return new EsHybridQueryResult(effectiveIndex, queryText, effectiveMode, hits, degradedMessage);
        } catch (ElasticsearchException e) {
            // Risk points 1 & 2: translate opaque 400-class failures into actionable errors.
            // The raw body keeps shard-level reasons (failed_shards[].reason) that the typed
            // ErrorCause graph drops - e.g. the knn-on-flat-field message wrapped by RRF.
            String detail = collectErrorText(e.error()) + extractErrorReasons(rawErrorBody(e));
            throw classifyElasticsearchError(e.status(), detail, effectiveIndex, effectiveMode, e);
        } catch (TransportException e) {
            throw classifyElasticsearchError(-1, e.getMessage() == null ? "" : e.getMessage(), effectiveIndex,
                    effectiveMode, new RuntimeException(e.getMessage(), e));
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            log.warn("Elasticsearch hybrid query failed for {}/{}: {}", uris, effectiveIndex, e.getMessage());
            throw new RuntimeException("Elasticsearch hybrid query failed for index '" + effectiveIndex
                    + "': " + e.getMessage(), e);
        }
    }

    /** Normalizes the search mode; null/blank defaults to hybrid, anything unknown is rejected. */
    static String normalizeMode(String mode) {
        if (mode == null || mode.isBlank()) {
            return MODE_HYBRID;
        }
        String normalized = mode.trim().toLowerCase(Locale.ROOT);
        if (MODE_HYBRID.equals(normalized) || MODE_KEYWORD.equals(normalized) || MODE_VECTOR.equals(normalized)) {
            return normalized;
        }
        throw new IllegalArgumentException("Unsupported search mode '" + mode
                + "': expected one of hybrid, keyword, vector");
    }

    /** Resolves an optional positive parameter against its default. */
    static int resolveDefault(Integer value, int defaultValue) {
        return (value != null && value > 0) ? value : defaultValue;
    }

    /**
     * Builds the ACL filter query shared by the BM25 leg (bool filter clause) and the kNN leg
     * (retriever filter). Returns {@code null} when no filtering is requested.
     */
    static Query buildAclQuery(EsAclFilter aclFilter) {
        if (aclFilter == null) {
            return null;
        }
        boolean hasAllowed = aclFilter.allowedRead() != null && !aclFilter.allowedRead().isEmpty();
        boolean hasAcl = aclFilter.acl() != null && !aclFilter.acl().isEmpty();
        boolean hasDenied = aclFilter.deniedRead() != null && !aclFilter.deniedRead().isEmpty();
        if (!hasAllowed && !hasAcl && !hasDenied) {
            return null;
        }
        List<Query> should = new ArrayList<>();
        if (hasAllowed) {
            should.add(termsQuery(ElasticsearchConstants.FIELD_SECURITY_ALLOWED_READ, aclFilter.allowedRead()));
        }
        if (hasAcl) {
            should.add(termsQuery(ElasticsearchConstants.FIELD_ACL, aclFilter.acl()));
        }
        return Query.of(q -> q.bool(b -> {
            if (!should.isEmpty()) {
                b.should(should).minimumShouldMatch("1");
            }
            if (hasDenied) {
                b.mustNot(termsQuery(ElasticsearchConstants.FIELD_SECURITY_DENIED_READ, aclFilter.deniedRead()));
            }
            return b;
        }));
    }

    private static Query termsQuery(String field, List<String> values) {
        List<FieldValue> fieldValues = new ArrayList<>();
        for (String value : values) {
            fieldValues.add(FieldValue.of(value));
        }
        return Query.of(q -> q.terms(t -> t.field(field).terms(tv -> tv.value(fieldValues))));
    }

    /**
     * Assembles the mode specific {@link SearchRequest}. Package visible so unit tests can assert on
     * the serialized request shape without a live cluster.
     */
    static SearchRequest buildHybridSearchRequest(String index, String mode, String queryText,
                                                  List<Float> queryVector, int k, int rankWindowSize,
                                                  int rankConstant, Query aclQuery) {
        SearchRequest.Builder builder = new SearchRequest.Builder().index(index).size(k);
        switch (mode) {
            case MODE_HYBRID -> builder.retriever(r -> r.rrf(rrf -> rrf
                    .retrievers(standardRetriever -> standardRetriever.standard(standard ->
                            standard.query(keywordQuery(queryText, aclQuery))))
                    .retrievers(knnRetriever -> knnRetriever.knn(knn -> {
                        knn.field(ElasticsearchConstants.FIELD_EMBEDDINGS)
                                .queryVector(queryVector)
                                .k(k)
                                .numCandidates(k * 10);
                        if (aclQuery != null) {
                            knn.filter(aclQuery);
                        }
                        return knn;
                    }))
                    .rankConstant(rankConstant)
                    .rankWindowSize(rankWindowSize)));
            case MODE_KEYWORD -> builder.retriever(r -> r.standard(standard ->
                    standard.query(keywordQuery(queryText, aclQuery))));
            case MODE_VECTOR -> builder.knn(knn -> {
                knn.field(ElasticsearchConstants.FIELD_EMBEDDINGS)
                        .queryVector(queryVector)
                        .k(k)
                        .numCandidates(Math.max(k * 10, 100));
                if (aclQuery != null) {
                    knn.filter(aclQuery);
                }
                return knn;
            });
            default -> throw new IllegalArgumentException("Unsupported search mode '" + mode
                    + "': expected one of hybrid, keyword, vector");
        }
        return builder.build();
    }

    /** BM25 leg: match on the text field, wrapped in a bool query when an ACL filter applies. */
    private static Query keywordQuery(String queryText, Query aclQuery) {
        if (aclQuery == null) {
            return Query.of(q -> q.match(m -> m.field(ElasticsearchConstants.FIELD_TEXT).query(queryText)));
        }
        return Query.of(q -> q.bool(b -> b
                .must(m -> m.match(mm -> mm.field(ElasticsearchConstants.FIELD_TEXT).query(queryText)))
                .filter(aclQuery)));
    }

    /**
     * Maps raw Elasticsearch failures to the two actionable error types this feature defines:
     * risk point 1 (kNN unavailable on the index mapping) and risk point 2 (cluster too old for
     * the retriever API). Anything else is rethrown unchanged.
     */
    static RuntimeException classifyElasticsearchError(int status, String errorText, String index,
                                                       String mode, RuntimeException original) {
        String text = errorText == null ? "" : errorText.toLowerCase(Locale.ROOT);
        // Check the kNN-mapping failure first: on newer clusters its message may also mention the
        // surrounding retriever wrapper, and must not be mistaken for a version incompatibility.
        boolean usesKnn = MODE_HYBRID.equals(mode) || MODE_VECTOR.equals(mode);
        if (status == 400 && usesKnn && text.contains("knn")
                && (text.contains("does not support") || text.contains("not indexed")
                        || text.contains("cannot") || text.contains("unsupported")
                        || text.contains("index false") || text.contains("flat")
                        || text.contains("set to [true]")
                        // RRF wrapper when the kNN leg fails on the shards (e.g. flat mapping);
                        // the shard-level reason is dropped by the typed error graph
                        || text.contains("[knn]' returned errors"))) {
            return new KnnNotSupportedException("kNN search is not available on index '" + index
                    + "' (field '" + ElasticsearchConstants.FIELD_EMBEDDINGS + "'): " + trimTo(errorText, 300)
                    + " 该索引不支持 kNN，需重建索引（dense_vector 需启用索引，如 index-type hnsw 或 int8_hnsw，"
                    + "而非 flat）。");
        }
        if (status == 400 && (text.contains("retriever") || text.contains("rrf"))) {
            return new RetrieverUnsupportedException("The Elasticsearch cluster does not support the "
                    + "retriever/RRF search API required for hybrid search: " + trimTo(errorText, 300)
                    + " 集群版本过低（需 ≥ " + MIN_RETRIEVER_VERSION_HINT + "），请升级集群后重试。");
        }
        return original;
    }

    /**
     * Reads the raw JSON error body of a failed request. Returns an empty string when the body
     * is unavailable (never throws): classification then falls back to the typed error alone.
     */
    static String rawErrorBody(ElasticsearchException e) {
        try {
            var httpResponse = e.httpResponse();
            if (httpResponse != null && httpResponse.body() != null) {
                java.nio.ByteBuffer buffer = httpResponse.body().asByteBuffer();
                byte[] bytes = new byte[buffer.remaining()];
                buffer.get(bytes);
                return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
            }
        } catch (Exception ex) {
            log.debug("Could not read raw Elasticsearch error body: {}", ex.getMessage());
        }
        return "";
    }

    /**
     * Extracts every {@code "reason"} value from a raw Elasticsearch error JSON (wrapper,
     * shard failures, caused_by chains), so actionable messages survive the typed parsing.
     */
    static String extractErrorReasons(String rawJson) {
        if (rawJson == null || rawJson.isEmpty()) {
            return "";
        }
        java.util.regex.Matcher matcher = java.util.regex.Pattern
                .compile("\"reason\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"").matcher(rawJson);
        StringBuilder sb = new StringBuilder();
        while (matcher.find()) {
            String reason = matcher.group(1).replace("\\\"", "\"").replace("\\\\", "\\");
            if (!reason.isBlank()) {
                sb.append(reason).append(' ');
            }
        }
        return sb.toString();
    }

    /** Flattens an ErrorCause tree (reason, caused_by, root_cause) into one searchable string. */
    static String collectErrorText(ErrorCause cause) {
        StringBuilder sb = new StringBuilder();
        appendErrorText(cause, sb);
        return sb.toString();
    }

    private static void appendErrorText(ErrorCause cause, StringBuilder sb) {
        if (cause == null) {
            return;
        }
        if (cause.type() != null) {
            sb.append(cause.type()).append(": ");
        }
        if (cause.reason() != null) {
            sb.append(cause.reason()).append(' ');
        }
        appendErrorText(cause.causedBy(), sb);
        if (cause.rootCause() != null) {
            for (ErrorCause root : cause.rootCause()) {
                appendErrorText(root, sb);
            }
        }
        // The RRF wrapper reports per-retriever shard failures as suppressed causes; they
        // carry the actionable reason (e.g. the knn-on-flat-field message)
        if (cause.suppressed() != null) {
            for (ErrorCause suppressed : cause.suppressed()) {
                appendErrorText(suppressed, sb);
            }
        }
    }

    private static String trimTo(String text, int maxLength) {
        if (text == null) {
            return "";
        }
        return text.length() <= maxLength ? text : text.substring(0, maxLength) + "... ";
    }

    private static String asString(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    @SuppressWarnings("unchecked")
    private static List<String> asStringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> result = new ArrayList<>();
        for (Object item : (List<Object>) list) {
            if (item != null) {
                result.add(String.valueOf(item));
            }
        }
        return result;
    }

    private RestClient buildRestClient(String uris) {
        List<HttpHost> hosts = new ArrayList<>();
        for (String uri : uris.split(",")) {
            String trimmed = uri.trim();
            if (!trimmed.isEmpty()) {
                hosts.add(HttpHost.create(trimmed));
            }
        }
        return RestClient.builder(hosts.toArray(new HttpHost[0])).build();
    }
}
