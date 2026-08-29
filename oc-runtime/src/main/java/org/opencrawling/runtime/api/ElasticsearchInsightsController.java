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
package org.opencrawling.runtime.api;

import org.opencrawling.runtime.service.ElasticsearchInsightsService;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsAclFilter;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHealthResult;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsHybridQueryResult;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsIndexCount;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.EsQueryResult;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.KnnNotSupportedException;
import org.opencrawling.runtime.service.ElasticsearchInsightsService.RetrieverUnsupportedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/elasticsearch")
public class ElasticsearchInsightsController {

    private static final Logger log = LoggerFactory.getLogger(ElasticsearchInsightsController.class);

    private final ElasticsearchInsightsService insightsService;

    @Autowired
    public ElasticsearchInsightsController(ElasticsearchInsightsService insightsService) {
        this.insightsService = insightsService;
    }

    @GetMapping("/health")
    public EsHealthResult getHealth(@RequestParam String uris) {
        return insightsService.checkHealth(uris);
    }

    @GetMapping("/document-counts")
    public EsIndexCount getDocumentCounts(@RequestParam String uris, @RequestParam String index) {
        return insightsService.getDocumentCounts(uris, index);
    }

    @PostMapping("/query")
    public EsQueryResult runQuery(@RequestBody EsQueryRequest request) {
        return insightsService.runQuery(request.uris(), request.index(), request.queryText(), request.topK());
    }

    public record EsQueryRequest(String uris, String index, String queryText, Integer topK) {}

    /**
     * Hybrid search (BM25 + kNN fused via RRF) against the Elasticsearch output connector's index.
     *
     * Request sample (AI generated):
     * <pre>
     * POST /api/elasticsearch/hybrid-query
     * {
     *   "uris": "http://localhost:9200",
     *   "index": "enterprise_kb",
     *   "queryText": "What is the Claim Check pattern?",
     *   "mode": "hybrid",
     *   "k": 5,
     *   "rankWindowSize": 100,
     *   "rankConstant": 60,
     *   "allowedRead": ["alice@enterprise.com", "engineering"],
     *   "acl": ["public"]
     * }
     * </pre>
     * Response sample (AI generated):
     * <pre>
     * {
     *   "index": "enterprise_kb",
     *   "queryText": "What is the Claim Check pattern?",
     *   "mode": "hybrid",
     *   "hits": [
     *     {
     *       "chunkId": "doc123_0",
     *       "score": 0.016393442,
     *       "uri": "file:///kb/claim-check.md",
     *       "text": "The Claim Check pattern stores large payloads...",
     *       "acl": "public",
     *       "allowedRead": [],
     *       "deniedRead": []
     *     }
     *   ],
     *   "message": null
     * }
     * </pre>
     */
    @PostMapping("/hybrid-query")
    public EsHybridQueryResult runHybridQuery(@RequestBody EsHybridQueryRequest request) {
        return insightsService.runHybridQuery(
                request.uris(),
                request.index(),
                request.queryText(),
                request.mode(),
                request.k(),
                request.rankWindowSize(),
                request.rankConstant(),
                new EsAclFilter(request.allowedRead(), request.acl(), null));
    }

    public record EsHybridQueryRequest(String uris, String index, String queryText, String mode,
                                       Integer k, Integer rankWindowSize, Integer rankConstant,
                                       List<String> allowedRead, List<String> acl) {}

    public record EsHybridQueryError(String error, String message) {}

    /** Risk point 1: the index cannot run kNN (e.g. flat dense_vector mapping) - surface the rebuild hint. */
    @ExceptionHandler(KnnNotSupportedException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public EsHybridQueryError handleKnnNotSupported(KnnNotSupportedException e) {
        log.warn("Hybrid query rejected: {}", e.getMessage());
        return new EsHybridQueryError("knn_not_supported", e.getMessage());
    }

    /** Risk point 2: cluster too old for the retriever/RRF DSL - surface the upgrade hint, no silent fallback. */
    @ExceptionHandler(RetrieverUnsupportedException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public EsHybridQueryError handleRetrieverUnsupported(RetrieverUnsupportedException e) {
        log.warn("Hybrid query rejected: {}", e.getMessage());
        return new EsHybridQueryError("cluster_version_unsupported", e.getMessage());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public EsHybridQueryError handleBadRequest(IllegalArgumentException e) {
        return new EsHybridQueryError("invalid_request", e.getMessage());
    }

    @ExceptionHandler(RuntimeException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public EsHybridQueryError handleUnexpected(RuntimeException e) {
        log.error("Hybrid query failed unexpectedly", e);
        return new EsHybridQueryError("query_failed", e.getMessage());
    }
}
