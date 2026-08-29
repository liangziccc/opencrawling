import { useState } from 'react'
import { Search, Database, Activity, Loader2 } from 'lucide-react'

interface ElasticsearchInsightsProps {
  uris: string
  indexName: string
}

interface HealthResult {
  reachable: boolean
  clusterName: string | null
  version: string | null
  message: string | null
}

interface IndexCount {
  index: string
  documentCount: number
  message: string | null
}

interface QueryHit {
  chunkId: string | null
  score: number
  uri: string | null
  text: string | null
}

interface QueryResult {
  index: string
  queryText: string
  mode: string
  hits: QueryHit[]
  message: string | null
}

type SearchMode = 'hybrid' | 'keyword' | 'vector'

export default function ElasticsearchInsights({ uris, indexName }: ElasticsearchInsightsProps) {
  const [health, setHealth] = useState<HealthResult | null>(null)
  const [counts, setCounts] = useState<IndexCount | null>(null)
  const [queryText, setQueryText] = useState('')
  const [searchMode, setSearchMode] = useState<SearchMode>('hybrid')
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [loadingHealth, setLoadingHealth] = useState(false)
  const [loadingCounts, setLoadingCounts] = useState(false)
  const [loadingQuery, setLoadingQuery] = useState(false)

  const checkHealth = async () => {
    setLoadingHealth(true)
    try {
      const res = await fetch(`/api/elasticsearch/health?uris=${encodeURIComponent(uris)}`)
      setHealth(await res.json())
    } catch (e) {
      setHealth({ reachable: false, clusterName: null, version: null, message: String(e) })
    } finally {
      setLoadingHealth(false)
    }
  }

  const loadCounts = async () => {
    setLoadingCounts(true)
    try {
      const res = await fetch(`/api/elasticsearch/document-counts?uris=${encodeURIComponent(uris)}&index=${encodeURIComponent(indexName)}`)
      setCounts(await res.json())
    } catch (e) {
      setCounts({ index: indexName, documentCount: 0, message: String(e) })
    } finally {
      setLoadingCounts(false)
    }
  }

  const runQuery = async () => {
    if (!queryText.trim()) return
    setLoadingQuery(true)
    try {
      const res = await fetch('/api/elasticsearch/hybrid-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uris, index: indexName, queryText, mode: searchMode, k: 5 }),
      })
      const data = await res.json()
      if (!res.ok) {
        // endpoint returns { error, message } bodies for knn/version/request failures
        setQueryResult({ index: indexName, queryText, mode: searchMode, hits: [], message: data.message || String(data.error) })
      } else {
        setQueryResult(data)
      }
    } catch (e) {
      setQueryResult({ index: indexName, queryText, mode: searchMode, hits: [], message: String(e) })
    } finally {
      setLoadingQuery(false)
    }
  }

  return (
    <div className="mt-6 p-4 rounded-lg border border-border bg-muted/30 space-y-4">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <Activity className="h-4 w-4 text-green-500" />
        <span>Elasticsearch Insights &amp; Query Test</span>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={checkHealth}
          disabled={loadingHealth}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-primary/10 text-primary hover:bg-primary/20 disabled:opacity-50"
        >
          {loadingHealth ? <Loader2 className="h-3 w-3 animate-spin" /> : <Activity className="h-3 w-3" />}
          Check Health
        </button>
        <button
          type="button"
          onClick={loadCounts}
          disabled={loadingCounts}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-primary/10 text-primary hover:bg-primary/20 disabled:opacity-50"
        >
          {loadingCounts ? <Loader2 className="h-3 w-3 animate-spin" /> : <Database className="h-3 w-3" />}
          Document Count
        </button>
      </div>

      {health && (
        <div className={`text-xs p-2 rounded ${health.reachable ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
          {health.reachable
            ? `Connected: cluster "${health.clusterName}" (Elasticsearch ${health.version})`
            : `Unreachable: ${health.message}`}
        </div>
      )}

      {counts && (
        <div className={`text-xs p-2 rounded ${counts.message ? 'bg-yellow-500/10 text-yellow-400' : 'bg-green-500/10 text-green-400'}`}>
          {counts.message
            ? `Index "${counts.index}": ${counts.message}`
            : `Index "${counts.index}" contains ${counts.documentCount} documents`}
        </div>
      )}

      <div className="space-y-2">
        <label className="text-sm font-medium">Query Test (Hybrid: BM25 + kNN + RRF)</label>
        <div className="flex gap-2">
          <select
            value={searchMode}
            onChange={(e) => setSearchMode(e.target.value as SearchMode)}
            className="bg-background border border-border rounded-md px-2 py-2 text-sm focus:ring-2 focus:ring-primary/50 outline-none"
            title="Search mode"
          >
            <option value="hybrid">Hybrid</option>
            <option value="keyword">Keyword</option>
            <option value="vector">Vector</option>
          </select>
          <input
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runQuery()}
            placeholder="e.g. What is the Claim Check pattern?"
            className="flex-1 bg-background border border-border rounded-md px-3 py-2 text-sm focus:ring-2 focus:ring-primary/50 outline-none"
          />
          <button
            type="button"
            onClick={runQuery}
            disabled={loadingQuery || !queryText.trim()}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {loadingQuery ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Search
          </button>
        </div>
      </div>

      {queryResult && (
        <div className="space-y-2">
          {queryResult.message && (
            <div className="text-xs p-2 rounded bg-red-500/10 text-red-400">{queryResult.message}</div>
          )}
          {queryResult.hits.length === 0 && !queryResult.message && (
            <div className="text-xs text-muted-foreground p-2">No results found.</div>
          )}
          {queryResult.hits.map((hit, i) => (
            <div key={i} className="p-2 rounded border border-border bg-background text-xs space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-mono text-muted-foreground truncate max-w-[55%]">
                  {hit.uri?.split('/').pop() || hit.uri || '(unknown)'}
                </span>
                <span className="text-muted-foreground font-mono truncate max-w-[25%]" title={hit.chunkId || undefined}>
                  {hit.chunkId || ''}
                </span>
                <span className="text-green-400 font-mono">score={hit.score.toFixed(4)}</span>
              </div>
              <p className="text-muted-foreground line-clamp-2">{hit.text?.slice(0, 200)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
