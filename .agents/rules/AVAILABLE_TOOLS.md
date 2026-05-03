# Available MCP Tools
> All agents must refer to this file to know what tools are available.
> Do not assume a tool exists if it is not listed here.

## Tavily
- `tavily_search` — web search
- `tavily_extract` — extract content from a URL
- `tavily_research` — deep multi-source research (token-heavy, use sparingly)

## ArXiv
- `search_papers` — search ArXiv
- `get_abstract` — fetch abstract without downloading
- `download_paper` — download full paper
- `read_paper` — read a downloaded paper
- `citation_graph` — citations via Semantic Scholar
- `watch_topic` / `check_alerts` — monitor new papers

## Semantic Scholar
- `search_papers` — paper discovery
- `get_paper` — full metadata for a specific paper
- `get_paper_citations` — who cited this paper
- `get_paper_references` — what this paper cites
- `get_paper_fulltext` — download and read as markdown
- `search_papers_match` — find paper by title
- `get_recommendations_for_paper` — related papers
- `bulk_search_papers` — broad literature sweep
- `search_snippets` — find a concept across many papers
- `batch_get_papers` — fetch multiple papers at once

## GitHub
- `search_repositories` — find repos
- `search_code` — search code across GitHub
- `get_file_contents` — read a specific file
- `list_issues` — read issues

## Fetch
- `fetch` — retrieve any URL as markdown

## Sequential Thinking
- `sequentialthinking` — structured step-by-step reasoning
  (use for complex multi-step decisions before producing output)

## Graphify
- query the live project knowledge graph
  (use to understand project structure before making code changes)
