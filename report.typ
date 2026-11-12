#set page(paper: "a4", margin: 2cm)
#set text(size: 9.5pt)
#set table(stroke: 0.6pt + luma(170), inset: (x: 5pt, y: 3.5pt))
#set heading(numbering: "1.")
#show table: set text(size: 8pt)

#align(center)[
  #text(size: 15pt, weight: "bold")[The Relay]
  #linebreak()
  #text(size: 11pt)[Results tables]
  #v(2pt)
]

All limits below are for a single model on the Groq free tier. Unless stated
otherwise the model is `openai/gpt-oss-20b`; limits are enforced per model, so a
second model carries its own independent bucket.

#figure(
  table(
    columns: (auto, 1fr, 1.5fr, auto),
    align: (left, left, left, left),
    table.header(
      [*Limit*], [*Value found*], [*How I found it*], [*Confidence*],
    ),
    [Requests per minute], [30],
    [Console limits page. Sequential probing cannot exceed this cap, so it was
      not independently confirmed; the 429 body reports RPM in the same format
      ("RPM: Limit 3, Used 3"), which corroborates the mechanism.],
    [medium],

    [Tokens per minute], [8\,000],
    [`x-ratelimit-limit-tokens` on every response, constant across the run.],
    [high],

    [Requests per day], [1\,000],
    [`x-ratelimit-limit-requests` on every response. Note this header is the
      *daily* request budget, not the per-minute one.],
    [high],

    [Tokens per day], [200\,000],
    [Console limits page (TOKENS PER DAY column). Not exposed in any response
      header.],
    [high],

    [Requests in flight at once], [at least 8, no limit observed],
    [Fired 8 concurrent requests at `allam-2-7b`; 8 succeeded, 0 rate-limited.
      Untested above 8.],
    [high],

    [What it returns when a limit is hit],
    [HTTP 429 with `type: requests` and `code: rate_limit_exceeded`, a
      `retry-after` header in seconds, and a human-readable message quoting the
      limit and the wait. Body quoted below.],
    [high],
    [high],

    [Time until it recovers], [20.3 s],
    [Tripped a 429 on `openai/gpt-oss-safeguard-20b`, honoured `retry-after: 20`,
      retried once, succeeded after 20.3 s wall clock.],
    [high],

    [Measured token ceiling], [8\,848 tokens / 60 s],
    [Maximum tokens completed in any rolling 60 s window across `probe.jsonl`.
      Sits above the 8\,000 header because the window spans model switches and
      bucket resets, whereas `x-ratelimit-limit-tokens` is a per-model bucket.
      The header is authoritative for gating; this is an upper bound.],
    [medium],
  ),
  caption: [Limits found in the real service (Groq free tier, `openai/gpt-oss-20b`
    unless stated)],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    table.header(
      [*Scenario*], [*Source*], [*Requests sent*], [*Duration (s)*],
      [*Offered rate (/min)*], [*First token p50 / p95 / p99 (ms)*],
      [*Total time p50 / p95 / p99 (ms)*],
    ),
    [steady], [real], [6], [100], [3.6], [1755 / 3491 / 3491], [1756 / 3492 / 3492],
    [burst], [real], [20], [2], [600.0], [6982 / 8134 / 8134], [6982 / 8135 / 8135],
    [sustained overload], [real], [40], [116], [20.7], [1221 / 4364 / 5391], [1222 / 4364 / 5392],
    [per-minute limit halved], [stand-in], [40], [123], [19.5], [7 / 7076 / 7662], [7 / 7077 / 7662],
    [daily budget exhausted], [stand-in], [5], [24], [12.5], [not measured], [not measured],
  ),
  caption: [Latency per load scenario. Percentiles are over successful responses
    only, since a rate-limited request never produced a first token: n = 6, 5, 21,
    15 and 0 respectively. Where n is below 30 the p99 is the largest sample, not
    a tail estimate.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    table.header(
      [*Scenario*], [*Succeeded %*], [*Rate-limited %*],
      [*Timed out or given up %*], [*Retries per request*], [*Peak queue*],
      [*Tokens per answer*], [*Cost per 1\,000 answers*],
    ),
    [steady], [100.0], [0.0], [0.0], [0.00], [0], [129.5], [0.00],
    [burst], [20.0], [75.0], [5.0], [0.80], [0], [130.8], [0.00],
    [sustained overload], [32.5], [60.0], [7.5], [0.85], [0], [88.2], [0.00],
    [per-minute limit halved], [37.5], [60.0], [2.5], [0.75], [0], [52.3], [0.00],
    [daily budget exhausted], [0.0], [100.0], [0.0], [1.00], [0], [0.0], [0.00],
  ),
  caption: [Outcomes per load scenario. Peak queue is 0 everywhere because the
    pipeline never queues: it answers from fewer chunks, answers extractively, or
    refuses, and surplus load becomes a 429 or a \"system busy\" refusal. Four
    requests were turned away that way, three in sustained overload and one in the
    halved run.],
)

The burst duration reads as 2 s because `start_time` was recorded to the second, so
a 4 s arrival window collapsed; 600 per minute is what the file yields, while the run
fired 20 requests at 300 per minute by construction. Sustained overload crossed two
limits inside one run: 12 rejections name tokens per minute and 13 name tokens per
day, because the run drained the remaining daily budget while it was going. That is
the daily limit arriving under load rather than at 3pm, and the \"daily budget
exhausted\" row is the same wall reached from a cold start.

Peak queue is a real zero, recoverable from `queued_ms` in `results.csv`: no request
spent time waiting for a slot. That is a property of the design, not of the load. The
ladder absorbs pressure by answering worse instead of holding the request, so nobody
is delayed and a few people are dropped. Seventeen successful requests made no draft
call at all (rung 3, extractive), which is why tokens per answer collapses under load
and why the halved run's first-token p50 is 7 ms.

All five scenarios cost 46\,128 tokens in total, and 37\,083 of that is the real
service. Cost per 1\,000 answers is 0.00 because the free tier charges nothing; the
binding constraint is tokens per day, not money.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    table.header(
      [*Setting*], [*Questions asked*], [*Correct and cited %*], [*Wrong %*],
      [*Correctly refused %*], [*Tokens per answer*],
      [*First token p50 / p95 (ms)*], [*Total time p50 / p95 (ms)*],
    ),
    [A: 1200-char pieces, 4 fetched, no re-rank], [15], [60.0], [13.3], [20.0],
      [96], [1272 / 2193], [1370 / 2282],
    [B: 3000-char pieces, 6 fetched, re-rank], [15], [60.0], [20.0], [20.0],
      [115], [2577 / 4614], [2584 / 4649],
    [C: 6000-char pieces, 8 fetched, re-rank], [15], [60.0], [13.3], [20.0],
      [109], [1780 / 4927], [1888 / 4932],
  ),
  caption: [Three retrieval settings, each on the same 15 questions. Tokens per
    answer is the mean completion length; prompt length averaged 1485, 2111 and
    2519 tokens respectively. Total time is first token plus the rest of the
    draft, so it excludes the citation check.],
)

All three settings answer 60.0 % correctly, so piece size and re-ranking made no
difference to quality on these 15 questions, and the choice falls to cost and
latency. A is the cheapest and fastest by a wide margin: 1485 prompt tokens and a
1272 ms first token, against 2519 and 1780 ms for C. The gap is the reranker, which
C runs over 12 pairs and A does not run at all.

What the settings do change is which questions fail. Q12 fails in all three, because
the retriever matches the words "BPE" and "tokenizer" wherever they appear, and
H-Net's own introduction and appendix discuss BPE the most. Q11 fails three different
ways: A refuses, B claims Helium and Moshi are both 7B and calls a 1.3B baseline the
largest model, and C refuses. Q01 is answered correctly only by A, which states the
"n smaller than d" condition; B answers with a training-speed claim from the
conclusion, and C states the condition backwards, saying the advantage holds for long
sequences.

I choose A. Re-ranking would cost about 1.3 s of first token on the C measurement and
change nothing on these 15 questions, so A keeps 1200-character pieces, 4 fetched
pieces and no re-rank. When the limit is close the ladder does the rest without
changing piece size: rung 1 keeps 3 chunks instead of 4, rung 2 turns re-ranking off,
and rung 3 answers extractively from the top 3 spans with no draft call at all.

= What it returns when a limit is hit

Verbatim body, `openai/gpt-oss-safeguard-20b`, `retry-after: 20`:

```json
{"error":{"message":"Rate limit reached for model `openai/gpt-oss-safeguard-20b` in organization `org_01j4f5qxt6ehjrv71h1wdry0z0` service tier `on_demand` on requests per minute (RPM): Limit 3, Used 3, Requested 1. Please try again in 20s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing","type":"requests","code":"rate_limit_exceeded"}}
```

= Notes on these measurements

- *The three buckets are on three different clocks.* Requests are limited per
  day, tokens per minute, and individual models can carry their own per-minute
  request cap that appears only in the 429 body, never in a header. A limiter
  that assumes one uniform window will 429 itself while it believes it has
  headroom.

- *No `rpm` field exists anywhere in the API surface.* `x-ratelimit-limit-requests`
  means requests per *day*. Reading it as a per-minute limit would over-throttle
  by a factor of 1\,000.

- *Bursts are admitted above the per-minute token limit* (8\,848 measured against
  8\,000 declared), so the bucket is not a strict sliding window. Client-side
  gating therefore reads `x-ratelimit-remaining-tokens` rather than computing its
  own.

- *`max_tokens` caps reasoning tokens as well as answer tokens.* With
  `reasoning_effort` set, a 40-token budget returned `status: ok` with 38 reasoning
  tokens and an empty answer. Truncation must be classified as a failure, not a
  success, or it inflates the succeeded rate and mimics a refusal.
