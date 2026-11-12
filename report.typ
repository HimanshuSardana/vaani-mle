#set page(paper: "a4", margin: (x: 1.2cm, y: 1.2cm), columns: 2)
#set text(size: 9.6pt)
#set par(leading: 0.55em, justify: true, spacing: 0.75em)
// #set heading(numbering: "1.")
#show heading: it => smallcaps()[#text(size: 10pt, it)]
#show heading: set block(above: 0.6em, below: 0.6em)
#set table(stroke: 1pt + black, inset: (x: 3pt, y: 3pt))
#show table: set text(size: 7.4pt)
#set figure(supplement: none)
#show figure.where(kind: table): set figure.caption(position: bottom)
#show figure.caption: set text(size: 7.6pt, style: "italic")

#place(scope: "parent", float: true, top + center)[#align(center)[#text(
    size: 13pt,
    weight: "bold",
  )[#smallcaps()[The Relay]] \ #text(
    size: 9.5pt,
  )[By Himanshu Sardana (#underline()[#link("himanshu.co")[himanshu.co])]]]]

#box(width: 100%, inset: 8pt, stroke: 1pt)[
  GitHub Repo:
  #underline()[#link("https://github.com/HimanshuSardana/vaani-mle")[https://github.com/HimanshuSardana/vaani-mle]]
  \
  *PS*: I also built a Next.js website as an interface for this RAG chatbot, I'd love for you
  guys to have a look at it at
  #underline()[#link("vaani-mle.himanshu.co")[vaani-mle.himanshu.co]]
]

= What I built and why

A retrieval service over three papers Attention Is All You Need
Moshi and Dynamic Chunking. Every response carries
a citation listing the paper, section and page, A question the papers do not answer/ is irrelevant is
refused.

The pipeline consists of four steps. First, match which paper the question targets and whether it wants a
prose answer or a number, achieved via plain keyword matching, for question types (prose
or number, I simply check for keywords such as "how much", "how many", etc.) I tried routing with `allam-2-7b` at
64 tokens and dropped it because name matching was just as accurate and faster. Second, keyword
search over 225 pieces returns 12 candidate chunks, chunks belonging to the possible
matched paper are scored 1.5 times higher, for numeric questions the chunks containing
tables are scored 1.5 times since tables are much more likely to contain a numeric answer. Third, one
draft call to `openai/gpt-oss-20b` capped at 320 tokens with low reasoning. Fourth, resolve
every citation against a retrieved passage and delete any that does not resolve.

Each request gets 8 seconds and there is no queue, so
the answer degrades until it fits. It keeps 3 passages instead of 4, then skips the
reordering pass, then stops calling the model and quotes the retrieved sentences with
citations attached, then refuses. The level is chosen from remaining time and remaining
token budget, so a busy system answers worse rather than answering late. Nobody waits.


= Groq's Limits and Stand-in

#figure(
  table(
    columns: (auto, auto, 1fr, 3.8em),
    align: (left, left, left, left),
    table.header([*Limit*], [*Found*], [*How*], [*Conf.*]),
    [Requests per minute], [30], [console page; 429 body quotes RPM], [medium],
    [Tokens per minute], [8\,000], [`x-ratelimit-limit-tokens`, constant], [high],
    [Requests per day], [1\,000], [`x-ratelimit-limit-requests`], [high],
    [Tokens per day], [200\,000], [console page, never in a header], [high],
    [Requests in flight], [at least 8], [8 concurrent succeeded, none limited], [high],
    [Reply at a limit], [HTTP 429], [`rate_limit_exceeded` plus `retry-after`], [high],
    [Time to recover], [20.3 s], [honoured `retry-after: 20`, retried once], [high],
  ),
  caption: [Limits found on the free tier for `openai/gpt-oss-20b`.],
)

Three buckets run on three clocks: requests per day, tokens per minute, and a per-model
request cap that appears only in a 429 body and never in a header. So the client reads
`x-ratelimit-remaining-tokens` instead of counting, because a rolling 60 s window completed
8\,848 tokens against an 8\,000 declaration and local arithmetic would have over-throttled.

The stand-in acts as a mock API, streams SSE, enforces rpm, tpm, rpd and tpd per model,
returns Groq-shaped 429 bodies, and can inject failures, slow successes and load slowdown.
The two limit rows below run on it with tokens per minute cut to 4\,000 and tokens per day
set to zero, shapes the real tier will not give me without spending a day of budget.
Time to first token is bootstrapped from measured samples in four input-length bands of 33,
5, 6 and 1 observations, with bands under 8 samples pooled at draw time, and a 70-token
template overhead that took tokenizer error from 0.5965 to 0.0001 below 400 tokens. Least
like the real service: it cannot exceed its own observed maximum, so slow tails are
reproduced rather than invented, and it echoes prompt text so it cannot answer at all,
which is why every quality number below comes from the real service.

= Results

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    table.header([*Scenario*], [*Source*], [*Sent*], [*Sec*], [*Offered*], [*First token*], [*Total*]),
    [steady], [real], [6], [100], [3.6], [1755 / 3491 / 3491], [1756 / 3492 / 3492],
    [burst], [real], [20], [2], [600.0], [6982 / 8134 / 8134], [6982 / 8135 / 8135],
    [overload], [real], [40], [116], [20.7], [1221 / 4364 / 5391], [1222 / 4364 / 5392],
    [halved], [stand-in], [40], [123], [19.5], [7 / 7076 / 7662], [7 / 7077 / 7662],
    [daily], [stand-in], [5], [24], [12.5], [not measured], [not measured],
  ),
  caption: [Latency, p50 / p95 / p99 in ms, over successful responses only: n = 6, 4, 13,
    15 and 0. Where n is under 30 the p99 is the maximum sample.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    table.header([*Scenario*], [*OK %*], [*429 %*], [*Given up %*], [*Retries*], [*Queue*], [*Tok/ans*], [*Cost*]),
    [steady], [100.0], [0.0], [0.0], [0.00], [0], [129.5], [0.00],
    [burst], [20.0], [75.0], [5.0], [0.80], [0], [130.8], [0.00],
    [overload], [32.5], [60.0], [7.5], [0.85], [0], [88.2], [0.00],
    [halved], [37.5], [60.0], [2.5], [0.75], [0], [52.3], [0.00],
    [daily], [0.0], [100.0], [0.0], [1.00], [0], [0.0], [0.00],
  ),
  caption: [Outcomes. Cost is per 1\,000 answers and is zero on the free tier; peak queue
    is recoverable from `queued_ms` and is a real zero.],
)

`results.csv` holds 186 rows: 75 graded answers and 111 load requests, each failure stating
which limit rejected it. Answer quality is 60.0 % correct on each of three retrieval
settings, 15 questions each, with 13.3 to 20.0 % wrong and 20.0 % correctly refused.
Failures across all rows: 49 rejected on tokens per minute, 19 on tokens per day, 3
refused at capacity, 2 past the deadline. Steady answered everything; burst answered 4 of
20 with `retry-after` between 6 and 17 s; overload crossed both limits inside one run, 12
rejections naming the minute and 13 naming the day. The number I could not have guessed:
the 429 that said `retry-after: 470` said "try again in 2.0025s" in its own body. Two
caveats on the tables. Burst reads 600 offered per minute because `start_time` was recorded
to the second when that row ran, so a 4 second arrival window collapsed; the run fired at
300 per minute by construction. And the 111 load rows carry an empty `answer_grade`, because
they measure latency and outcomes, not accuracy; the 75 graded rows are the quality run.

= Where the time goes

Burst Q03, 8\,135 ms end to end: keyword search 1.6 ms, cross-encoder reordering 463.6 ms
over 12 pairs, first token 7\,537 ms, generation 130 ms, citation check 0.4 ms. Waiting is
the 464 ms of ours that happens before the model is called, working is the 7\,537 ms inside
the call, and there is nothing to retry. So 93 % of
the wait is before the first token and 5.7 % is mine. Quality Q15 with two retries spent
8\,088 ms while its draft first token took 1\,840 ms, so 5.8 seconds of that request is
retry sleep, also before the first token, which is the worst place because nothing is on
screen yet. Every request writes per-stage milliseconds to `traces/`; Groq's own
`queue_time`, `prompt_time` and `completion_time` arrive in the response but are not copied
into the trace, so this split is client-side.

= When a limit is hit

At the minute limit the pipeline returns the 429 immediately with the wait the server
named, because a wait longer than the deadline cannot help; burst rejected 15 of 20 that
way while the first four were answered. The ladder thresholds are read from the same bucket
the header reports: under 2\,600 remaining tokens it stops reordering, under 1\,500 it keeps
three passages, under 700 it answers extractively, and under 300 it refuses. At the day limit every attempt fails, so it stops
trying and says the quota is spent; daily rejected 5 of 5 from the first request. Delayed:
nobody, peak queue is zero in every scenario. Dropped: 25 rejected, 3 refused at capacity,
2 past the deadline. Given a weaker answer: 14 requests answered extractively with no
model call, which is why tokens per answer falls to 88.2 under overload against 130.8 at
steady.

= Retries

At most two attempts, only on 429, timeout or 5xx, the wait taken from the response body,
and no wait longer than the time left on the deadline. Retrying made things worse once, in
the way that matters: an early version honoured `retry-after: 868` inside an 8 second
deadline, so one request slept 14.5 minutes and then timed out anyway, spending nothing and
producing nothing. The load runs repeat the lesson cheaply: burst spent 0.80 retries per
request and rescued nothing, since the four successes were through on their first attempt,
and overload spent 0.85 while leaving p95 at 4\,364 ms against p50 1\,221 ms.

= What the person sees

At 3pm with the day spent, the reply is: "The assistant has used its daily model quota and
cannot answer until it resets. Try again in about 60 seconds." That is a fix. The first
version reused the refusal sentence, so a spent quota read "Not stated in the provided
papers. Upstream rate_limited.", which tells a waiting person the library lacks the answer
when the library is fine. There are now four messages, one for the day limit, the minute
limit, a blown deadline and the at-capacity step, and a test asserts none of them contains
the refusal sentence.

= What I'm unsure about, what I left out, what I learned

Least certain: that a bootstrap over four bands, three of them sparse, imitates a
200-request scenario. If it is wrong, my stand-in rows understate slow tails and the
overload rows look better than the real service would. Also thin: 15 questions tie three
settings at 60 %, which cannot separate settings that differ by one answer. Deliberately not
built: an admission queue, which is why peak queue is zero, and runtime switching of piece
size, fixed at process start. New to me: `max_tokens` caps reasoning tokens as well as answer tokens, so a
40-token budget returned a successful call with 38 reasoning tokens and an empty answer,
which means truncation must count as failure or it inflates the success rate and mimics a
refusal.

= Retrieval: quality versus cost

#place(float: true, top, scope: "parent")[
  #figure(
    table(
      columns: (1fr, auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right, right),
      table.header(
        [*Setting*], [*Q*], [*Correct %*], [*Wrong %*], [*Refused %*], [*Tok/ans*], [*First token*], [*Total*]
      ),
      [A: 1200-char, 4 kept, no re-rank], [15], [60.0], [13.3], [20.0], [96], [1272 / 2193], [1370 / 2282],
      [B: 3000-char, 6 kept, re-rank], [15], [60.0], [20.0], [20.0], [115], [2577 / 4614], [2584 / 4649],
      [C: 6000-char, 8 kept, re-rank], [15], [60.0], [13.3], [20.0], [109], [1780 / 4927], [1888 / 4932],
    ),
    caption: [Three settings on the same 15 questions. p50 / p95 in ms. Correct means correct
      and cited.],
  )
]

All three tie at 60.0 %, so cost decides and the answer is A: 96 output tokens and
1\,272 ms to first token, against 115 and 2\,577 ms for B and 109 and 1\,780 ms for C.
Reordering costs about 1.3 s of first token here and changed nothing, because the right
passage was already in the top four on 9 of 12 questions. When a limit is close the ladder
keeps 3 passages instead of 4, then drops reordering, then answers extractively.

= How this library will mislead a naive retriever

Gold-passage recall is 0.848 at k of 4, 6 and 8, perfect on 9 of 12 questions, and the
three misses are three distinct traps. Questions naming no paper: "the parameter counts of
the three headline models" needs a passage from each, recall is 0 at every k, and it is
refused in A and C because the per-paper search never runs without a named paper.
Vocabulary that does not match the source: "which papers rely on BPE tokenization" fails in
all three settings, since one paper says byte-pair encoding and another discusses it in an
appendix, so the answer cites H-Net's appendix instead of the Transformer. Long pieces
burying the sentence: "how much faster is self-attention than recurrent layers" is answered
only with 1200-character pieces, because at 6000 the operative sentence sits past the
1\,000-character cut applied to each passage.

= Your feedback on Vaani
I did tinker around with #underline()[#link("vaani.dev")[vaani.dev]] a bit to get a feel for what you guys do

= How I worked with the assistant
My entire agent session transcript is available at: #underline()[#link("https://pi.dev/session/#088e20bd4a25ab3a0f78039e9a34d2ad")[https://pi.dev/session/#088e20bd4a25ab3a0f78039e9a34d2ad]]

= Gallery
Padding this extra space with images of the Next.js site I built, cuz why not :p (pls
visit: #underline()[#link("vaani-mle.himanshu.co")[vaani-mle.himanshu.co]])


#image("assets/1.png", width: 100%)
#image("assets/2.png", width: 100%)
#image("assets/3.png", width: 100%)
#image("assets/4.png", width: 100%)
