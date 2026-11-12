#set page(margin: (y: 0.6in, x: 0.8in))

// #set page(fill: black.lighten(10%), margin: (y: 0.6in, x: 0.8in))
// #set text(fill: white.darken(20%))

#show heading.where(level: 1): it => smallcaps(it)

#set par(justify: true)
// #show par: it => text(font: "Montserrat", it)
// #show list: it => text(font: "Montserrat", it)

#box(width: 100%)[
  #align(center)[
    #text(weight: "black", size: 16pt)[#smallcaps()[Vaani's MLE Assignment]] \
    Himanshu Sardana (#text()[#underline()[#link("himanshu.co")[himanshu.co]]])
  ]
]

#box(
  stroke: 1pt,
  width: 100%,
  inset: 8pt,
)[I have a web demo of the RAG app live here: #link("vaani-mle.himanshu.co")[vaani-mle.himanshu.co]]

= What I built and why
The brief required I build a RAG service over 3 research papers:

- "Attention Is All You Need", arXiv 1706.03762v7
- "Moshi: a speech-text foundation model for real-time dialogue", 2410.00037v1
- "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", 2507.07955v2

Every answer must come from one of these papers as and when applicable and should be refused in case of an irrelevant question

// #line(length: 100%)

To achieve this, I built a 4 step pipeline

1. *Figuring out the relevant paper*: The pipeline starts out by matching which paper the query targets via plain keyword matching and whether they answer demands prose (text) or a number (numeric value), this is done by matching terms such as "how many", "how much" etc. At first, I attempted using a small model `allam-2-7b` (had a generous limit on Groq) for classifying which paper the query was for, however keyword matching performed just as well in my experience and was faster, which is why I got rid of it. Once we have the relevant paper.

A minor (or major, depending on how you look at it :p) issue with this approach is that it often struggles on cross-paper queries. However this is not always the case.

2. *Retrieval*: Having identified the paper and question type, we then search across a BM25 index of 225 chunks across all 3 papers (chunk size: 1200, I tested with 1200, 3000 and 6000) and identify the top 12 chunks, the chunks belonging to the paper identified in step 1 have their score multiplied by 1.5, if the answer type was numeric, any chunks containing tables also have their scores multiplied by 1.5 (since numeric values are more likely to be in tables).
