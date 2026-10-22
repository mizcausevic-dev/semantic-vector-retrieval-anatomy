Title: How semantic vector search retrieves a context window

Subtitle: A 34-second orthographic film of the objects between a question and the chunks a generator reads, with the cosine math left in.

Suggested Medium tags: Artificial Intelligence, Machine Learning, Information Retrieval, Software Engineering, LLM

Canonical links, verified 22 October 2026:

- Repository: https://github.com/mizcausevic-dev/semantic-vector-retrieval-anatomy
- Live player: https://mizcausevic-dev.github.io/semantic-vector-retrieval-anatomy/

---

A semantic retrieval system does not look up the words in the question. It turns the question into a vector, compares that vector with vectors already stored for chunks of source text, and copies the winners into the prompt the generator is about to read. The index that makes the comparison cheap at large scale is usually a graph. The score is usually a cosine. The thing that lands in the prompt is not the winning vector. It is the text the vector points at.

I built a 34-second film of that path and put the player on GitHub Pages. It runs for 1,020 frames at 30 fps, 1,920 by 1,080. The camera pitch at frame 300 is 45.0 degrees, because that key is `Math.PI / 4` and the frame lands on the key. The graph has 108 nodes and 266 edges. Every score printed on a card is a real dot product of two unit vectors in the drawing code. The film is a diagram with honest arithmetic. It is not a benchmark, and it is not a trace of hnswlib, Faiss, or any production database.

## What you are looking at

The question on screen is "where does the model store meaning". Six tokens move into a block labeled with the ordinary transformer stages, TOK, MHA, FFN, and POOL. Eight signed components then collapse into a magenta query point. The label on that register says "q 8 of 768". That label is a scale marker. The vectors the drawing code actually builds are 32-dimensional. I show eight numbers because a 768-wide register would be a texture, not a readable instrument.

From there the query drops onto a three-layer graph. Layer two sits at y = 1.85 and holds 8 nodes. Layer one sits on the origin plane and holds 28. Layer zero sits at y = -1.95 and holds 72. Edges inside a layer connect each node to its nearest neighbors in the ground plane: 2, 3, and 4 neighbors on layers two, one, and zero. Each upper node also has one link down to the nearest node on the layer below. That is the picture's HNSW, Hierarchical Navigable Small World: a sparse top, a dense floor, and a way to descend. It is the idea from Malkov and Yashunin's IEEE TPAMI paper, drawn small enough to see. It is not their insertion algorithm. There is no `M`, no `efConstruction`, and no `efSearch` in this file.

The clock is the frame number. Nothing in the drawing depends on the wall clock, so frame 640 is the same picture in Remotion, in the browser, and on the Pages deployment.

| Frame | Time | What the camera is doing |
| --- | --- | --- |
| 0 to 150 | 0.0 to 5.0s | Tokens travel into the block and the register collapses. |
| 151 to 300 | 5.0 to 10.0s | Pitch rises to 45 degrees. The query drops onto the top layer. |
| 301 to 510 | 10.0 to 17.0s | A wavefront scores nodes. Near, high-cosine nodes stay mint. Far nodes fall back to the grid color. |
| 511 to 750 | 17.0 to 25.0s | One node opens onto chunk text and a pointer: document id, chunk key, span, cosine. |
| 751 to 1020 | 25.0 to 34.0s | The camera spirals out. Four chunk cards fill a context window. |

## A query is a direction

An embedding model maps a string to a list of floats. After normalization, that list is a direction on a sphere. Two chunks about the same fact tend to point the same way. Two chunks about unrelated facts tend not to. That tendency is the whole bet. It is empirical, it fails in known ways, and it is still the reason retrieval systems spend their time on geometry instead of keyword overlap.

In the film the query direction is fixed. It is a 32-component vector filled from a sine hash and then normalized. Each stored node gets its own direction from the same recipe: a proximity weight pulls it toward the query, and a seeded noise term pulls it away. Nodes drawn near a basin at `(1.15, -0.72)` on the ground plane become neighbors in vector space. Nodes drawn far away do not. I did that so the picture would have a visible near set and a visible far set. A production embedder does not place vectors with a basin function. It places them by running the model.

The cosine between the query `q` and a stored vector `v` is:

```
cos = (q · v) / (|q| |v|)
```

That line is on screen from frame 300 to frame 560. Because both sides are normalized, the denominator is 1 and the cosine is the dot product. Magnitude never gets a vote. A long vector and a short vector that point the same way score the same, which is what you want when the length is an accident of the encoder rather than a property of the chunk.

Across the 108 nodes, the cosines run from -0.461 to 0.990. The median is 0.138. Most of the graph is unrelated to the query. A handful of nodes sit near 1. That distribution is a property of this constructed world. Do not quote it as a recall figure for any real corpus.

## The graph is there so you do not touch every row

At 108 vectors, the honest search is a loop. Score all of them, sort, take four. Approximate nearest neighbor search exists for the corpora where that loop stops being the cheap part: millions of chunks, a latency budget measured in milliseconds, and a willingness to miss a neighbor sometimes.

HNSW pays for that with a graph built at insert time. A new vector is assigned to a layer, often with a geometric distribution so most vectors live only on the bottom. The upper layers are thin. Search starts at an entry point on the top layer, walks to the neighbor closest to the query, and drops a layer when the local neighborhood stops improving. The bottom layer is where the candidate list becomes precise. The parameters that matter in production are how many links a node may keep, and how wide a beam the builder and the searcher are willing to explore. Wider beams cost more and miss less. That trade is the product decision. The film does not run it.

What the film does run is the score. Node color is a function of cosine. Before the wave arrives, nodes rest in mint, darker when their cosine is low. As the wave passes, a node interpolates toward mint if its cosine is high and toward the slate grid color if it is not. I capped the wave at a radius of 9.1 world units so the ring would not paint the frame shut. The ring is a teaching device. A real HNSW search does not expand a circle. It walks edges. If you take one sentence out of this piece, take that one.

## The number on the card is real. The choice of card is staged.

At frame 640 the open node shows a cosine of 0.989. The three cards beside it show 0.960, 0.926, and 0.892. Those are `toFixed(3)` of these dot products:

- 0.9889, chunk key `hnsw#04`, titled "Base layer"
- 0.9595, `sim#11`, "Cosine score"
- 0.9257, `ptr#02`, "Source pointer"
- 0.8921, `ctx#01`, "Context assembly"

I placed those four nodes by hand on layer zero, around the basin, and then sorted them by cosine. The sort is real. The claim "these are the top four in the graph" would be false. I checked. The highest cosine in the graph is 0.9895, on a layer-one node that is not in the cluster. A layer-two node scores 0.9876, also outside the cluster. Both sit on the basin, so the same proximity function that makes the cluster hot makes the column nodes hot too. Two nodes outside the labeled set outrank the weakest labeled card.

I am leaving the picture that way and writing it down, rather than quietly swapping the cards after the fact. The instrument is the cosine. The staging is the camera. A retrieval engine that cannot tell those apart will ship a demo as a metric.

The titles, file names, and spans are labels I wrote for the diagram. `notes/hnsw.md` is not a document I retrieved. There is no corpus behind the pointer. The pointer fields are the right fields: a document id, a chunk key, a character span, and the score that selected the row. They are empty of any real source.

## The generator never sees the vector

This is the step diagrams usually skip, and it is the step that decides answer quality.

The winning object in the index is a vector plus a key. The object in the prompt is a string. Something has to resolve the key. In a normal pipeline that something is a payload store, often the same database, sometimes a separate document store. The lookup returns the chunk text and whatever metadata the prompt template knows how to place. The generator then reads a prompt that contains the question and those chunks. It does not read the HNSW. It does not know the cosine, unless you print the cosine into the prompt on purpose.

The last act of the film is that copy. Four cards travel from their nodes into a panel labeled "Context window", with `k = 4` and `metric = cosine`. The panel is the only part of the machine the language model would see. Everything before it is index machinery.

k is a budget. Four chunks is a small budget, chosen so the panel can be read at 1080p. Raise k and you raise the chance that the needed fact is present. You also spend context, and you invite the model to blend a weak chunk into the answer. The right k is an evaluation result on your questions and your corpus. It is not a property of the graph.

## What has to be true before any of this is worth running

The film starts with the query already a well-formed question and the corpus already a tidy set of vectors. Production retrieval fails earlier than that.

Chunking decides what a vector is allowed to mean. A 200-token window and a 2,000-token window of the same manual do not point to the same place, and neither is "the document." Overlap, headings, and tables change the neighborhood. If the chunk boundary cuts the sentence that contains the answer, no index will rescue it. The best cosine in the index will be the cosine of a chunk that almost said it.

Metadata filters run before or during the walk: tenant, date, language, access. A global nearest neighbor that the caller is not allowed to read is not a result. Hybrid search exists because cosine is blind to rare exact strings. Identifiers, error codes, SKUs, and names often want a lexical score beside the vector score. A reranker, usually a cross-encoder that sees the query and the chunk together, exists because the bi-encoder's cosine is a fast guess. The guess is good enough to propose fifty candidates and not good enough to order the top five on a question where the wording is awkward.

Quantization is the other omission. Product quantization and binary quantization cut memory so the bottom layer fits. They also perturb the score. A diagram that draws full-precision unit vectors and a system that scores 4-bit codes are not reporting the same cosine. If you evaluate one and ship the other, the regression will look like a prompt bug.

## How to read the player, and how to render the film

The live player is the Pages site. GitHub reports the site built from the `gh-pages` branch at `/`. I fetched the page and `player.js` over HTTP. Both returned 200. In a browser I jumped to the sweep chapter, scrubbed to frame 640, and paused. The payload card was on screen with cosine 0.989, the play control read Play, and the chapter buttons wrapped cleanly at a 390-pixel width.

The same `drawFrame` function is what Remotion renders. From a checkout:

```
npm install
npm run dev
npm run still -- --frame=300 out/frame-300.png
npm run render
```

`npm run render` writes `out/llm-vector-search-anatomy-Cursor.mp4`. That directory is gitignored. The repository does not contain the video. The Pages preview is the canvas, which is the artifact that can update without a 30-megabyte binary. On this machine, `tsc --noEmit` completed with exit code 0 against Remotion 4.0.537 and React 19.3.1, and stills at frames 70, 300, 430, 640, and 1000 rendered. Frame 300 shows pitch 45.0 degrees. A pixel sample of that still found mint nodes and no black nodes. An earlier version of the color mixer parsed its own `rgb()` output as hex, produced NaN, and painted the graph black. The stills are there to keep that regression from coming back unnoticed.

Reduced motion pauses the player on the 45-degree view instead of autoplaying. The drawing itself is still a moving film if you press Play.

## Questions worth answering before you adopt the picture

**Is the model storing meaning in this graph?**
The graph stores vectors computed from chunks, plus the keys needed to load those chunks. "Meaning" is a claim about the embedding model's geometry. The index does not add meaning. It makes a particular neighborhood retrievable.

**Why not just use keyword search?**
Use both when the query contains a string that must match exactly. Vector search is the right tool when the question and the source use different words for the same fact. It is the wrong tool when you need the row whose identifier was typed into the question.

**Does a higher cosine mean a better answer?**
It means a closer direction. Closeness tracks usefulness only to the extent the embedder and the chunking put the useful text near the question. A chunk can score 0.99 and still be the wrong tenant, the wrong date, or a paraphrase that omits the number the user asked for. That is what filters and rerankers are for.

**Why does the film refuse to call the four cards the top four?**
Because they are not. They are the four nodes I placed in the hero cluster, ordered by a real cosine. The global top of this graph includes two nodes the camera does not open. I would rather the article contradict the poster than the poster invent a ranking.

**Can I drop this drawing into a pitch as a measured latency or recall number?**
No. 108 nodes, a scripted descent, and a constructed basin have nothing to say about your p50, your recall at 10, or your cost per thousand queries. If you need those numbers, run them on your corpus and publish the corpus policy with the number.

## What I would measure next, on a real index

Take a few hundred questions you already know the answer to. Freeze the chunking. Freeze the embedder. Report recall of the known chunk at k, the fraction of queries where a filter removed the true neighbor, and the change in answer accuracy when a reranker reorders the same candidates. Keep the cosine distribution. If the true chunk is routinely outside the top 50, no prompt edit will fix it. If it is routinely inside the top 5 and the answer is still wrong, you have left the index and entered the generator.

The film is the map of that argument. The index is the walk. The score is a cosine. The prompt receives text. Everything else is a parameter you owe your future self a measurement for.

---

Miz Causevic is the author of the film and this draft. He works at Kinetic Gain in Cambridge, Massachusetts. The public bio on his GitHub account, mizcausevic-dev, describes engineering, platform architecture, and B2B SaaS work, and about 30 years across IBM, CyberArk, and Alteryx. LinkedIn: https://www.linkedin.com/in/mirzacausevic/

The player and the Remotion composition share one drawing file, `player.js`. Chunk titles in the film are diagram labels. They are not citations and they are not retrieved documents.
