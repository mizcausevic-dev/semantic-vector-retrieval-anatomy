# Semantic vector retrieval anatomy

Orthographic film of a query moving from tokens, through an HNSW cosine search, into a top-k context window. 1,920×1,080, 30 fps, 1,020 frames, 34.00 seconds.

The interactive preview is `index.html`. It imports `player.js`, which is also the drawing engine Remotion renders.

## Preview

Open `index.html` in a browser. GitHub Pages serves this same file from the `gh-pages` branch:

https://mizcausevic-dev.github.io/semantic-vector-retrieval-anatomy/

Playback pauses on the 45-degree graph view when the browser requests reduced motion.

## Studio and render

```bash
npm install
npm run dev
npm run render
```

`npm run render` writes `out/llm-vector-search-anatomy-Cursor.mp4`. That directory is gitignored. A still is `npm run still -- --frame=300 out/frame-300.png`.

## What the picture is

The drawing code builds 32-dimensional unit vectors and scores them with a real dot product. The on-screen register is labeled as 8 of a 768-dimensional embedding. Chunk titles, file names, and keys are diagram labels. They are not citations and they are not retrieved documents.

The picture leaves out quantization, metadata filters, hybrid lexical search, and a reranker.
