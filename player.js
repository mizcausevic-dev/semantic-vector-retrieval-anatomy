/**
 * Semantic vector retrieval anatomy.
 * 1,020 frames at 30 fps. Deterministic: the frame number is the only clock.
 *
 * 0-150    tokens compact into a query embedding
 * 151-300  camera tilts to 45 degrees and the query drops onto HNSW
 * 301-510  cosine wavefront, near nodes hold, far nodes recede
 * 511-750  winning node opens onto chunk text and a document pointer
 * 751-1020 spiral zoom-out, top-k streams into the context window
 */

export const FPS = 30;
export const DURATION = 1020;

const COLOR = {
  ink: '#0A0D14',
  grid: '#232B3E',
  gridMajor: '#31415C',
  magenta: '#FF007F',
  mint: '#00F5D4',
  amber: '#FFB703',
  bone: '#E6E2D8',
  muted: '#93A0B5',
  dim: '#66758C',
  panel: '#0C1018',
  panelStroke: '#31415C',
};

const TOKENS = ['where', 'does', 'the', 'model', 'store', 'meaning'];
const ACTS = [
  {id: '01', name: 'TOKEN TO VECTOR', line: 'Tokens compact into a query embedding.'},
  {id: '02', name: 'QUERY DROP', line: 'The query vector drops onto the HNSW entry layer.'},
  {id: '03', name: 'COSINE SWEEP', line: 'Rays score neighbors by cosine. Near nodes hold. Far nodes recede.'},
  {id: '04', name: 'PAYLOAD', line: 'The hit opens onto chunk text, metadata keys, and a document pointer.'},
  {id: '05', name: 'TOP-K', line: 'Top-k chunks stream into the synthesis context window.'},
];
const CHUNKS = [
  {
    title: 'Base layer',
    body: 'Every stored embedding sits on layer zero. The layers above keep long-range links so the walk starts coarse.',
    doc: 'notes/hnsw.md',
    key: 'hnsw#04',
    span: 'L0-L2',
  },
  {
    title: 'Cosine score',
    body: 'The query and a stored vector are scored by the cosine of the angle between them. Magnitude is divided out.',
    doc: 'notes/metric.md',
    key: 'sim#11',
    span: 'cos',
  },
  {
    title: 'Source pointer',
    body: 'The hit does not hold the paragraph. It points at a chunk key, a document id, and a character span.',
    doc: 'corpus/retrieval.md',
    key: 'ptr#02',
    span: '128-196',
  },
  {
    title: 'Context assembly',
    body: 'The top four chunks are copied into the prompt, ahead of the question, for the generator to read.',
    doc: 'notes/prompt.md',
    key: 'ctx#01',
    span: 'k = 4',
  },
];

const SCENE = {pipeY: 5.55, blockX: -0.15, stackX: 3.65, tokenLeft: -11.3};
const BASIN = {x: 1.15, z: -0.72};
const DIM = 32;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoother = (t) => {
  const x = clamp(t, 0, 1);
  return x * x * x * (x * (x * 6 - 15) + 10);
};
const range = (frame, a, b) => smoother((frame - a) / (b - a));
const lerp3 = (a, b, t) => ({x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), z: lerp(a.z, b.z, t)});

const RGB = new Map();
const parseColor = (input) => {
  const cached = RGB.get(input);
  if (cached) return cached;
  let rgb;
  if (input.startsWith('#')) {
    const n = parseInt(input.slice(1), 16);
    rgb = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  } else {
    const match = input.match(/(\d+),\s*(\d+),\s*(\d+)/);
    rgb = match ? [Number(match[1]), Number(match[2]), Number(match[3])] : [0, 0, 0];
  }
  RGB.set(input, rgb);
  return rgb;
};
const rgba = (hex, a) => {
  const [r, g, b] = parseColor(hex);
  return `rgba(${r}, ${g}, ${b}, ${clamp(a, 0, 1)})`;
};
const mix = (a, b, t) => {
  const u = clamp(t, 0, 1);
  const A = parseColor(a);
  const B = parseColor(b);
  const r = Math.round(A[0] + (B[0] - A[0]) * u);
  const g = Math.round(A[1] + (B[1] - A[1]) * u);
  const bl = Math.round(A[2] + (B[2] - A[2]) * u);
  return `rgb(${r}, ${g}, ${bl})`;
};

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
const normalize = (v) => {
  const n = Math.sqrt(dot(v, v)) || 1;
  return v.map((x) => x / n);
};
const mulberry32 = (seed) => {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
const sample = (keys, frame) => {
  if (frame <= keys[0].f) return keys[0].v;
  const last = keys[keys.length - 1];
  if (frame >= last.f) return last.v;
  let i = 0;
  while (keys[i + 1].f < frame) i++;
  const a = keys[i];
  const b = keys[i + 1];
  const prev = keys[Math.max(0, i - 1)];
  const next = keys[Math.min(keys.length - 1, i + 2)];
  const span = b.f - a.f;
  const t = (frame - a.f) / span;
  const m0 = ((b.v - prev.v) / (b.f - prev.f)) * span;
  const m1 = ((next.v - a.v) / (next.f - a.f)) * span;
  const t2 = t * t;
  const t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * a.v + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * b.v + (t3 - t2) * m1;
};

const frac = (n) => n - Math.floor(n);
const QUERY = normalize(
  Array.from({length: DIM}, (_, i) => {
    const raw = Math.sin((i + 1) * 12.9898) * 43758.5453;
    return frac(Math.abs(raw)) * 2 - 1;
  }),
);

const hypot2 = (ax, az, bx, bz) => {
  const dx = ax - bx;
  const dz = az - bz;
  return dx * dx + dz * dz;
};
const proxAt = (x, z) => Math.exp(-hypot2(x, z, BASIN.x, BASIN.z) / (2 * 2.2 * 2.2));
const embed = (prox, seed) => {
  const rng = mulberry32(seed);
  return normalize(
    QUERY.map((q) => {
      const noise = rng() * 2 - 1;
      return q * prox + noise * (1 - prox) + (rng() * 2 - 1) * 0.045;
    }),
  );
};

const GOLDEN = Math.PI * (3 - Math.sqrt(5));
const disk = (count, radius, seed) => {
  const pts = [];
  let i = 0;
  while (pts.length < count && i < count * 12) {
    const r = radius * Math.sqrt((i + 0.5) / (count * 1.35));
    const a = i * GOLDEN + seed;
    const x = Math.cos(a) * r;
    const z = Math.sin(a) * r;
    i++;
    if (r > radius) continue;
    if (hypot2(x, z, BASIN.x, BASIN.z) < 2.05 * 2.05) continue;
    pts.push({x, z});
  }
  return pts;
};

const buildWorld = () => {
  const drafts = [];
  disk(9, 3.45, 2.2).forEach((p) => drafts.push({layer: 2, y: 1.85, column: false, cluster: false, ...p}));
  drafts.push({layer: 2, y: 1.85, column: true, cluster: false, x: BASIN.x, z: BASIN.z});
  disk(27, 5.45, 5.7).forEach((p) => drafts.push({layer: 1, y: 0, column: false, cluster: false, ...p}));
  drafts.push({layer: 1, y: 0, column: true, cluster: false, x: BASIN.x + 0.16, z: BASIN.z - 0.1});
  disk(68, 7.45, 8.4).forEach((p) => drafts.push({layer: 0, y: -1.95, column: false, cluster: false, ...p}));
  [
    {x: BASIN.x, z: BASIN.z},
    {x: BASIN.x + 0.98, z: BASIN.z + 0.32},
    {x: BASIN.x - 0.7, z: BASIN.z - 0.58},
    {x: BASIN.x + 0.4, z: BASIN.z - 1.08},
  ].forEach((p, i) => drafts.push({layer: 0, y: -1.95, column: i === 0, cluster: true, ...p}));

  const nodes = drafts.map((d, id) => {
    const vec = embed(proxAt(d.x, d.z), 4000 + id * 17);
    return {
      id,
      layer: d.layer,
      pos: {x: d.x, y: d.y, z: d.z},
      vec,
      cosine: dot(QUERY, vec),
      column: d.column,
      cluster: d.cluster,
    };
  });
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const cluster = nodes.filter((n) => n.cluster).sort((a, b) => b.cosine - a.cosine);
  const topK = cluster.map((node, rank) => ({...CHUNKS[rank], nodeId: node.id, cosine: node.cosine, rank}));
  const entry = nodes.find((n) => n.layer === 2 && n.column);
  const l1 = nodes.find((n) => n.layer === 1 && n.column);
  const hero = nodes.find((n) => n.id === topK[0].nodeId);
  const side = nodes.find((n) => n.id === topK[2].nodeId);

  const edges = [];
  const seen = new Set();
  const link = (a, b, layer, vertical) => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = `${lo}:${hi}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({a: lo, b: hi, layer, vertical});
  };
  for (const layer of [2, 1, 0]) {
    const k = layer === 2 ? 2 : layer === 1 ? 3 : 4;
    const group = nodes.filter((n) => n.layer === layer);
    for (const n of group) {
      group
        .filter((m) => m.id !== n.id)
        .map((m) => ({id: m.id, d: hypot2(n.pos.x, n.pos.z, m.pos.x, m.pos.z)}))
        .sort((a, b) => a.d - b.d)
        .slice(0, k)
        .forEach((m) => link(n.id, m.id, layer, false));
    }
  }
  for (const upper of [2, 1]) {
    const ups = nodes.filter((n) => n.layer === upper);
    const downs = nodes.filter((n) => n.layer === upper - 1);
    for (const n of ups) {
      let best = downs[0];
      let bestD = Infinity;
      for (const m of downs) {
        const d = hypot2(n.pos.x, n.pos.z, m.pos.x, m.pos.z);
        if (d < bestD) {
          bestD = d;
          best = m;
        }
      }
      link(n.id, best.id, upper, true);
    }
  }
  return {
    nodes,
    edges,
    byId,
    entry: entry.pos,
    hero: hero.pos,
    path: [entry.pos, l1.pos, side.pos, hero.pos],
    topK,
  };
};

let worldCache = null;
const getWorld = () => {
  if (!worldCache) worldCache = buildWorld();
  return worldCache;
};

const cameraAt = (frame, world) => {
  const e = world.entry;
  const h = world.hero;
  const keys = [
    {f: 0, yaw: -0.2, pitch: 0.08, roll: 0, zoom: 62, x: -3.5, y: SCENE.pipeY, z: 0},
    {f: 85, yaw: -0.04, pitch: 0.1, roll: 0, zoom: 66, x: -2.1, y: SCENE.pipeY, z: 0},
    {f: 150, yaw: 0.12, pitch: 0.15, roll: 0, zoom: 70, x: 0.55, y: SCENE.pipeY - 0.05, z: 0},
    {f: 226, yaw: 0.34, pitch: 0.46, roll: 0, zoom: 76, x: e.x * 0.45 + 0.8, y: 2.5, z: e.z * 0.35},
    {f: 300, yaw: 0.52, pitch: Math.PI / 4, roll: 0, zoom: 78, x: e.x, y: -0.05, z: e.z},
    {f: 410, yaw: 0.7, pitch: 0.8, roll: 0, zoom: 86, x: (e.x + h.x) / 2, y: -0.45, z: (e.z + h.z) / 2},
    {f: 510, yaw: 0.86, pitch: 0.86, roll: 0, zoom: 102, x: h.x, y: h.y + 0.15, z: h.z},
    {f: 650, yaw: 1.02, pitch: 0.98, roll: 0, zoom: 168, x: h.x, y: h.y, z: h.z},
    {f: 750, yaw: 1.1, pitch: 1.02, roll: 0, zoom: 196, x: h.x, y: h.y, z: h.z},
    {f: 860, yaw: 1.1 + Math.PI * 0.72, pitch: 0.78, roll: 0.04, zoom: 86, x: 0.15, y: -0.25, z: 0},
    {f: 960, yaw: 1.1 + Math.PI * 1.15, pitch: 0.64, roll: 0.02, zoom: 54, x: -0.35, y: -0.35, z: 0},
    {f: 1020, yaw: 1.1 + Math.PI * 1.28, pitch: 0.58, roll: 0, zoom: 48, x: -0.45, y: -0.3, z: 0},
  ];
  const field = (name) => sample(keys.map((k) => ({f: k.f, v: k[name]})), frame);
  return {
    yaw: field('yaw'),
    pitch: clamp(field('pitch'), 0.04, 1.2),
    roll: clamp(field('roll'), -0.08, 0.08),
    zoom: clamp(field('zoom'), 34, 220),
    target: {x: field('x'), y: field('y'), z: field('z')},
  };
};

const layoutAt = (frame, w, h) => {
  const t = smoother(clamp((frame - 790) / 150, 0, 1));
  return {w, h, biasX: -t * w * 0.11};
};

const rotateToCamera = (p, cam) => {
  const cy = Math.cos(-cam.yaw);
  const sy = Math.sin(-cam.yaw);
  const x1 = p.x * cy - p.z * sy;
  const z1 = p.x * sy + p.z * cy;
  const cp = Math.cos(-cam.pitch);
  const sp = Math.sin(-cam.pitch);
  const y2 = p.y * cp - z1 * sp;
  const z2 = p.y * sp + z1 * cp;
  const cr = Math.cos(cam.roll);
  const sr = Math.sin(cam.roll);
  return {x: x1 * cr - y2 * sr, y: x1 * sr + y2 * cr, z: z2};
};
const project = (p, cam, layout) => {
  const r = rotateToCamera({x: p.x - cam.target.x, y: p.y - cam.target.y, z: p.z - cam.target.z}, cam);
  return {x: layout.w / 2 + layout.biasX + r.x * cam.zoom, y: layout.h / 2 - r.y * cam.zoom, z: r.z};
};

const actIndex = (frame) => {
  if (frame <= 150) return 0;
  if (frame <= 300) return 1;
  if (frame <= 510) return 2;
  if (frame <= 750) return 3;
  return 4;
};
const formatTime = (frame) => {
  const total = frame / 30;
  const m = Math.floor(total / 60);
  const sec = total - m * 60;
  const whole = Math.floor(sec);
  const fracPart = Math.round((sec - whole) * 100);
  const carry = fracPart === 100;
  return `${String(m).padStart(2, '0')}:${String(carry ? whole + 1 : whole).padStart(2, '0')}.${String(carry ? 0 : fracPart).padStart(2, '0')}`;
};
const signed = (n, digits = 3) => `${n < 0 ? '-' : '+'}${Math.abs(n).toFixed(digits)}`;
const mono = (px, weight = 400) => `${weight} ${px}px "JetBrains Mono", ui-monospace, monospace`;
const display = (px, weight = 500) => `${weight} ${px}px "Space Grotesk", sans-serif`;

const roundRect = (ctx, x, y, w, h, r) => {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
};
const wrap = (ctx, text, maxWidth) => {
  const words = text.split(' ');
  const lines = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else line = test;
  }
  if (line) lines.push(line);
  return lines;
};

const queryPosition = (frame, world) => {
  const stack = {x: SCENE.stackX, y: SCENE.pipeY, z: 0};
  const mid = {
    x: lerp(SCENE.stackX, world.entry.x, 0.42),
    y: lerp(SCENE.pipeY, world.entry.y, 0.48),
    z: world.entry.z * 0.3,
  };
  if (frame <= 150) return stack;
  if (frame < 222) return lerp3(stack, mid, smoother((frame - 150) / 72));
  if (frame < 300) return lerp3(mid, world.entry, smoother((frame - 222) / 78));
  return world.entry;
};
const waveRadius = (frame) => {
  if (frame < 301) return 0;
  if (frame >= 488) return 9.1;
  return lerp(0.15, 9.1, smoother((frame - 301) / 187));
};
const hotness = (cosine) => clamp((cosine - 0.2) / 0.72, 0, 1);
const reachedAmount = (node, frame, origin) => {
  if (frame < 301) return 0;
  const dist = Math.sqrt(hypot2(node.pos.x, node.pos.z, origin.x, origin.z));
  return clamp((waveRadius(frame) - dist) / 0.65, 0, 1);
};
const nodeEnergy = (node, frame, origin) => {
  const hot = hotness(node.cosine);
  const reached = reachedAmount(node, frame, origin);
  const rest = 0.26 + 0.4 * hot;
  return frame < 301 ? rest : lerp(rest, hot, reached);
};
const heroOpen = (frame) => range(frame, 526, 578) * (1 - range(frame, 718, 768));

const drawQuad = (ctx, cam, layout, center, hw, hh, stroke, fill, alpha, lineWidth = 1) => {
  if (alpha <= 0.01) return;
  const corners = [
    {x: center.x - hw, y: center.y - hh, z: center.z},
    {x: center.x + hw, y: center.y - hh, z: center.z},
    {x: center.x + hw, y: center.y + hh, z: center.z},
    {x: center.x - hw, y: center.y + hh, z: center.z},
  ].map((p) => project(p, cam, layout));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.restore();
};

const plate = (ctx, x, y, w, h, accent) => {
  ctx.fillStyle = COLOR.panel;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = accent;
  ctx.fillRect(x + 1, y + 10, 3, h - 20);
};

const drawScreenGrid = (ctx, w, h) => {
  const step = 64;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLOR.grid;
  ctx.beginPath();
  for (let x = 0; x <= w; x += step) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
  }
  for (let y = 0; y <= h; y += step) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
  }
  ctx.stroke();
  ctx.strokeStyle = COLOR.gridMajor;
  ctx.globalAlpha = 0.55;
  ctx.beginPath();
  for (let x = 0; x <= w; x += step * 4) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
  }
  for (let y = 0; y <= h; y += step * 4) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
  }
  ctx.stroke();
  ctx.restore();
};

const drawPipeline = (ctx, frame, cam, layout) => {
  const alpha = 1 - range(frame, 186, 286);
  if (alpha <= 0.02) return;
  const y = SCENE.pipeY;
  const block = {x: SCENE.blockX, y, z: 0};
  const railA = project({x: SCENE.tokenLeft - 8, y: y + 0.62, z: 0}, cam, layout);
  const railB = project({x: block.x - 2.15, y: y + 0.62, z: 0}, cam, layout);
  ctx.save();
  ctx.globalAlpha = alpha * 0.9;
  ctx.strokeStyle = COLOR.gridMajor;
  ctx.setLineDash([2, 7]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(railA.x, railA.y);
  ctx.lineTo(railB.x, railB.y);
  ctx.stroke();
  ctx.restore();

  drawQuad(ctx, cam, layout, {x: block.x, y: y + 1.62, z: 0}, 2.05, 0.16, COLOR.muted, null, alpha * 0.9, 1.25);
  drawQuad(ctx, cam, layout, block, 2.35, 1.32, COLOR.panelStroke, '#10151F', alpha, 1.25);
  const lanes = [
    {name: 'TOK', dy: 0.78},
    {name: 'MHA', dy: 0.26},
    {name: 'FFN', dy: -0.26},
    {name: 'POOL', dy: -0.78},
  ];
  lanes.forEach((lane, i) => {
    const light = range(frame, 66 + i * 16, 92 + i * 16);
    drawQuad(
      ctx,
      cam,
      layout,
      {x: block.x + 0.15, y: y + lane.dy, z: 0},
      1.85,
      0.16,
      mix(COLOR.panelStroke, COLOR.magenta, light),
      mix('#141A24', '#3A1024', light * 0.85),
      alpha * (0.35 + 0.65 * light),
    );
    const label = project({x: block.x - 2.05, y: y + lane.dy, z: 0}, cam, layout);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = mono(12, 500);
    ctx.fillStyle = mix(COLOR.dim, COLOR.magenta, light);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(lane.name, label.x, label.y);
    ctx.restore();
  });

  const matrixAlpha = alpha * range(frame, 112, 136) * (1 - range(frame, 168, 204));
  if (matrixAlpha > 0.02) {
    const origin = project(block, cam, layout);
    const cell = 11;
    const left = origin.x - (6 * cell) / 2 + 18;
    const top = origin.y - (6 * cell) / 2;
    ctx.save();
    ctx.globalAlpha = matrixAlpha;
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 6; col++) {
        const diag = (row + col) / 10;
        const sweep = clamp((range(frame, 64, 140) - diag) * 3.2, 0, 1);
        const weight = Math.exp(-((row - col) ** 2) / 3.2) * (col === 5 ? 1 : 0.55);
        ctx.fillStyle = rgba(COLOR.magenta, 0.15 + 0.75 * weight * sweep);
        ctx.fillRect(left + col * cell, top + row * cell, cell - 2, cell - 2);
      }
    }
    ctx.restore();
  }

  TOKENS.forEach((word, index) => {
    const f = frame - index * 16;
    if (f < -10) return;
    const travel = smoother(clamp(f / 70, 0, 1));
    const x = lerp(SCENE.tokenLeft - index * 1.8, SCENE.blockX - 2.9, travel);
    const fadeIn = f < 0 ? smoother(clamp((f + 10) / 10, 0, 1)) : 1;
    const fadeOut = 1 - smoother(clamp((f - 46) / 12, 0, 1));
    const tokenAlpha = fadeIn * fadeOut;
    if (tokenAlpha < 0.02) return;
    const p = project({x, y: y + 0.62, z: 0}, cam, layout);
    ctx.save();
    ctx.globalAlpha = alpha * tokenAlpha;
    ctx.font = display(18, 500);
    const bw = ctx.measureText(word).width + 28;
    ctx.fillStyle = '#140812';
    ctx.strokeStyle = COLOR.magenta;
    ctx.lineWidth = 1.25;
    roundRect(ctx, p.x - bw / 2, p.y - 16, bw, 32, 16);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = COLOR.bone;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(word, p.x, p.y + 0.5);
    ctx.restore();
  });

  const collapse = range(frame, 146, 198);
  QUERY.slice(0, 8).forEach((value, i) => {
    const a = range(frame, 84 + i * 6, 100 + i * 6) * (1 - collapse) * alpha;
    if (a <= 0.02) return;
    const home = {x: SCENE.stackX, y: y + (3.5 - i) * 0.34, z: 0};
    const p = project(lerp3(home, {x: SCENE.stackX, y, z: 0}, collapse), cam, layout);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.font = mono(15, 500);
    ctx.fillStyle = COLOR.magenta;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(signed(value), p.x, p.y);
    ctx.restore();
  });
  const stackAlpha = alpha * range(frame, 96, 120) * (1 - collapse);
  if (stackAlpha > 0.02) {
    const top = project({x: SCENE.stackX - 0.28, y: y + 1.25, z: 0}, cam, layout);
    const bot = project({x: SCENE.stackX - 0.28, y: y - 1.25, z: 0}, cam, layout);
    const head = project({x: SCENE.stackX - 0.15, y: y + 1.55, z: 0}, cam, layout);
    ctx.save();
    ctx.globalAlpha = stackAlpha;
    ctx.strokeStyle = COLOR.magenta;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(top.x + 10, top.y);
    ctx.lineTo(top.x, top.y);
    ctx.lineTo(bot.x, bot.y);
    ctx.lineTo(bot.x + 10, bot.y);
    ctx.stroke();
    ctx.font = mono(12, 500);
    ctx.fillStyle = COLOR.magenta;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('q    8 of 768', head.x, head.y);
    ctx.restore();
  }
};

const drawRing = (ctx, cam, layout, center, radius, color, alpha, width) => {
  if (alpha <= 0.01 || radius <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (let i = 0; i <= 96; i++) {
    const a = (i / 96) * Math.PI * 2;
    const p = project({x: center.x + Math.cos(a) * radius, y: center.y, z: center.z + Math.sin(a) * radius}, cam, layout);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();
};

const drawGraph = (ctx, frame, world, cam, layout, origin) => {
  const graphAlpha = range(frame, 142, 248);
  if (graphAlpha <= 0.01) return;
  const planes = [
    {y: 1.85, r: 3.45, label: 'L2', a: 0.9},
    {y: 0, r: 5.45, label: 'L1', a: 0.75},
    {y: -1.95, r: 7.45, label: 'L0', a: 0.9},
  ];
  for (const plane of planes) {
    ctx.save();
    ctx.globalAlpha = graphAlpha * plane.a;
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLOR.grid;
    ctx.beginPath();
    for (let x = -8; x <= 8; x += 2) {
      const a = project({x, y: plane.y, z: -8}, cam, layout);
      const b = project({x, y: plane.y, z: 8}, cam, layout);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    for (let z = -8; z <= 8; z += 2) {
      const a = project({x: -8, y: plane.y, z}, cam, layout);
      const b = project({x: 8, y: plane.y, z}, cam, layout);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const a = (i / 80) * Math.PI * 2;
      const p = project({x: Math.cos(a) * plane.r, y: plane.y, z: Math.sin(a) * plane.r}, cam, layout);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = COLOR.mint;
    ctx.stroke();
    const tag = project({x: -plane.r - 0.15, y: plane.y, z: 0.2}, cam, layout);
    ctx.font = mono(13, 500);
    ctx.fillStyle = COLOR.mint;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(plane.label, tag.x, tag.y);
    ctx.restore();
  }

  ctx.save();
  ctx.lineWidth = 1;
  for (const edge of world.edges) {
    const a = world.byId.get(edge.a);
    const b = world.byId.get(edge.b);
    const energy = Math.max(nodeEnergy(a, frame, origin), nodeEnergy(b, frame, origin));
    const pa = project(a.pos, cam, layout);
    const pb = project(b.pos, cam, layout);
    const column = edge.vertical && (a.column || b.column);
    ctx.strokeStyle = column
      ? rgba(COLOR.magenta, 0.55 * graphAlpha)
      : rgba(mix('#3C4E68', COLOR.mint, 0.28 + 0.72 * energy), (0.38 + 0.4 * energy) * graphAlpha);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  ctx.restore();

  if (frame >= 301 && frame <= 680) {
    const fade = (1 - range(frame, 548, 660)) * range(frame, 301, 324);
    const ringOrigin = queryPosition(frame, world);
    drawRing(ctx, cam, layout, ringOrigin, waveRadius(frame), COLOR.magenta, fade * 0.8, 1.35);
    drawRing(ctx, cam, layout, {...ringOrigin, y: -1.95}, waveRadius(frame), COLOR.mint, fade * 0.35, 1.1);
    const op = project(ringOrigin, cam, layout);
    ctx.save();
    ctx.lineWidth = 1;
    for (const node of world.nodes) {
      const reached = reachedAmount(node, frame, ringOrigin);
      if (reached <= 0.05) continue;
      const hot = hotness(node.cosine);
      const emphasis = node.cluster || hot > 0.48;
      const p = project(node.pos, cam, layout);
      ctx.strokeStyle = rgba(emphasis ? COLOR.mint : COLOR.gridMajor, fade * reached * (emphasis ? 0.5 : 0.1));
      ctx.beginPath();
      ctx.moveTo(op.x, op.y);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  const open = heroOpen(frame);
  const heroId = world.topK[0].nodeId;
  const ordered = [...world.nodes].sort((a, b) => project(a.pos, cam, layout).z - project(b.pos, cam, layout).z);
  for (const node of ordered) {
    if (node.id === heroId && open > 0.2) continue;
    const reached = reachedAmount(node, frame, origin);
    const hot = hotness(node.cosine);
    const rest = mix('#0E4F49', COLOR.mint, 0.62 + 0.38 * hot);
    const scored = mix('#46536C', COLOR.mint, Math.pow(hot, 0.8));
    let color = mix(rest, scored, reached);
    if (node.cluster) color = mix(color, COLOR.amber, range(frame, 512, 590));
    const p = project(node.pos, cam, layout);
    const base = node.layer === 2 ? 0.3 : node.layer === 1 ? 0.22 : 0.17;
    const radius = Math.max(3, base * (node.cluster ? 1.45 : 1) * cam.zoom);
    ctx.save();
    ctx.globalAlpha = graphAlpha;
    ctx.fillStyle = color;
    ctx.strokeStyle = mix(color, COLOR.bone, node.cluster ? 0.35 : 0.12);
    ctx.lineWidth = node.cluster ? 1.75 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
};

const drawQuery = (ctx, frame, world, cam, layout) => {
  if (frame < 146) return;
  const pos = queryPosition(frame, world);
  if (frame > 150 && frame < 330) {
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = rgba(COLOR.magenta, 0.45);
    ctx.beginPath();
    for (let back = 42; back >= 0; back -= 6) {
      const p = project(queryPosition(Math.max(0, frame - back), world), cam, layout);
      if (back === 42) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.restore();
  }
  const p = project(pos, cam, layout);
  const formed = frame > 188 ? 1 : range(frame, 146, 188);
  const radius = lerp(3, 8, formed);
  const halo = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 34);
  halo.addColorStop(0, rgba(COLOR.magenta, 0.45));
  halo.addColorStop(1, rgba(COLOR.magenta, 0));
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 34, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = COLOR.magenta;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
  const pulse = 0.28 + 0.22 * (0.5 + 0.5 * Math.sin((frame / 96) * Math.PI * 2));
  ctx.strokeStyle = rgba(COLOR.magenta, pulse);
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius + 9, 0, Math.PI * 2);
  ctx.stroke();
  if (frame > 200) {
    ctx.font = mono(13, 500);
    ctx.fillStyle = COLOR.magenta;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('q', p.x + 16, p.y);
  }
};

const drawSplit = (ctx, x, y, radius, gap, color) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.75;
  ctx.beginPath();
  ctx.arc(x - gap, y, radius, Math.PI * 0.55, Math.PI * 1.45);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x + gap, y, radius, Math.PI * 1.55, Math.PI * 0.45);
  ctx.stroke();
  ctx.restore();
};

const drawPayload = (ctx, frame, world, cam, layout) => {
  const open = heroOpen(frame);
  const hero = world.byId.get(world.topK[0].nodeId);
  if (open > 0.02 && hero) {
    const anchor = project(hero.pos, cam, layout);
    const shellR = lerp(Math.max(8, 0.2 * cam.zoom), 52, open);
    const gap = open * 16;
    ctx.save();
    ctx.globalAlpha = open;
    drawSplit(ctx, anchor.x, anchor.y, shellR, gap, mix(COLOR.mint, COLOR.amber, open));
    const pw = 436;
    const ph1 = 132;
    const ph2 = 118;
    let px = anchor.x + shellR + 36;
    let py = anchor.y - 78;
    if (px + pw > layout.w - 48) px = anchor.x - shellR - 36 - pw;
    if (py < 168) py = 168;
    if (py + ph1 + ph2 + 16 > layout.h - 168) py = layout.h - 168 - ph1 - ph2 - 16;
    px += (1 - open) * 28;
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = COLOR.amber;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(anchor.x + gap, anchor.y);
    ctx.lineTo(px, py + 28);
    ctx.stroke();
    ctx.setLineDash([]);
    const hit = world.topK[0];
    plate(ctx, px, py, pw, ph1, COLOR.amber);
    ctx.font = mono(11, 500);
    ctx.fillStyle = COLOR.amber;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('07   CHUNK', px + 18, py + 14);
    ctx.font = display(16, 500);
    ctx.fillStyle = COLOR.bone;
    ctx.fillText(hit.title, px + 18, py + 34);
    ctx.font = display(15, 400);
    wrap(ctx, hit.body, pw - 40)
      .slice(0, 3)
      .forEach((line, i) => ctx.fillText(line, px + 18, py + 62 + i * 20));
    const y2 = py + ph1 + 12;
    plate(ctx, px, y2, pw, ph2, COLOR.amber);
    ctx.font = mono(11, 500);
    ctx.fillStyle = COLOR.amber;
    ctx.fillText('08   POINTER', px + 18, y2 + 14);
    [
      ['doc_id', hit.doc],
      ['chunk_key', hit.key],
      ['span', hit.span],
      ['cos', hit.cosine.toFixed(3)],
    ].forEach((row, i) => {
      const ry = y2 + 40 + i * 18;
      ctx.font = mono(12, 400);
      ctx.fillStyle = COLOR.dim;
      ctx.fillText(row[0], px + 18, ry);
      ctx.fillStyle = COLOR.amber;
      ctx.fillText(row[1], px + 130, ry);
    });
    ctx.restore();
  }
  const legend = range(frame, 536, 572) * (1 - range(frame, 724, 768));
  if (legend <= 0.02) return;
  ctx.save();
  ctx.globalAlpha = legend;
  ctx.font = mono(12, 500);
  ctx.fillStyle = COLOR.amber;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('NEAR HITS', 62, 176);
  world.topK.slice(1).forEach((hit, index) => {
    const y = 198 + index * 62;
    plate(ctx, 56, y, 268, 54, COLOR.amber);
    ctx.font = mono(12, 500);
    ctx.fillStyle = COLOR.amber;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`0${hit.rank + 1}   ${hit.title}`, 72, y + 8);
    ctx.font = mono(11, 400);
    ctx.fillStyle = COLOR.muted;
    ctx.fillText(`${hit.key}    ${hit.cosine.toFixed(3)}`, 72, y + 30);
  });
  ctx.restore();
};

const drawContext = (ctx, frame, world, cam, layout) => {
  const t = range(frame, 766, 834);
  if (t <= 0.01) return;
  const width = 488;
  const height = 688;
  const box = {x: lerp(layout.w + 30, layout.w - 52 - width, t), y: 168, w: width, h: height};
  ctx.save();
  ctx.globalAlpha = t;
  plate(ctx, box.x, box.y, box.w, box.h, COLOR.amber);
  ctx.font = mono(12, 500);
  ctx.fillStyle = COLOR.amber;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('SYNTHESIS PROMPT', box.x + 20, box.y + 16);
  ctx.font = display(26, 500);
  ctx.fillStyle = COLOR.bone;
  ctx.fillText('Context window', box.x + 20, box.y + 38);
  ctx.strokeStyle = COLOR.panelStroke;
  ctx.beginPath();
  ctx.moveTo(box.x + 18, box.y + 78);
  ctx.lineTo(box.x + box.w - 18, box.y + 78);
  ctx.stroke();
  ctx.font = mono(12, 500);
  ctx.fillStyle = COLOR.magenta;
  ctx.fillText('QUERY', box.x + 20, box.y + 92);
  ctx.font = display(18, 500);
  ctx.fillStyle = COLOR.bone;
  ctx.fillText('where does the model', box.x + 20, box.y + 114);
  ctx.fillText('store meaning', box.x + 20, box.y + 138);
  ctx.font = mono(12, 500);
  ctx.fillStyle = COLOR.amber;
  ctx.fillText('RETRIEVED    k = 4    metric = cosine', box.x + 20, box.y + 172);
  ctx.restore();

  world.topK.forEach((hit, index) => {
    const node = world.byId.get(hit.nodeId);
    const progress = frame < 790 ? 0 : smoother(clamp((frame - (802 + index * 10)) / 148, 0, 1));
    const slotY = box.y + 204 + index * 108;
    const slotX = box.x + 18;
    const slotW = box.w - 36;
    const fill = clamp((progress - 0.58) / 0.32, 0, 1) * t;
    ctx.save();
    ctx.globalAlpha = t;
    ctx.strokeStyle = rgba(COLOR.amber, 0.28 + 0.72 * fill);
    roundRect(ctx, slotX, slotY, slotW, 96, 6);
    ctx.stroke();
    ctx.font = mono(12, 500);
    ctx.fillStyle = rgba(COLOR.amber, 0.35 + 0.65 * fill);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`0${index + 1}`, slotX + 12, slotY + 10);
    ctx.font = display(16, 500);
    ctx.fillStyle = rgba(COLOR.bone, 0.28 + 0.72 * fill);
    ctx.fillText(hit.title, slotX + 44, slotY + 8);
    ctx.font = display(14, 400);
    ctx.fillStyle = rgba(COLOR.muted, 0.25 + 0.75 * fill);
    wrap(ctx, hit.body, slotW - 28)
      .slice(0, 2)
      .forEach((line, lineIndex) => ctx.fillText(line, slotX + 12, slotY + 36 + lineIndex * 18));
    ctx.font = mono(11, 400);
    ctx.fillStyle = rgba(COLOR.amber, 0.25 + 0.75 * fill);
    ctx.fillText(`${hit.key}    ${hit.doc}`, slotX + 12, slotY + 76);
    ctx.restore();

    if (!node || progress <= 0.02 || progress >= 0.72) return;
    const from = project(node.pos, cam, layout);
    const to = {x: box.x - 8, y: slotY + 48};
    const u = progress / 0.72;
    const arc = Math.sin(u * Math.PI) * (28 + index * 22) * (index % 2 === 0 ? -1 : 1);
    const x = lerp(from.x, to.x, u);
    const y = lerp(from.y, to.y, u) + arc;
    if (x > box.x - 20) return;
    ctx.save();
    ctx.globalAlpha = t;
    ctx.strokeStyle = rgba(COLOR.amber, 0.7);
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    plate(ctx, x - 8, y - 16, 176, 32, COLOR.amber);
    ctx.font = mono(12, 500);
    ctx.fillStyle = COLOR.amber;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(`0${index + 1}  ${hit.title}`, x + 4, y);
    ctx.restore();
  });
};

const drawHud = (ctx, frame, cam, w, h) => {
  ctx.save();
  ctx.fillStyle = rgba(COLOR.ink, 0.78);
  roundRect(ctx, 44, 46, 470, 100, 8);
  ctx.fill();
  ctx.font = mono(12, 500);
  ctx.fillStyle = COLOR.mint;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('ANATOMY JIG', 62, 60);
  ctx.font = display(28, 500);
  ctx.fillStyle = COLOR.bone;
  ctx.fillText('Semantic vector retrieval', 62, 80);
  ctx.font = mono(12, 400);
  ctx.fillStyle = COLOR.dim;
  ctx.fillText('ORTHOGRAPHIC   ·   30 FPS   ·   1020 FRAMES', 62, 118);

  ctx.fillStyle = rgba(COLOR.ink, 0.78);
  roundRect(ctx, w - 292, 46, 236, 112, 8);
  ctx.fill();
  const pitch = (cam.pitch * 180) / Math.PI;
  const yaw = (cam.yaw * 180) / Math.PI;
  ctx.textAlign = 'right';
  ctx.font = mono(20, 500);
  ctx.fillStyle = COLOR.bone;
  ctx.fillText(formatTime(frame), w - 68, 58);
  ctx.font = mono(12, 400);
  ctx.fillStyle = COLOR.muted;
  ctx.fillText(`f${String(frame).padStart(4, '0')}  /  1020`, w - 68, 84);
  ctx.fillStyle = Math.abs(pitch - 45) < 1.2 && frame > 250 && frame < 360 ? COLOR.amber : COLOR.muted;
  ctx.fillText(`PITCH  ${pitch.toFixed(1)}°`, w - 68, 106);
  ctx.fillStyle = COLOR.muted;
  ctx.fillText(`YAW  ${yaw.toFixed(1)}°`, w - 68, 126);

  const act = actIndex(frame);
  ctx.fillStyle = rgba(COLOR.ink, 0.82);
  ctx.fillRect(40, h - 156, w - 80, 112);
  ctx.font = display(20, 500);
  ctx.fillStyle = COLOR.bone;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(`${ACTS[act].id}    ${ACTS[act].name}`, 62, h - 142);
  ctx.font = display(15, 400);
  ctx.fillStyle = COLOR.muted;
  ctx.fillText(ACTS[act].line, 62, h - 114);

  const trackX = 62;
  const trackY = h - 78;
  const trackW = w - 124;
  ctx.fillStyle = COLOR.gridMajor;
  ctx.fillRect(trackX, trackY, trackW, 2);
  const play = trackX + (frame / 1019) * trackW;
  ctx.fillStyle = COLOR.magenta;
  ctx.fillRect(trackX, trackY, Math.max(0, play - trackX), 2);
  ctx.beginPath();
  ctx.arc(play, trackY + 1, 4, 0, Math.PI * 2);
  ctx.fill();
  const bounds = [0, 150, 300, 510, 750, 1020];
  const colors = [COLOR.magenta, COLOR.magenta, COLOR.mint, COLOR.amber, COLOR.amber];
  for (let i = 0; i < 5; i++) {
    const x = trackX + (bounds[i] / 1020) * trackW;
    ctx.font = mono(11, 500);
    ctx.fillStyle = i === act ? colors[i] : COLOR.dim;
    ctx.fillText(ACTS[i].id, x, trackY + 10);
    ctx.font = mono(11, 400);
    ctx.fillStyle = i === act ? COLOR.bone : COLOR.dim;
    ctx.fillText(ACTS[i].name, x + 26, trackY + 10);
  }
  if (frame >= 300 && frame <= 560) {
    ctx.globalAlpha = range(frame, 300, 328) * (1 - range(frame, 520, 560));
    ctx.font = mono(13, 400);
    ctx.fillStyle = COLOR.mint;
    ctx.fillText('cos = (q · v) / (|q| |v|)', 62, h - 168);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = COLOR.gridMajor;
  ctx.lineWidth = 1;
  ctx.strokeRect(32.5, 32.5, w - 65, h - 65);
  ctx.restore();
};

export const drawFrame = (ctx, frame, w, h) => {
  const world = getWorld();
  const cam = cameraAt(frame, world);
  const layout = layoutAt(frame, w, h);
  const origin = queryPosition(Math.max(frame, 300), world);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = COLOR.ink;
  ctx.fillRect(0, 0, w, h);
  drawScreenGrid(ctx, w, h);
  ctx.save();
  ctx.beginPath();
  ctx.rect(34, 34, w - 68, h - 68);
  ctx.clip();
  drawGraph(ctx, frame, world, cam, layout, origin);
  drawPipeline(ctx, frame, cam, layout);
  drawQuery(ctx, frame, world, cam, layout);
  ctx.restore();
  const vignette = ctx.createRadialGradient(w / 2, h / 2, w * 0.28, w / 2, h / 2, w * 0.72);
  vignette.addColorStop(0, 'rgba(0,0,0,0)');
  vignette.addColorStop(1, 'rgba(0,0,0,0.42)');
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, w, h);
  drawPayload(ctx, frame, world, cam, layout);
  drawContext(ctx, frame, world, cam, layout);
  drawHud(ctx, frame, cam, w, h);
};

export const CHAPTERS = [
  {frame: 0, label: '01 Tokens'},
  {frame: 151, label: '02 Drop'},
  {frame: 301, label: '03 Sweep'},
  {frame: 511, label: '04 Payload'},
  {frame: 751, label: '05 Top-k'},
];

export const mountPlayer = (canvas, ui) => {
  const ctx = canvas.getContext('2d');
  let frame = 0;
  let playing = true;
  let last = 0;
  let acc = 0;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) {
    playing = false;
    frame = 300;
  }
  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
  };
  const paint = () => {
    ctx.setTransform(canvas.width / 1920, 0, 0, canvas.height / 1080, 0, 0);
    drawFrame(ctx, frame, 1920, 1080);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (ui.scrub) ui.scrub.value = String(frame);
    if (ui.time) ui.time.textContent = `${formatTime(frame)}   f${String(frame).padStart(4, '0')}`;
    if (ui.play) ui.play.textContent = playing ? 'Pause' : 'Play';
  };
  const loop = (now) => {
    if (!last) last = now;
    const delta = now - last;
    last = now;
    if (playing) {
      acc += delta;
      const step = 1000 / FPS;
      while (acc >= step) {
        frame = (frame + 1) % DURATION;
        acc -= step;
      }
    }
    paint();
    requestAnimationFrame(loop);
  };
  resize();
  window.addEventListener('resize', () => {
    resize();
    paint();
  });
  ui.play?.addEventListener('click', () => {
    playing = !playing;
  });
  ui.scrub?.addEventListener('input', () => {
    frame = Number(ui.scrub.value);
    acc = 0;
    paint();
  });
  ui.chapters?.forEach((button) => {
    button.addEventListener('click', () => {
      frame = Number(button.dataset.frame);
      acc = 0;
      playing = true;
      paint();
    });
  });
  requestAnimationFrame(loop);
  return {get frame() { return frame; }};
};
