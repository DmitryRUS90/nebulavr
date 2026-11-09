import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DodecahedronGeometry,
  DoubleSide,
  EdgesGeometry,
  Float32BufferAttribute,
  Group,
  IcosahedronGeometry,
  LineBasicMaterial,
  LineSegments,
  Material,
  Mesh,
  MeshBasicMaterial,
  OctahedronGeometry,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Quaternion,
  ShaderMaterial,
  SphereGeometry,
  Sprite,
  SpriteMaterial,
  TorusGeometry,
  Vector3,
} from 'three';
import { fx } from '../../core/fx';
import { TIERS } from '../../core/perf';
import { sceneDef } from '../../core/scenes';
import { useStore } from '../../core/store';
import { prefersReducedMotion } from '../../core/util';
import { NOISE } from '../../gfx/glsl';
import { getGlowTexture } from '../../gfx/textures';
import { game } from './game';
import { CHANNEL_COLORS } from './palette';

/**
 * Stage 0.5.4a: the static look of the energy core. A layered reactor in the middle, three channels that run
 * out to three energy nodes, and slow continuous motion. Nothing here reads the game yet; the reactions to
 * stabVis, instVis, flows, alertMid and alertHigh arrive in 0.5.4b.
 *
 * Everything animated is driven from useFrame through uniforms and transforms, never through React state.
 */

const CAMERA_Z = 13;
/** Centre of the composition, lifted so the core sits above the HUD that fills the lower part of the screen */
const CENTER_Y = 1.15;
const CORE_EDGE = 1.4;

type NodeKind = 'octa' | 'ico' | 'dodeca';

interface ChannelDef {
  id: 'alpha' | 'beta' | 'gamma';
  color: string;
  /** Position of the energy node relative to the core */
  at: [number, number, number];
  kind: NodeKind;
}

const CHANNELS: ChannelDef[] = [
  { id: 'alpha', color: CHANNEL_COLORS[0], at: [-4.35, -1.15, -0.5], kind: 'octa' },
  { id: 'beta', color: CHANNEL_COLORS[1], at: [4.35, -1.15, -0.5], kind: 'ico' },
  { id: 'gamma', color: CHANNEL_COLORS[2], at: [0, 2.65, -0.9], kind: 'dodeca' },
];

const CYAN = new Color('#27D8F2');
const VIOLET = new Color('#7356D9');
const SHELL_BASE = new Color('#4a3aa8');
const HOT = new Color('#ffb35c');
const ALARM = new Color('#ff4d6a');
/** Instability turns the glow from cyan toward pink and then red, a path that never passes through green */
const WARN = new Color('#ff5ca8');
const DEEP_RED = new Color('#8a1636');
const WIN_COLOR = new Color('#b8f6ff');
const WHITE = new Color('#ffffff');
const RING_BASE = new Color('#7fdcf0');
const WIRE_BASE = new Color('#8f7dff');
const HEART_BASE = new Color('#d8f8ff');
const SPEEDS = [0.35, -0.24, 0.17];

/** Eases a value toward a target at a rate that does not depend on the frame time */
const damp = (a: number, b: number, k: number, dt: number) => a + (b - a) * (1 - Math.exp(-k * dt));
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/* ---------------------------------------------------------------- shaders */

const CORE_VERT = /* glsl */ `
  varying vec3 vN; varying vec3 vV; varying vec3 vP;
  void main(){
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    vP = position;
    gl_Position = projectionMatrix * mv;
  }
`;

/**
 * Churning plasma ball. uStab orders the glow, uChaos stirs it with extra turbulence, uGain is the overall brightness.
 * The hue comes in through uA and uB, which the frame loop blends from the game state.
 */
const CORE_FRAG = /* glsl */ `
  uniform float uTime; uniform float uPulse; uniform float uStab; uniform float uChaos; uniform float uGain;
  uniform vec3 uA; uniform vec3 uB;
  varying vec3 vN; varying vec3 vV; varying vec3 vP;
  ${NOISE}
  void main(){
    vec3 d = normalize(vP);
    float n = fbm(d * (2.1 + uChaos * 1.5) + vec3(0.0, uTime * 0.22, uTime * 0.15));
    if (uChaos > 0.02) n += uChaos * 0.45 * snoise(d * 6.5 + vec3(uTime * 0.8, -uTime * 0.6, uTime * 0.5));
    float veins = 1.0 - abs(snoise(d * 3.6 - vec3(uTime * 0.12, 0.0, 0.0)));
    float facing = abs(dot(normalize(vN), normalize(vV)));
    float heat = clamp(0.55 + n * 0.9, 0.0, 1.0);
    heat = mix(heat, 0.78, uStab * (1.0 - uChaos) * 0.45);
    vec3 col = mix(uB, uA, heat);
    col += pow(veins, mix(5.0, 3.2, uStab)) * (0.55 + 0.35 * uStab) * uA;
    col += vec3(1.0) * pow(facing, 3.2) * (0.55 + 0.25 * uPulse);
    col *= (0.8 + 0.45 * uPulse) * uGain;
    gl_FragColor = vec4(col, 1.0);
  }
`;

/** Translucent outer shell: fresnel rim, a fine lat/long grid and a slow scan band */
const SHELL_FRAG = /* glsl */ `
  uniform float uTime; uniform float uChaos; uniform float uGain; uniform vec3 uA; uniform vec3 uB;
  varying vec3 vN; varying vec3 vV; varying vec3 vP;
  ${NOISE}
  void main(){
    vec3 d = normalize(vP);
    float rim = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.3);
    float lon = atan(d.z, d.x) / 6.2831853 * 24.0;
    float lat = asin(clamp(d.y, -1.0, 1.0)) / 3.14159265 * 12.0;
    if (uChaos > 0.02) {
      float w = snoise(d * 3.0 + vec3(uTime * 0.5));
      lon += w * uChaos * 0.7;
      lat += w * uChaos * 0.5;
    }
    float gl1 = 1.0 - smoothstep(0.0, 0.05, abs(fract(lon) - 0.5) * 2.0 - 0.9);
    float gl2 = 1.0 - smoothstep(0.0, 0.05, abs(fract(lat) - 0.5) * 2.0 - 0.9);
    float grid = max(gl1, gl2);
    float scan = pow(0.5 + 0.5 * sin(d.y * 6.0 - uTime * 0.9), 10.0);
    vec3 col = mix(uB, uA, rim);
    float a = rim * 0.75 + grid * (0.06 + 0.2 * rim) + scan * 0.08;
    gl_FragColor = vec4(col * (0.7 + rim) * uGain, a * (0.9 + 0.4 * uChaos));
  }
`;

const TUBE_VERT = /* glsl */ `
  uniform float uX0; uniform float uX1; uniform float uOff; uniform float uWob;
  varying float vA; varying vec3 vN; varying vec3 vV;
  void main(){
    vec3 p = position;
    p.yz *= 1.0 + uOff * 0.4 * sin(position.x * 4.0 - uWob * (5.0 + uOff * 7.0));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vA = (position.x - uX0) / (uX1 - uX0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

/** Open tube: a faint volume with bright packets running toward the core */
const TUBE_FRAG = /* glsl */ `
  uniform float uTime; uniform vec3 uColor; uniform float uBase; uniform float uGain; uniform float uRepeat;
  uniform float uOff; uniform float uWob; uniform float uDim; uniform float uSel;
  varying float vA; varying vec3 vN; varying vec3 vV;
  void main(){
    float rim = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 1.3);
    float ph = fract((1.0 - vA) * uRepeat - uTime * 0.28);
    float packet = pow(1.0 - ph, 5.0);
    float ends = smoothstep(0.0, 0.07, vA) * smoothstep(1.0, 0.93, vA);
    float flick = 1.0 + uOff * 0.7 * sin(uWob * (6.0 + uOff * 10.0) + vA * 7.0);
    float a = (uBase + uGain * (1.0 + uOff * 1.2) * packet) * (0.35 + 0.65 * rim) * ends * flick * uDim * (1.0 + uSel * 0.25);
    vec3 c = mix(uColor, vec3(1.0), uOff * 0.35);
    gl_FragColor = vec4(c * (0.9 + packet * 0.9), a);
  }
`;

/** Particles travel from the node to the core on a loose helix around the channel axis */
const FLOW_VERT = /* glsl */ `
  attribute vec4 aSeed;
  uniform float uTime; uniform float uX0; uniform float uX1; uniform float uScale; uniform float uSize;
  uniform float uWob; uniform float uOff; uniform float uDim;
  varying float vFade;
  void main(){
    float t = fract(aSeed.x + uTime * aSeed.w * 0.16);
    float x = mix(uX1, uX0, t);
    float ang = aSeed.y + t * 9.4 + uTime * 0.5;
    float r = aSeed.z * (0.07 + 0.08 * sin(t * 3.14159)) * (1.0 + uOff * 3.0 * (0.5 + 0.5 * sin(uWob * 7.0 + aSeed.y * 3.0)));
    vec3 p = vec3(x, cos(ang) * r, sin(ang) * r);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vFade = sin(t * 3.14159) * uDim;
    gl_PointSize = uSize * uScale / max(0.1, -mv.z) * (0.6 + 0.8 * aSeed.z) * (1.0 + uOff * 0.6);
    gl_Position = projectionMatrix * mv;
  }
`;
const FLOW_FRAG = /* glsl */ `
  uniform vec3 uColor;
  varying float vFade;
  void main(){
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = 1.0 - clamp(d, 0.0, 1.0);
    a *= a;
    gl_FragColor = vec4(mix(uColor, vec3(1.0), a * 0.5) * 1.3, a * vFade);
  }
`;

/* --------------------------------------------------------------- helpers */

/** Small deterministic generator so the layout is the same on every visit */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function nodeGeometry(kind: NodeKind, r: number) {
  if (kind === 'octa') return new OctahedronGeometry(r);
  if (kind === 'ico') return new IcosahedronGeometry(r, 0);
  return new DodecahedronGeometry(r, 0);
}

/** Short radial ticks around a circle, like the scale on an instrument */
function tickRing(radius: number, count: number, len: number) {
  const pos = new Float32Array(count * 6);
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    const l = i % 6 === 0 ? len * 1.9 : len;
    const c = Math.cos(a);
    const s = Math.sin(a);
    pos.set([c * radius, s * radius, 0, c * (radius + l), s * (radius + l), 0], i * 6);
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(pos, 3));
  return g;
}

function setTube(m: ShaderMaterial, time: number, wob: number, off: number, dim: number, sel: number) {
  const u = m.uniforms;
  u.uTime.value = time;
  u.uWob.value = wob;
  u.uOff.value = off;
  u.uDim.value = dim;
  u.uSel.value = sel;
}

function additive<T extends Material>(m: T) {
  m.transparent = true;
  m.depthWrite = false;
  m.blending = AdditiveBlending;
  return m;
}

interface ChannelParts {
  def: ChannelDef;
  color: Color;
  length: number;
  quat: Quaternion;
  pos: Vector3;
  tubeOuter: ShaderMaterial;
  tubeInner: ShaderMaterial;
  flow: ShaderMaterial;
  flowGeo: BufferGeometry;
  nodeMat: MeshBasicMaterial;
  nodeEdge: LineBasicMaterial;
  nodeRing: MeshBasicMaterial;
  nodeGlow: SpriteMaterial;
  nodeGeo: BufferGeometry;
  nodeEdgeGeo: EdgesGeometry;
  clampMat: MeshBasicMaterial;
  arcMat: MeshBasicMaterial;
}

/* ---------------------------------------------------------------- scene */

export function CoreGraphics() {
  const tier = useStore((s) => s.tier);
  const cfg = TIERS[tier];
  const lite = tier === 'lite';
  const camera = useThree((s) => s.camera) as PerspectiveCamera;
  const size = useThree((s) => s.size);
  const reduced = useMemo(() => prefersReducedMotion(), []);
  const glow = useMemo(() => getGlowTexture(), []);

  const root = useRef<Group>(null);
  const coreGroup = useRef<Group>(null);
  const innerSphere = useRef<Mesh>(null);
  const wire = useRef<LineSegments>(null);
  const ringRefs = useRef<(Group | null)[]>([]);
  const tickRef = useRef<Group>(null);
  const dust = useRef<Points>(null);
  const heart = useRef<Sprite>(null);
  const halo = useRef<Sprite>(null);
  const clampRefs = useRef<(Mesh | null)[][]>([[], [], []]);
  const nodeRefs = useRef<(Group | null)[]>([]);
  const nodeRingA = useRef<(Mesh | null)[]>([]);
  const nodeRingB = useRef<(Mesh | null)[]>([]);
  const nodeBead = useRef<(Mesh | null)[]>([]);
  const nodeSpin = useRef<(Group | null)[]>([]);
  const shock = useRef<Mesh>(null);
  /** Eased copies of the game state and animation clocks. Plain data, never React state. */
  const sim = useRef({
    stab: 0.3, inst: 0.1, alert: 0, win: 0, loss: 0, pulse: 0,
    ch: [0, 1, 2].map(() => ({ off: 0, level: 0.5, sel: 0, t: 0, dim: 1 })),
  });

  /* ----- resources, built once per tier and released with the scene ----- */
  const res = useMemo(() => {
    const seg = Math.max(24, Math.round(cfg.labDetail * 1.6));
    const oct = lite ? 2 : 3;

    const coreMat = new ShaderMaterial({
      defines: { OCT: oct },
      uniforms: { uTime: { value: 0 }, uPulse: { value: 0 }, uStab: { value: 0 }, uChaos: { value: 0 }, uGain: { value: 1 }, uA: { value: CYAN.clone() }, uB: { value: VIOLET.clone() } },
      vertexShader: CORE_VERT,
      fragmentShader: CORE_FRAG,
    });
    const shellMat = new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
      blending: AdditiveBlending,
      uniforms: { uTime: { value: 0 }, uChaos: { value: 0 }, uGain: { value: 1 }, uA: { value: CYAN.clone() }, uB: { value: SHELL_BASE.clone() } },
      vertexShader: CORE_VERT,
      fragmentShader: SHELL_FRAG,
    });

    const coreGeo = new SphereGeometry(0.85, seg, Math.round(seg * 0.75));
    const shellGeo = new SphereGeometry(CORE_EDGE, seg, Math.round(seg * 0.75));
    const wireGeo = new EdgesGeometry(new IcosahedronGeometry(1.12, lite ? 0 : 1));
    const ringGeo = new TorusGeometry(1, 0.012, 6, lite ? 72 : 144);
    const arcGeo = new TorusGeometry(1, 0.032, 6, lite ? 40 : 72, 2.2);
    const ticksGeo = tickRing(2.55, lite ? 48 : 96, 0.07);
    const beadGeo = new SphereGeometry(0.06, 10, 8);

    const wireMat = additive(new LineBasicMaterial({ color: '#8f7dff', opacity: 0.32 }));
    const ringMat = additive(new MeshBasicMaterial({ color: '#7fdcf0', opacity: 0.5 }));
    const tickMat = additive(new LineBasicMaterial({ color: '#7fdcf0', opacity: 0.4 }));
    const beadMat = additive(new MeshBasicMaterial({ color: '#ffffff', opacity: 0.95 }));
    const arcMats = CHANNELS.map((c) => additive(new MeshBasicMaterial({ color: c.color, opacity: 0.85 })));
    const heartMat = additive(new SpriteMaterial({ map: glow, color: '#d8f8ff', opacity: 0.7 }));
    const haloMat = additive(new SpriteMaterial({ map: glow, color: '#6b52e0', opacity: 0.4 }));

    // slow orbiting dust around the reactor
    const dustCount = Math.round(cfg.fieldPoints / 9);
    const r1 = rng(7);
    const dustPos = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      const u = r1() * 2 - 1;
      const a = r1() * Math.PI * 2;
      const rr = 2.1 + r1() * 1.6;
      const s = Math.sqrt(1 - u * u);
      dustPos.set([s * Math.cos(a) * rr, u * rr * 0.8, s * Math.sin(a) * rr], i * 3);
    }
    const dustGeo = new BufferGeometry();
    dustGeo.setAttribute('position', new BufferAttribute(dustPos, 3));
    const dustMat = additive(new PointsMaterial({ map: glow, color: '#a9b8ff', size: 0.11, opacity: 0.6, sizeAttenuation: true }));

    // channels
    const perChannel = Math.max(8, Math.round(cfg.particles / 36));
    const channels: ChannelParts[] = CHANNELS.map((def, ci) => {
      const color = new Color(def.color);
      const pos = new Vector3(...def.at);
      const length = pos.length();
      const quat = new Quaternion().setFromUnitVectors(new Vector3(1, 0, 0), pos.clone().normalize());
      const x0 = CORE_EDGE - 0.1;
      const x1 = length - 0.42;

      const tube = (base: number, gain: number, repeat: number) =>
        new ShaderMaterial({
          transparent: true,
          depthWrite: false,
          side: DoubleSide,
          blending: AdditiveBlending,
          uniforms: {
            uTime: { value: 0 },
            uColor: { value: color.clone() },
            uBase: { value: base },
            uGain: { value: gain },
            uRepeat: { value: repeat },
            uX0: { value: x0 },
            uX1: { value: x1 },
            uOff: { value: 0 },
            uWob: { value: 0 },
            uDim: { value: 1 },
            uSel: { value: 0 },
          },
          vertexShader: TUBE_VERT,
          fragmentShader: TUBE_FRAG,
        });

      const rnd = rng(31 + ci * 17);
      const seeds = new Float32Array(perChannel * 4);
      for (let i = 0; i < perChannel; i++) {
        seeds.set([(i + rnd() * 0.8) / perChannel, rnd() * Math.PI * 2, 0.4 + rnd() * 0.6, 0.7 + rnd() * 0.6], i * 4);
      }
      const flowGeo = new BufferGeometry();
      flowGeo.setAttribute('position', new BufferAttribute(new Float32Array(perChannel * 3), 3));
      flowGeo.setAttribute('aSeed', new BufferAttribute(seeds, 4));
      const flow = new ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: AdditiveBlending,
        uniforms: {
          uTime: { value: 0 },
          uColor: { value: color.clone() },
          uX0: { value: x0 },
          uX1: { value: x1 },
          uScale: { value: 400 },
          uSize: { value: 0.11 },
          uWob: { value: 0 },
          uOff: { value: 0 },
          uDim: { value: 1 },
        },
        vertexShader: FLOW_VERT,
        fragmentShader: FLOW_FRAG,
      });

      const nodeGeo = nodeGeometry(def.kind, 0.26);
      return {
        def,
        color,
        length,
        quat,
        pos,
        tubeOuter: tube(0.1, 0.5, 2.5),
        tubeInner: tube(0.18, 0.95, 4),
        flow,
        flowGeo,
        nodeMat: additive(new MeshBasicMaterial({ color, opacity: 0.85 })),
        nodeEdge: additive(new LineBasicMaterial({ color: color.clone().lerp(new Color('#ffffff'), 0.45), opacity: 0.95 })),
        nodeRing: additive(new MeshBasicMaterial({ color, opacity: 0.75 })),
        nodeGlow: additive(new SpriteMaterial({ map: glow, color, opacity: 0.55 })),
        nodeGeo,
        nodeEdgeGeo: new EdgesGeometry(nodeGeometry(def.kind, 0.42)),
        clampMat: additive(new MeshBasicMaterial({ color, opacity: 0.7 })),
        arcMat: arcMats[ci],
      };
    });

    const tubeGeo = (c: ChannelParts, radius: number) => {
      const len = c.length - 0.42 - (CORE_EDGE - 0.1);
      const g = new CylinderGeometry(radius, radius, len, lite ? 10 : 16, lite ? 12 : 24, true);
      g.rotateZ(-Math.PI / 2);
      g.translate(CORE_EDGE - 0.1 + len / 2, 0, 0);
      return g;
    };
    const tubesOuter = channels.map((c) => tubeGeo(c, 0.19));
    const tubesInner = channels.map((c) => tubeGeo(c, 0.04));
    const clampGeo = new TorusGeometry(0.27, 0.016, 6, lite ? 18 : 28);
    clampGeo.rotateY(Math.PI / 2);
    const nodeRingGeo = new TorusGeometry(0.62, 0.012, 6, lite ? 40 : 72);
    const nodeBeadGeo = new SphereGeometry(0.045, 8, 6);
    const shockGeo = new TorusGeometry(1, 0.02, 6, lite ? 72 : 128);
    const shockMat = additive(new MeshBasicMaterial({ color: WIN_COLOR, opacity: 0 }));

    return {
      coreMat, shellMat, coreGeo, shellGeo, wireGeo, ringGeo, arcGeo, ticksGeo, beadGeo,
      wireMat, ringMat, tickMat, beadMat, arcMats, heartMat, haloMat,
      dustGeo, dustMat, channels, tubesOuter, tubesInner, clampGeo, nodeRingGeo, nodeBeadGeo, shockGeo, shockMat,
    };
  }, [cfg.labDetail, cfg.fieldPoints, cfg.particles, lite, glow]);

  // R3F does not dispose geometries and materials that are handed over as props, so release them here
  useEffect(() => {
    return () => {
      const dispose = (o: { dispose: () => void }) => o.dispose();
      [res.coreMat, res.shellMat, res.coreGeo, res.shellGeo, res.wireGeo, res.ringGeo, res.arcGeo, res.ticksGeo, res.beadGeo,
        res.wireMat, res.ringMat, res.tickMat, res.beadMat, res.heartMat, res.haloMat, res.dustGeo, res.dustMat,
        res.clampGeo, res.nodeRingGeo, res.nodeBeadGeo, res.shockGeo, res.shockMat].forEach(dispose);
      res.arcMats.forEach(dispose);
      res.tubesOuter.forEach(dispose);
      res.tubesInner.forEach(dispose);
      res.channels.forEach((c) => {
        [c.tubeOuter, c.tubeInner, c.flow, c.flowGeo, c.nodeMat, c.nodeEdge, c.nodeRing, c.nodeGlow, c.nodeGeo, c.nodeEdgeGeo, c.clampMat].forEach(dispose);
      });
      // the glow texture is shared by other scenes, so it stays
    };
  }, [res]);

  // the camera is shared between scenes: put it in front of the reactor and hand the drift on when leaving
  useEffect(() => {
    fx.baseFov = sceneDef('core').baseFov;
    camera.position.set(0, 0, CAMERA_Z);
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    return () => {
      fx.cruise = sceneDef(useStore.getState().scene).cruise;
    };
  }, [camera]);

  // keep the whole composition inside the frame on narrow screens
  const aspect = size.width / size.height;
  const halfH = Math.tan((sceneDef('core').baseFov * Math.PI) / 360) * CAMERA_Z;
  const portrait = aspect < 0.85;
  const fitScale = Math.min(1, (halfH * aspect * 0.92) / 5.2, portrait ? 0.7 : 1);
  const centerY = portrait ? 1.7 : CENTER_Y;

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    const dtc = dt < 0.1 ? dt : 0.1;
    const slow = reduced ? 0.2 : 1;
    const g = game;
    const s = sim.current;

    /* ----- ease the game state: read-only, nothing here feeds back into the game ----- */
    const won = g.phase === 'won';
    const lost = g.phase === 'lost';
    const playing = g.phase === 'playing';
    s.stab = damp(s.stab, g.stabVis, 5, dtc);
    s.inst = damp(s.inst, g.instVis, 5, dtc);
    s.alert = damp(s.alert, won || lost ? 0 : g.alertHigh ? 1 : g.alertMid ? 0.5 : 0, 3, dtc);
    s.win = damp(s.win, won ? 1 : 0, won ? 3 : 6, dtc);
    s.loss = damp(s.loss, lost ? 1 : 0, lost ? 4 : 6, dtc);

    const win = s.win;
    const loss = s.loss;
    const stab = Math.max(s.stab, win);
    const inst = s.inst * (1 - win);
    const aMid = Math.min(s.alert * 2, 1);
    const aHi = Math.max(0, (s.alert - 0.5) * 2);
    const chaos = clamp01(inst * 0.75 + aMid * 0.2 + aHi * 0.5 + loss * 0.9);
    const since = won || lost ? g.since : 0;
    // one soft flash when the round ends, and a recoil that dies away
    const flash = won ? Math.min(since / 0.2, 1) * Math.exp(-since * 1.5) : 0;
    const recoil = lost ? Math.min(since / 0.1, 1) * Math.exp(-since * 2.2) : 0;
    const stutter = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(t * 14 + Math.sin(t * 3) * 2);
    const flick = loss * stutter * 0.4;
    // slow, smooth surges while the reactor is critical
    const burst = aHi * Math.pow(Math.max(0, Math.sin(t * 2.6 * slow)), 10);

    // the pulse keeps its phase, so a change of rate never makes it jump
    s.pulse += dtc * (1.5 + stab * 0.7 + inst * 0.9 + s.alert * 2.2) * (1 - win * 0.35) * (1 - loss * 0.3) * (reduced ? 0.5 : 1);
    const amp = 0.65 + stab * 0.25 + s.alert * 0.5 + inst * 0.2;
    const pulse = clamp01(0.5 + (0.5 + 0.5 * Math.sin(s.pulse) - 0.5) * amp);
    const gain = Math.max(0.15, (0.9 + stab * 0.35 + s.alert * 0.15 + burst * 0.35 + win * 0.25 + flash * 0.8) * (1 - loss * 0.35) - flick);

    /* ----- colour: cyan when calm, warm with instability, red when critical or lost, pale when stabilized ----- */
    const hot = clamp01(inst * 0.55 + aMid * 0.35);
    const alarm = clamp01(aHi * 0.8 + loss * 0.7);
    const coreA = res.coreMat.uniforms.uA.value as Color;
    const coreB = res.coreMat.uniforms.uB.value as Color;
    const deep = clamp01(inst * 0.5 + aHi * 0.7 + loss * 0.8);
    coreA.copy(CYAN).lerp(WARN, clamp01(hot * 1.3)).lerp(ALARM, alarm).lerp(WIN_COLOR, win * 0.5);
    coreB.copy(VIOLET).lerp(DEEP_RED, deep);
    (res.shellMat.uniforms.uA.value as Color).copy(coreA);
    (res.shellMat.uniforms.uB.value as Color).copy(SHELL_BASE).lerp(DEEP_RED, deep);

    const cu = res.coreMat.uniforms;
    cu.uTime.value = t * slow;
    cu.uPulse.value = pulse;
    cu.uStab.value = stab;
    cu.uChaos.value = chaos;
    cu.uGain.value = gain;
    const su = res.shellMat.uniforms;
    su.uTime.value = t * slow;
    su.uChaos.value = chaos;
    su.uGain.value = 0.85 + gain * 0.15;

    /* ----- core ----- */
    const wobble = 0.5 * (Math.sin(t * 3.1) + Math.sin(t * 5.3 + 1));
    if (innerSphere.current) {
      innerSphere.current.scale.setScalar(
        1 + (0.03 + stab * 0.02 + s.alert * 0.04) * pulse + (reduced ? 0 : 0.025 * chaos * wobble) + burst * 0.06 + flash * 0.12 - loss * 0.1,
      );
    }
    if (heart.current) heart.current.scale.setScalar(2.3 + 0.35 * pulse + flash * 2.5 + burst * 0.6 - loss * 0.5);
    res.heartMat.color.copy(coreA).lerp(WHITE, 0.6);
    res.heartMat.opacity = clamp01((0.55 + 0.2 * pulse) * (0.7 + gain * 0.3) - flick);
    if (halo.current) halo.current.scale.setScalar(6.2 + 0.5 * pulse + flash * 1.5 + aHi * 0.6);
    res.haloMat.color.copy(coreB).lerp(coreA, 0.3);
    res.haloMat.opacity = clamp01(0.4 * (0.6 + gain * 0.4) + flash * 0.4 - loss * 0.15);

    const spinK = (1 + stab * 0.5 + chaos * 0.7) * (1 - loss * 0.7);
    if (coreGroup.current) coreGroup.current.rotation.y += dt * 0.12 * slow * spinK;
    if (wire.current) {
      wire.current.rotation.y -= dt * 0.2 * slow * spinK;
      wire.current.rotation.x += dt * 0.07 * slow * spinK;
    }
    res.wireMat.color.copy(WIRE_BASE).lerp(HOT, chaos * 0.6).lerp(ALARM, alarm * 0.5);
    res.ringMat.color.copy(RING_BASE).lerp(HOT, hot * 0.6).lerp(ALARM, alarm * 0.6);
    res.ringMat.opacity = clamp01(0.5 * (0.6 + gain * 0.4));
    if (tickRef.current) tickRef.current.rotation.z -= dt * 0.05 * slow * (1 + chaos);
    if (dust.current) {
      dust.current.rotation.y += dt * 0.04 * slow * (1 + chaos * 2);
      dust.current.rotation.x = Math.sin(t * 0.1) * 0.12;
      dust.current.scale.setScalar(1 + flash * 0.35 + recoil * 0.4);
    }

    /* ----- channels: each one reads its own flow ----- */
    const fovScale = (size.height * state.viewport.dpr) / (2 * Math.tan((camera.fov * Math.PI) / 360));
    for (let i = 0; i < 3; i++) {
      const f = g.flows[i];
      const c = s.ch[i];
      const p = res.channels[i];
      c.off = damp(c.off, clamp01(f.off) * (1 - win), 6, dtc);
      c.level = damp(c.level, f.level, 6, dtc);
      c.sel = damp(c.sel, playing && g.selected === i ? 1 : 0, 8, dtc);
      const st = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(t * (7 + i * 2.9) + i * 2);
      c.dim = 1 - loss * (0.5 + 0.4 * st);
      const off = c.off;
      // the packets follow the real level of the flow and speed up when it strays
      c.t += dtc * (0.6 + 0.8 * c.level) * (1 + off * 0.5) * (1 - loss * 0.7) * (1 + win * 0.15) * slow;
      const dim = c.dim * (0.8 + 0.4 * c.level) * (1 + win * 0.15 + flash * 0.4);
      const wob = t * slow;

      setTube(p.tubeOuter, c.t, wob, off, dim, c.sel);
      setTube(p.tubeInner, c.t, wob, off, dim, c.sel);
      const fu = p.flow.uniforms;
      fu.uTime.value = c.t;
      fu.uWob.value = wob;
      fu.uOff.value = off;
      fu.uDim.value = c.dim;
      fu.uScale.value = fovScale * fitScale;

      const nodePulse = Math.sin(t * (7 + off * 10) * slow + i * 2.1);
      p.nodeGlow.opacity = clamp01((0.45 + 0.15 * Math.sin(t * 1.1 + i * 2.1) + off * 0.25 * nodePulse + c.sel * 0.15 + flash * 0.3) * c.dim);
      const alive = 0.55 + 0.45 * c.dim;
      p.nodeMat.opacity = 0.85 * alive;
      p.nodeEdge.opacity = 0.95 * alive;
      p.nodeRing.opacity = 0.75 * alive;
      p.clampMat.opacity = 0.7 * alive;

      // the arc of the matching ring in the core shows the same channel
      p.arcMat.color.copy(p.color).lerp(WHITE, off * 0.45);
      p.arcMat.opacity = clamp01(0.85 * (0.6 + 0.4 * c.dim) * (1 + off * 0.2 * nodePulse));

      const cl = clampRefs.current[i];
      for (let k = 0; k < cl.length; k++) {
        const m = cl[k];
        if (m) m.rotation.x = t * 0.5 * slow * (k % 2 ? -1 : 1);
      }
      const spin = nodeSpin.current[i];
      if (spin) {
        const k = dt * slow * (1 + off * 3) * (1 - loss * 0.8);
        spin.rotation.y += k * 0.35;
        spin.rotation.x += k * 0.12;
      }
      const a = nodeRingA.current[i];
      const b = nodeRingB.current[i];
      if (a) a.rotation.z += dt * 0.7 * slow * (1 + off * 2);
      if (b) b.rotation.z -= dt * 0.5 * slow * (1 + off * 2);
      const bead = nodeBead.current[i];
      if (bead) {
        const ang = t * 0.9 * slow + i * 2.1;
        bead.position.set(Math.cos(ang) * 0.62, Math.sin(ang) * 0.31, Math.sin(ang) * 0.53);
      }
      const n = nodeRefs.current[i];
      if (n) {
        const jitter = reduced ? 0 : off * 0.06 * Math.sin(t * 13 + i * 2);
        n.position.set(p.pos.x + jitter, p.pos.y + Math.sin(t * 0.7 + i * 1.7) * 0.05 * slow, p.pos.z);
        n.scale.setScalar(1 + off * 0.15 * nodePulse + c.sel * 0.06 + flash * 0.15);
      }

      // the ring that carries this channel's arc turns faster and wavers when this flow strays
      const rg = ringRefs.current[i];
      if (rg) {
        rg.rotation.z += dt * SPEEDS[i] * slow * (1 + stab * 0.4 + chaos * 1 + off * 1.2) * (1 - loss * 0.75);
        rg.rotation.x = reduced ? 0 : (chaos * 0.2 + off * 0.2) * Math.sin(t * 2.3 + i * 2);
      }
    }

    /* ----- the end of a round: an expanding ring, pale on a win and red on a loss ----- */
    if (shock.current) {
      const live = (won || lost) && since < 2.6;
      shock.current.visible = live;
      if (live) {
        const k = since / 2.6;
        shock.current.scale.setScalar(1.2 + (1 - (1 - k) * (1 - k)) * 2);
        res.shockMat.color.copy(won ? WIN_COLOR : ALARM);
        res.shockMat.opacity = (1 - k) * (1 - k) * (won ? 0.7 : 0.55);
      }
    }

    // calm parallax with the pointer, plus a short jolt on a loss
    if (root.current) {
      const r = root.current;
      const k = Math.min(1, dt * 2);
      r.rotation.x += (-state.pointer.y * 0.06 - r.rotation.x) * k;
      r.rotation.y += (state.pointer.x * 0.1 - r.rotation.y) * k;
      const shake = reduced ? 0 : recoil * 0.09;
      r.position.set(Math.sin(since * 53) * shake, centerY + Math.cos(since * 61) * shake * 0.7, 0);
    }
  });

  return (
    <group ref={root} position={[0, centerY, 0]} scale={fitScale}>
      <sprite ref={halo} material={res.haloMat} renderOrder={0} />

      <group ref={coreGroup}>
        <mesh ref={innerSphere} geometry={res.coreGeo} material={res.coreMat} renderOrder={1} />
        <mesh geometry={res.shellGeo} material={res.shellMat} renderOrder={4} />
        <lineSegments ref={wire} geometry={res.wireGeo} material={res.wireMat} renderOrder={3} />
      </group>
      <sprite ref={heart} material={res.heartMat} renderOrder={2} />

      {/* three tilted rings, each carrying one of the channel arcs and a bright bead */}
      {CHANNELS.map((c, i) => {
        const r = 1.75 + i * 0.3;
        const tilt: [number, number, number][] = [[1.15, 0.2, 0], [0.4, 0.1, 1.0], [0.95, 0.85, 0.3]];
        return (
          <group key={c.id} rotation={tilt[i]}>
            <group
              ref={(g) => {
                ringRefs.current[i] = g;
              }}
            >
              <mesh geometry={res.ringGeo} material={res.ringMat} scale={r} renderOrder={3} />
              <mesh geometry={res.arcGeo} material={res.arcMats[i]} scale={r} rotation={[0, 0, i * 2.1]} renderOrder={3} />
              <mesh geometry={res.beadGeo} material={res.beadMat} position={[r, 0, 0]} renderOrder={3} />
            </group>
          </group>
        );
      })}
      <group ref={tickRef} rotation={[1.35, 0.1, 0.2]}>
        <lineSegments geometry={res.ticksGeo} material={res.tickMat} renderOrder={3} />
      </group>

      <mesh ref={shock} geometry={res.shockGeo} material={res.shockMat} renderOrder={6} visible={false} />

      <points ref={dust} geometry={res.dustGeo} material={res.dustMat} renderOrder={2} frustumCulled={false} />

      {res.channels.map((c, i) => (
        <group key={c.def.id}>
          <group quaternion={c.quat}>
            <mesh geometry={res.tubesOuter[i]} material={c.tubeOuter} renderOrder={5} />
            <mesh geometry={res.tubesInner[i]} material={c.tubeInner} renderOrder={5} />
            <points geometry={c.flowGeo} material={c.flow} renderOrder={6} frustumCulled={false} />
            {[0.25, 0.5, 0.75].map((f, k) => (
              <mesh
                key={f}
                ref={(m) => {
                  clampRefs.current[i][k] = m;
                }}
                geometry={res.clampGeo}
                material={c.clampMat}
                position={[CORE_EDGE + (c.length - 0.42 - CORE_EDGE) * f, 0, 0]}
                renderOrder={5}
              />
            ))}
          </group>

          {/* the energy node at the end of the channel */}
          <group
            ref={(g) => {
              nodeRefs.current[i] = g;
            }}
            position={c.pos}
          >
            <sprite material={c.nodeGlow} scale={[2.1, 2.1, 1]} renderOrder={1} />
            <group
              ref={(g) => {
                nodeSpin.current[i] = g;
              }}
            >
              <mesh geometry={c.nodeGeo} material={c.nodeMat} renderOrder={2} />
              <lineSegments geometry={c.nodeEdgeGeo} material={c.nodeEdge} renderOrder={3} />
            </group>
            <group rotation={[1.1, 0.3 + i * 0.4, 0]}>
              <mesh
                ref={(m) => {
                  nodeRingA.current[i] = m;
                }}
                geometry={res.nodeRingGeo}
                material={c.nodeRing}
                renderOrder={3}
              />
            </group>
            <group rotation={[0.4, -0.5 - i * 0.3, 0.9]}>
              <mesh
                ref={(m) => {
                  nodeRingB.current[i] = m;
                }}
                geometry={res.nodeRingGeo}
                material={c.nodeRing}
                scale={0.78}
                renderOrder={3}
              />
            </group>
            <mesh
              ref={(m) => {
                nodeBead.current[i] = m;
              }}
              geometry={res.nodeBeadGeo}
              material={res.beadMat}
              renderOrder={3}
            />
          </group>
        </group>
      ))}
    </group>
  );
}
