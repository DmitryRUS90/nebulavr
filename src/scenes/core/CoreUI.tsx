import { ArrowRight, ArrowsClockwise, Play } from '@phosphor-icons/react';
import { useEffect, useRef, useSyncExternalStore, type CSSProperties } from 'react';
import { Button, Panel, SceneTitle } from '../../components/ui';
import { useRaf } from '../../core/hooks';
import { goTo } from '../../core/navigation';
import { useStore } from '../../core/store';
import { fmtTime } from '../../core/util';
import { CHANNEL_COLORS } from './palette';
import { FLOW_COUNT, FLOW_DEFS, TIME_LIMIT, game, isLive, selectFlow, setValve, startRound, subscribe } from './game';
import './core.css';

// The game only announces phase changes, so these two are the only values React follows.
const usePhase = () => useSyncExternalStore(subscribe, () => game.phase);
const useCount = () => useSyncExternalStore(subscribe, () => game.count);

/** Writes only when the text really changed, so the per frame loop leaves the DOM alone most of the time. */
const setText = (el: HTMLElement | null, text: string) => {
  if (el && el.textContent !== text) el.textContent = text;
};

/** True while the person is working in a field, a slider or a menu, where the arrow keys belong to that element. */
const isTyping = (target: EventTarget | null) =>
  target instanceof Element && target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])') !== null;

/**
 * Arrow keys steer the selected flow through the game's own `keyDir`.
 * Everything is attached while the scene is open and removed when it closes, so re-entering never stacks handlers.
 */
function useCoreInput() {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target) || !isLive()) return;
      e.preventDefault();
      game.keyDir = e.key === 'ArrowUp' ? 1 : -1;
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if ((e.key === 'ArrowUp' && game.keyDir === 1) || (e.key === 'ArrowDown' && game.keyDir === -1)) game.keyDir = 0;
    };
    // a key released while the window was not listening would leave the valve moving by itself
    const release = () => {
      game.keyDir = 0;
    };
    // Pausing the round itself is done by CoreScene; here a held key is only let go.
    const onVisibility = () => {
      if (document.hidden) release();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', release);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', release);
      document.removeEventListener('visibilitychange', onVisibility);
      release();
    };
  }, []);
}

interface FlowRefs {
  row: HTMLLIElement | null;
  band: HTMLSpanElement | null;
  level: HTMLSpanElement | null;
  state: HTMLElement | null;
  input: HTMLInputElement | null;
  fill: HTMLSpanElement | null;
  value: HTMLElement | null;
}

/** Time, stability, instability, and the three flows with their valves. Values are written straight to the DOM every frame, no React renders. */
function Hud() {
  const time = useRef<HTMLElement>(null);
  const stabText = useRef<HTMLElement>(null);
  const stabBar = useRef<HTMLDivElement>(null);
  const instText = useRef<HTMLElement>(null);
  const instBar = useRef<HTMLDivElement>(null);
  const flows = useRef<FlowRefs[]>(FLOW_DEFS.map(() => ({ row: null, band: null, level: null, state: null, input: null, fill: null, value: null })));

  useRaf(() => {
    setText(time.current, fmtTime(Math.ceil(Math.max(0, TIME_LIMIT - game.t))));

    const stab = Math.round(game.stability);
    setText(stabText.current, `${stab}%`);
    const stabFill = stabBar.current?.firstElementChild as HTMLElement | null;
    if (stabFill) stabFill.style.transform = `scaleX(${game.stability / 100})`;
    stabBar.current?.setAttribute('aria-valuenow', String(stab));

    const inst = Math.round(game.instability);
    setText(instText.current, `${inst}%`);
    const instFill = instBar.current?.firstElementChild as HTMLElement | null;
    if (instFill) instFill.style.transform = `scaleX(${game.instability / 100})`;
    instBar.current?.classList.toggle('meter-warn', game.instability >= 50);
    instBar.current?.setAttribute('aria-valuenow', String(inst));

    const playing = game.phase === 'playing';
    for (let i = 0; i < FLOW_COUNT; i++) {
      const f = game.flows[i];
      const r = flows.current[i];
      if (r.band) {
        r.band.style.left = `${Math.max(0, f.center - f.half) * 100}%`;
        r.band.style.width = `${Math.min(1, f.half * 2) * 100}%`;
      }
      if (r.level) r.level.style.left = `${f.level * 100}%`;
      if (r.row) {
        r.row.classList.toggle('is-off', !f.inBand);
        r.row.classList.toggle('is-selected', game.selected === i);
        if (game.selected === i) r.row.setAttribute('aria-current', 'true');
        else r.row.removeAttribute('aria-current');
      }
      setText(r.state, f.inBand ? 'In band' : 'Out of band');

      // The valve can also move from the arrow keys, so the slider follows the game and not the other way round.
      const pct = Math.round(f.valve * 100);
      if (r.input) {
        if (Math.abs(Number(r.input.value) - f.valve) > 0.005) r.input.value = String(f.valve);
        // the round has not started during the countdown, so the valves wait until it does
        if (r.input.disabled === playing) r.input.disabled = !playing;
        if (r.input.getAttribute('aria-valuetext') !== `${pct}%`) r.input.setAttribute('aria-valuetext', `${pct}%`);
      }
      if (r.fill) r.fill.style.transform = `scaleX(${f.valve})`;
      setText(r.value, `${pct}%`);
    }
  });

  return (
    <Panel className="core-hud" label="Core readings" index={0}>
      <div className="core-stats">
        <p className="core-stat core-stat-time">
          <span className="caption-4 core-label">Time left</span>
          <strong ref={time} className="heading-4 mono core-time">
            {fmtTime(TIME_LIMIT)}
          </strong>
        </p>
        <div className="core-stat">
          <div className="core-stat-head">
            <span className="caption-4 core-label">Stability</span>
            <strong ref={stabText} className="mono body-5">
              0%
            </strong>
          </div>
          <div ref={stabBar} className="meter meter-ok" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={0} aria-label="Stability">
            <span style={{ transform: 'scaleX(0)' }} />
          </div>
        </div>
        <div className="core-stat">
          <div className="core-stat-head">
            <span className="caption-4 core-label">Instability</span>
            <strong ref={instText} className="mono body-5">
              0%
            </strong>
          </div>
          <div ref={instBar} className="meter meter-violet" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={0} aria-label="Instability">
            <span style={{ transform: 'scaleX(0)' }} />
          </div>
        </div>
      </div>

      <ul className="core-flows" aria-label="Energy flows">
        {FLOW_DEFS.map((d, i) => (
          <li
            key={d.id}
            ref={(el) => void (flows.current[i].row = el)}
            className="core-flow"
            style={{ ['--flow' as string]: CHANNEL_COLORS[i] } as CSSProperties}
            onPointerDown={() => selectFlow(i)}
          >
            <div className="core-flow-head">
              <span className="core-flow-name body-6">
                <i className="core-flow-dot" aria-hidden />
                {d.name}
              </span>
              <span ref={(el) => void (flows.current[i].state = el)} className="core-flow-state mono caption-4">
                In band
              </span>
            </div>
            <span className="core-flow-track" aria-hidden>
              <span ref={(el) => void (flows.current[i].band = el)} className="core-flow-band" />
              <span ref={(el) => void (flows.current[i].level = el)} className="core-flow-level" />
            </span>
            <label className="range core-valve">
              <span className="range-head">
                <span className="range-label caption-4">Valve</span>
                <span ref={(el) => void (flows.current[i].value = el)} className="range-value mono caption-4">
                  50%
                </span>
              </span>
              <span className="range-track">
                <span className="range-rail" aria-hidden />
                <span ref={(el) => void (flows.current[i].fill = el)} className="range-fill" aria-hidden style={{ transform: 'scaleX(0.5)' }} />
                <input
                  ref={(el) => void (flows.current[i].input = el)}
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  defaultValue={0.5}
                  disabled
                  aria-label={`${d.name} valve`}
                  onFocus={() => selectFlow(i)}
                  onChange={(e) => setValve(i, Number(e.currentTarget.value))}
                />
              </span>
            </label>
          </li>
        ))}
      </ul>
      <p className="caption-4 core-hint">Drag a valve, or select a flow and use the up and down arrow keys.</p>
    </Panel>
  );
}

export function CoreUI() {
  const phase = usePhase();
  const count = useCount();
  const busy = useStore((s) => s.transitioning);
  useCoreInput();

  const live = phase === 'playing' || phase === 'countdown';
  const announce =
    phase === 'countdown'
      ? `Starting in ${Math.max(1, count)}`
      : phase === 'playing'
        ? 'The round has started'
        : phase === 'won'
          ? 'The core is stable'
          : phase === 'lost'
            ? 'The core failed'
            : '';

  return (
    <section className="scene-ui core-ui" aria-label="Energy core">
      <SceneTitle n="06" name="Energy core" sub="Stabilization" />
      <p className="visually-hidden" aria-live="assertive">
        {announce}
      </p>

      <div className={`core-main ${live ? 'is-playing' : ''}`}>
        {phase === 'intro' && (
          <Panel className="core-card is-intro" eyebrow="Stabilization" title="Hold the core steady" index={1}>
            <p className="body-6 core-lede">
              Three energy flows feed the core, and each one has a safe band that drifts over time. Move the valves to keep every flow inside its
              band and build stability, before instability reaches the limit or the time runs out.
            </p>
            <ul className="core-facts mono caption-4">
              <li>{FLOW_COUNT} flows</li>
              <li>{TIME_LIMIT} seconds</li>
            </ul>
            <div className="core-card-actions">
              <Button variant="primary" size="lg" sound={null} icon={<Play size={20} weight="bold" />} onClick={() => startRound()}>
                Start
              </Button>
            </div>
          </Panel>
        )}

        {live && <Hud />}

        {phase === 'countdown' && (
          <p key={count} className="core-count heading-1 mono" aria-hidden>
            {Math.max(1, count)}
          </p>
        )}

        {phase === 'won' && (
          <Panel className="core-card is-won" eyebrow="Core stable" title="The core holds" index={1}>
            <p className="body-6 core-lede">Every flow settled inside its band and the containment field is steady.</p>
            <dl className="core-result">
              <div>
                <dt className="caption-4">Time</dt>
                <dd className="mono body-5">{fmtTime(game.result.time)}</dd>
              </div>
              <div>
                <dt className="caption-4">Integrity</dt>
                <dd className="mono body-5">{game.result.integrity}%</dd>
              </div>
            </dl>
            <div className="core-card-actions">
              <Button variant="primary" size="lg" disabled={busy} iconAfter={<ArrowRight size={20} weight="bold" />} onClick={() => goTo('beyond')}>
                Continue
              </Button>
            </div>
          </Panel>
        )}

        {phase === 'lost' && (
          <Panel className="core-card is-lost" eyebrow="Core unstable" title="The core failed" index={1}>
            {game.loseReason && <p className="body-6 core-reason">{game.loseReason}</p>}
            <div className="core-card-actions">
              <Button variant="primary" size="lg" sound={null} icon={<ArrowsClockwise size={20} weight="bold" />} onClick={() => startRound()}>
                Retry
              </Button>
            </div>
          </Panel>
        )}
      </div>
    </section>
  );
}
