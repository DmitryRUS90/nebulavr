import { useFrame } from '@react-three/fiber';
import { useLayoutEffect } from 'react';
import { CoreGraphics } from './CoreGraphics';
import { game, isLive, resetToIntro, step } from './game';

/** Longest slice of time the simulation takes in one step, so a stalled frame cannot jump the round ahead. */
const MAX_STEP = 0.1;

/**
 * Runs the energy core simulation. The scene is only mounted while scene 06 is on screen,
 * so this is the single place that steps the game. The 3D reactor itself lives in CoreGraphics.
 */
export function CoreScene() {
  // Entering starts from the intro. Leaving puts the game back there, which also drops a round that was still running.
  useLayoutEffect(() => {
    resetToIntro();
    const onVisibility = () => {
      game.paused = document.hidden && isLive();
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      resetToIntro();
    };
  }, []);

  useFrame((_, dt) => step(Math.min(dt, MAX_STEP)));

  return <CoreGraphics />;
}
