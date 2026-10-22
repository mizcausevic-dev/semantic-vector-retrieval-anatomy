import React, {useEffect, useLayoutEffect, useRef, useState} from 'react';
import {AbsoluteFill, continueRender, delayRender, useCurrentFrame} from 'remotion';
import {drawFrame} from '../player.js';
import '@fontsource/space-grotesk/400.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';

export const VectorAnatomy: React.FC = () => {
  const frame = useCurrentFrame();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fontsReady = useRef(false);
  const [fontHandle] = useState(() => delayRender('fonts'));
  const drawHandle = delayRender(`draw-${frame}`);

  useEffect(() => {
    const loads = [
      document.fonts.load('400 32px "Space Grotesk"'),
      document.fonts.load('500 64px "Space Grotesk"'),
      document.fonts.load('400 22px "JetBrains Mono"'),
      document.fonts.load('500 28px "JetBrains Mono"'),
    ];
    Promise.all(loads)
      .then(() => {
        fontsReady.current = true;
      })
      .catch(() => {
        fontsReady.current = true;
      })
      .finally(() => {
        continueRender(fontHandle);
      });
  }, [fontHandle]);

  useLayoutEffect(() => {
    let settled = false;
    const paint = () => {
      if (settled) return;
      settled = true;
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) drawFrame(ctx, frame, 1920, 1080);
      continueRender(drawHandle);
    };
    if (fontsReady.current) {
      paint();
      return;
    }
    const timer = window.setInterval(() => {
      if (!fontsReady.current) return;
      window.clearInterval(timer);
      paint();
    }, 20);
    return () => {
      window.clearInterval(timer);
      if (!settled) continueRender(drawHandle);
    };
  }, [drawHandle, frame]);

  return (
    <AbsoluteFill style={{background: '#0A0D14'}}>
      <canvas ref={canvasRef} width={1920} height={1080} />
    </AbsoluteFill>
  );
};
