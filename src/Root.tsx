import React from 'react';
import {Composition} from 'remotion';
import {VectorAnatomy} from './VectorAnatomy';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="VectorAnatomy"
      component={VectorAnatomy}
      durationInFrames={1020}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
