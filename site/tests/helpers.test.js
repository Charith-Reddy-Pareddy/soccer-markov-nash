import test from 'node:test';
import assert from 'node:assert/strict';
import { certify, stepPolicy, simulateGames } from '../src/explorer/helpers.js';

const M = [[0,0,0,0],[.729,0,.729,0],[0,0,0,0],[.81,.81,-.59049,0]];
function board(p = [0,1,0,0], q = [0,0,0,1]) {
  return { state_list: ['s'], states: {s: [0,p,q,M,Array(16).fill(-1)]} };
}

test('reported security ties are not a probability distribution', () => {
  assert.deepEqual(certify(M).rowTies, [0,1,2]);
  const b = board();
  for (let i=0; i<100; i++) {
    const step = stepPolicy(b, 's', 'minimax', 'minimax');
    assert.equal(step.a0, 1);
    assert.equal(step.a1, 3);
  }
});

test('random draw zero cannot select a zero-probability action', () => {
  const random = Math.random;
  try {
    Math.random = () => 0;
    const step = stepPolicy(board(), 's', 'minimax', 'minimax');
    assert.equal(step.a0, 1);
    assert.equal(step.a1, 3);
  } finally { Math.random = random; }
});

test('simulation uses the same selected policy as stepping', () => {
  const b = board();
  b.states.s[4].fill(-2);
  b.states.s[4][7] = -1;
  assert.equal(simulateGames(b, 's', 100, 'minimax', 'minimax', .9).p0, 100);
});

test('two-axis ties still have a saddle', () => {
  const cert = certify([[0,0,0,0],[0,0,0,0],[1,-1,0,0],[0,0,0,0]]);
  assert.equal(cert.kind, 'pure');
  assert.equal(cert.gap, 0);
});
