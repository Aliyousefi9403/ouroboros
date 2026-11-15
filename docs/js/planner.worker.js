// Runs the budgeted search off the main thread so the board keeps animating.
import { loadModel, plan, boardValue } from './agents.js';

const ready = loadModel('../models', 'board');
onmessage = async ({ data }) => {
  const model = await ready;
  postMessage({ id: data.id, plan: plan(boardValue(model), data.st, data.budget, data.gamma) });
};
