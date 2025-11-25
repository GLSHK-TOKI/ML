import { assert } from 'chai';
import { isHealthy } from '../src/controllers/health.js';

describe('Should be healthy', () => {
  it('should return true at the end', () => {
    assert.isOk(isHealthy());
  });
});

