const assert = require('assert');
const ActF = require('../../build/models/ActFunction');

// -----------------------------------------------------------------------------

describe('models/ActFunction', () => {
    describe('isActFunction()', () => {
        it('should return true for valid activation functions', () => {
            assert.ok(ActF.isActFunction(ActF.EActFunction.Linear));
            assert.ok(ActF.isActFunction(ActF.EActFunction.Sigmoid));
            assert.ok(ActF.isActFunction(ActF.EActFunction.Tanh));
            assert.ok(ActF.isActFunction(ActF.EActFunction.HardTanh));
            assert.ok(ActF.isActFunction(ActF.EActFunction.RectifiedLinear));
            assert.ok(ActF.isActFunction(ActF.EActFunction.LeakyRectifiedLinear));
            assert.ok(ActF.isActFunction(ActF.EActFunction.SoftPlus));
            assert.ok(ActF.isActFunction(ActF.EActFunction.Softmax));
            assert.ok(ActF.isActFunction(ActF.EActFunction.Binary));
        });

        it('should return false for invalid activation functions', () => {
            assert.ok(!ActF.isActFunction(-1));
            assert.ok(!ActF.isActFunction(9));
        });

        it('should return false for wrong data type', () => {
            assert.ok(!ActF.isActFunction(undefined));
            assert.ok(!ActF.isActFunction(null));
            assert.ok(!ActF.isActFunction(false));
            assert.ok(!ActF.isActFunction(300.5));
            assert.ok(!ActF.isActFunction('string'));
            assert.ok(!ActF.isActFunction([1,2,3,4]));
            assert.ok(!ActF.isActFunction({name:'Hello',age:12}));
        });
    });

    describe('applyToVector()', () => {
        const INPUT = [1,2];

        // TODO: implement tests for each activation function separately
        it('should return correct results', () => {
            assert.deepStrictEqual(ActF.applyToVector(INPUT, ActF.EActFunction.Linear), INPUT);
            assert.deepStrictEqual(ActF.applyToVector(INPUT, ActF.EActFunction.Linear, true), [1,1]);
        });
    });
});