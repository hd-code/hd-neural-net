const assert = require('assert');
const Vector = require('../../build/models/Vector');

// -----------------------------------------------------------------------------

describe('models/Vector', () => {
    describe('isVector()', () => {
        it('should return true for valid vectors', () => {
            assert.ok(Vector.isVector([1]));
            assert.ok(Vector.isVector([1,2]));
            assert.ok(Vector.isVector([-1,1]));
            assert.ok(Vector.isVector([3,4,5,6,12]));
        });

        it('should return false for wrong data type', () => {
            assert.ok(!Vector.isVector(undefined));
            assert.ok(!Vector.isVector(null));
            assert.ok(!Vector.isVector(false));
            assert.ok(!Vector.isVector(['1','2']));
            assert.ok(!Vector.isVector('string'));
            assert.ok(!Vector.isVector({name:'Hello',age:12}));
        });
    });

    it('createRandom()', () => {
        it('should create valid vectors', () => {
            assert.ok(Vector.isVector(Vector.createRandom(2)));
            assert.ok(Vector.isVector(Vector.createRandom(3)));
            assert.ok(Vector.isVector(Vector.createRandom(0)));
        });
    });

    it('add()', () => {
        const VEC1 = [1,2,3];
        const VEC2 = [4,5,6];
        const RESULT = [5,7,9];

        it('should return correct results', () => {
            assert.deepStrictEqual(Vector.add(VEC1, VEC2), RESULT);
            assert.deepStrictEqual(Vector.add(VEC2, VEC1), RESULT);
        });
    });

    it('sub()', () => {
        const VEC1 = [1,2,3];
        const VEC2 = [4,5,6];

        it('should return correct results', () => {
            assert.deepStrictEqual(Vector.sub(VEC1, VEC2), [-3,-3,-3]);
            assert.deepStrictEqual(Vector.sub(VEC2, VEC1), [3,3,3]);
        });
    });

    it('mul()', () => {
        const VEC1 = [1,2,3];
        const VEC2 = [4,5,6];
        const RESULT = [4,10,18];

        it('should return correct results', () => {
            assert.deepStrictEqual(Vector.mul(VEC1, VEC2), RESULT);
            assert.deepStrictEqual(Vector.mul(VEC2, VEC1), RESULT);
        });
    });

    it('scale()', () => {
        const VEC1 = [1,2,3];
        const VEC2 = [4,5,6];

        it('should return correct results', () => {
            assert.deepStrictEqual(Vector.scale(1, VEC1), VEC1);
            assert.deepStrictEqual(Vector.scale(1, VEC2), VEC2);

            assert.deepStrictEqual(Vector.scale(2, VEC1), [2,4,6]);
            assert.deepStrictEqual(Vector.scale(2, VEC2), [8,10,12]);
        });
    });

    it('dot()', () => {
        const VEC1 = [1,2,3];
        const VEC2 = [4,5,6];

        it('should return correct results', () => {
            assert.deepStrictEqual(Vector.dot(VEC1, VEC2), 32);
            assert.deepStrictEqual(Vector.dot(VEC2, VEC1), 32);
        });
    });
});