const assert = require('assert');
const Matrix = require('../../build/models/Matrix');

// -----------------------------------------------------------------------------

describe('models/Matrix', () => {
    describe('isMatrix()', () => {
        it('should return true for valid matrizes', () => {
            assert.ok(Matrix.isMatrix([[1]]));
            assert.ok(Matrix.isMatrix([[1,2],[1,2],[1,2]]));
            assert.ok(Matrix.isMatrix([[-1,1]]));
            assert.ok(Matrix.isMatrix([[3,4,5,6,12],[3,4,5,6,12]]));
        });

        it('should return false for invalid matrizes', () => {
            assert.ok(!Matrix.isMatrix([[1,2],[1,2],[1]]));
            assert.ok(!Matrix.isMatrix([[-1],[-1,1]]));
            assert.ok(!Matrix.isMatrix([[3,4,5,6],[3,4,5,6,12]]));
        });

        it('should return false for wrong data type', () => {
            assert.ok(!Matrix.isMatrix(undefined));
            assert.ok(!Matrix.isMatrix(null));
            assert.ok(!Matrix.isMatrix(false));
            assert.ok(!Matrix.isMatrix([1,2]));
            assert.ok(!Matrix.isMatrix(['1','2']));
            assert.ok(!Matrix.isMatrix('string'));
            assert.ok(!Matrix.isMatrix({name:'Hello',age:12}));
        });
    });

    it('createRandom()', () => {
        it('should create valid vectors', () => {
            assert.ok(Matrix.isMatrix(Matrix.createRandom(2,3)));
            assert.ok(Matrix.isMatrix(Matrix.createRandom(3,5)));
            assert.ok(Matrix.isMatrix(Matrix.createRandom(1,1)));
        });
    });

    it('transpose()', () => {
        const MAT1 = [[1,2,3],[4,5,6]];
        const MAT2 = [[4,2,5],[1,3,7]];

        it('should return correct results', () => {
            assert.deepStrictEqual(Matrix.transpose(MAT1), [[1,4],[2,5],[3,6]]);
            assert.deepStrictEqual(Matrix.transpose(MAT2), [[4,1],[2,3],[5,7]]);
        });
    });

    it('add()', () => {
        const MAT1 = [[1,2,3],[4,5,6]];
        const MAT2 = [[4,2,5],[1,3,7]];
        const RESULT = [[5,4,8],[5,8,13]];

        it('should return correct results', () => {
            assert.deepStrictEqual(Matrix.add(MAT1, MAT2), RESULT);
            assert.deepStrictEqual(Matrix.add(MAT2, MAT1), RESULT);
        });
    });

    it('sub()', () => {
        const MAT1 = [[1,2,3],[4,5,6]];
        const MAT2 = [[4,2,5],[1,3,7]];

        it('should return correct results', () => {
            assert.deepStrictEqual(Matrix.sub(MAT1, MAT2), [[-3,0,-2],[3,2,-1]]);
            assert.deepStrictEqual(Matrix.sub(MAT2, MAT1), [[3,0,2],[-3,-1,1]]);
        });
    });

    it('mulVector()', () => {
        const MAT = [[1,2,3],[4,5,6]];
        const VEC = [4,2];

        it('should return correct results', () => {
            assert.deepStrictEqual(Matrix.mulVector(VEC, MAT), [12,18,24]);
        });
    });

    it('mul()', () => {
        const MAT1 = [[4,2],[1,3]];
        const MAT2 = [[1,2,3],[4,5,6]];

        it('should return correct results', () => {
            assert.deepStrictEqual(Matrix.mul(MAT1, MAT2), [[12,18,24],[13,17,19]]);
        });
    });

    it('scale()', () => {
        const MAT1 = [[4,2],[1,3]];
        const MAT2 = [[1,2,3],[4,5,6]];

        it('should return correct results', () => {
            assert.deepStrictEqual(Matrix.scale(1, MAT1), MAT1);
            assert.deepStrictEqual(Matrix.scale(1, MAT2), MAT2);

            assert.deepStrictEqual(Matrix.scale(2, MAT1), [[8,4],[2,6]]);
            assert.deepStrictEqual(Matrix.scale(2, MAT2), [[2,4,6],[8,10,12]]);
        });
    });
});