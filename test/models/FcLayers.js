const assert = require('assert');
const FcLayer = require('../../build/models/FcLayer');

const ActFunction = require('../../build/models/ActFunction');

// -----------------------------------------------------------------------------

describe('models/FcLayer', () => {
    const TEST_LAYER = {
        actFunction: ActFunction.EActFunction.Linear,
        bias: [.5, .3],
        weights: [
            [.2, .3],
            [.6, .1]
        ]
    };
    const INPUT = [1,1];
    const OUTPUT = [1.3, .7];
    const EXP_OUTPUT = [1,1];
    const ERROR = [.3, -.3];

    describe('isFcLayer()', () => {
        it('should return true for valid activation functions', () => {
            assert.ok(FcLayer.isFcLayer(TEST_LAYER));
        });

        it('should return false for invalid activation functions', () => {
            const actFunction = TEST_LAYER.actFunction;
            const weights = TEST_LAYER.weights;
            const bias = TEST_LAYER.bias;

            const BIAS_TOO_LONG = { actFunction, weights, bias: [.5, .3, .4] };
            const BIAS_TOO_SHORT = { actFunction, weights, bias: [.5] };
            assert.ok(!FcLayer.isFcLayer(BIAS_TOO_LONG));
            assert.ok(!FcLayer.isFcLayer(BIAS_TOO_SHORT));

            const WEIGHTS_NO_MATRIX = { actFunction, bias,
                weights: [...weights, [.1]]
            };
            assert.ok(!FcLayer.isFcLayer(WEIGHTS_NO_MATRIX));

            const INVALID_ACT_FUNC = { actFunction: -1, weights, bias };
            assert.ok(!FcLayer.isFcLayer(INVALID_ACT_FUNC));
        });

        it('should return false for wrong data type', () => {
            assert.ok(!FcLayer.isFcLayer(undefined));
            assert.ok(!FcLayer.isFcLayer(null));
            assert.ok(!FcLayer.isFcLayer(false));
            assert.ok(!FcLayer.isFcLayer(300.5));
            assert.ok(!FcLayer.isFcLayer('string'));
            assert.ok(!FcLayer.isFcLayer([1,2,3,4]));
            assert.ok(!FcLayer.isFcLayer({name:'Hello',age:12}));
        });
    });

    it('getNumOfIn_Output()', () => {
        const ACTUAL1 = FcLayer.getNumOfIn_Output(TEST_LAYER);
        assert.strictEqual(ACTUAL1.input, 2);
        assert.strictEqual(ACTUAL1.output,2);

        const GENERATED = FcLayer.create(4,3,2);
        const ACTUAL2   = FcLayer.getNumOfIn_Output(GENERATED);
        assert.strictEqual(ACTUAL2.input, 4);
        assert.strictEqual(ACTUAL2.output,3);
    });

    it('create()', () => {
        it('should create valid vectors', () => {
            assert.ok(FcLayer.isFcLayer(FcLayer.create(2,3,ActFunction.EActFunction.Linear)));
            assert.ok(FcLayer.isFcLayer(FcLayer.create(3,5,ActFunction.EActFunction.Sigmoid)));
            assert.ok(FcLayer.isFcLayer(FcLayer.create(1,1,ActFunction.EActFunction.HardTanh)));
        });
    });

    it('calc()', () => {
        assert.deepStrictEqual(FcLayer.calc(TEST_LAYER, INPUT), OUTPUT);
    });

    it('train()', () => {
        const ACTUAL = FcLayer.train(TEST_LAYER, INPUT, ERROR, .1);
        assert.deepStrictEqual(ACTUAL.delta, ERROR);
        assert.deepStrictEqual(ACTUAL.layer.bias, [ 0.47, 0.32999999999999996 ]);
        assert.deepStrictEqual(ACTUAL.layer.weights, [ [ 0.17, 0.32999999999999996 ], [ 0.57, 0.13 ] ]);
    });
});