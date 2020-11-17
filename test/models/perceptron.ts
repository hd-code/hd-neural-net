import { getFloat, setSeed } from '../../src/helper/random';
import * as e from '../../src/models/error';
import * as p from '../../src/models/perceptron';

// -----------------------------------------------------------------------------

const errFunc = e.Error.meanSquared;
const numOfSamples = 10000;

const initError = 0.1;
const learnError = 0.01;

const learnRate = 0.01;

// -----------------------------------------------------------------------------

setSeed(1);

const N = [...Array(numOfSamples)].map(() => 1);

const input = N.map(() => [getFloat(-5, 5), getFloat(-5, 5)]);
const expected = input.map(([x,y]) => x > 0 ? 1 : 0);

let perceptron = p.create(2);
let actual;
let i = 0;

console.log('init perceptrons until one is found with an error rate of', initError);

do {
    perceptron = p.create(2);
    actual = p.calcBatch(perceptron, input);
    i++;
} while (e.calc(actual, expected, errFunc) > initError);

console.log('took', i, 'tries to get a good enough perceptron\n');

// -----------------------------------------------------------------------------

console.log('train found perceptron with a learn rate of', learnRate, 'until error is below', learnError);

i = 0;

while (e.calc(actual, expected, errFunc) > learnError) {
    perceptron = p.create(2);
    for (let i = 0, ie = input.length; i < ie; i++) {
        perceptron = p.train(perceptron, input[i], expected[i], learnRate);
    }
    actual = p.calcBatch(perceptron, input);
    i++;
}

console.log('took', i, 'epochs to get a good enough perceptron');