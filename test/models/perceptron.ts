import { getFloat, setSeed } from '../../src/helper/random';
import { deepClone } from '../../src/helper/clone';
import * as e from '../../src/models/error';
import * as p from '../../src/models/perceptron';

// -----------------------------------------------------------------------------

const errFunc = e.Error.meanSquared;
const numOfSamples = 100;

const initError = 0.1;
const learnError = 0.2;

const learnRate = 0.1;

// -----------------------------------------------------------------------------

// TODO: create real example on paper !!!!

setSeed(1);

const N = [...Array(numOfSamples)].map(() => 1);

const input = N.map(() => [getFloat(-5, 5), getFloat(-5, 5)]);
const expected = input.map(([x,y]) => y > 0 ? 1 : 0);

const initialPerceptron = p.create(2);

let perceptron = deepClone(initialPerceptron);
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

console.log('train first perceptron with a learn rate of', learnRate, 'until error is below', learnError);

perceptron = deepClone(initialPerceptron);
actual = p.calcBatch(perceptron, input);
i = 0;

while (e.calc(actual, expected, errFunc) > learnError) {
    for (let i = 0, ie = input.length; i < ie; i++) {
        perceptron = p.train(perceptron, input[i], expected[i], learnRate);
    }
    actual = p.calcBatch(perceptron, input);
    i++;
}

console.log('took', i, 'epochs to get a good enough perceptron\n');

// -----------------------------------------------------------------------------

console.log('train first perceptron again with batch learning');

perceptron = deepClone(initialPerceptron);
actual = p.calcBatch(perceptron, input);
i = 0;

while (e.calc(actual, expected, errFunc) > learnError) {
    perceptron = p.trainBatch(perceptron, input, expected, learnRate);
    actual = p.calcBatch(perceptron, input);
    i++;
}

console.log('took', i, 'epochs to get a good enough perceptron');