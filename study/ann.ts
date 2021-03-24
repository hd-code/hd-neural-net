import * as Matrix from '../src/helper/matrix';
import * as Vector from '../src/helper/vector';

// -----------------------------------------------------------------------------

const cases = [{
    name: 'XOR & AND',
    ann: [{
        bias: [0.2, 0.8],
        weights: [[0.2,0.6],[0.7,0.3]]
    }, {
        bias: [-0.1, -0.6],
        weights: [[0.1,0.1],[-0.4,0.7]]
    }],
    data: [
       { input: [0,0], expect: [0,0] },
       { input: [0,1], expect: [0,1] },
       { input: [1,0], expect: [0,1] },
       { input: [1,1], expect: [1,0] },
    ],
}];

cases.forEach(main);

function main({name, ann, data}: typeof cases[0]) {
    console.log('\n\n-----------------------------------');
    console.log(name);
    console.log(ann);
    data.forEach(data => handleOneDataSet(ann, data));
}

// -----------------------------------------------------------------------------

function handleOneDataSet(ann: {bias: number[], weights: number[][]}[], data: {input: number[], expect: number[]}) {
    console.log('\n', data);

    const layerResults: any[] = [{output: data.input}];
    for (let i = 0, ie = ann.length; i < ie; i++) {
        layerResults.push(calcLayer(ann[i], layerResults[i].output));
    }
    
    const output = layerResults[layerResults.length - 1].output;
    const err = error(output, data.expect);
    
    const errDeriv = errorDeriv(output, data.expect);

    const index = layerResults.length - 1;
    layerResults[index]['deltaB'] = Vector.mul(errDeriv, layerResults[index].actDeriv);
    layerResults[index]['deltaW'] = mulVecAndTransVec(layerResults[index]['deltaB'], layerResults[index - 1].output);

    for (let i = index-1; i > 0; i--) {
        layerResults[i]['deltaB'] = Vector.mul(
            Vector.mulMatrix(
                layerResults[i+1]['deltaB'],
                Matrix.transpose(ann[i-1].weights),
            ),
            layerResults[i].actDeriv,
        );
        layerResults[i]['deltaW'] = mulVecAndTransVec(layerResults[i]['deltaB'], layerResults[i - 1].output);
    }
    
    console.log('error:', err);
    console.log(layerResults);
}

// -----------------------------------------------------------------------------

function calcLayer(layer: {bias: number[], weights: number[][]}, input: number[]) {
    const weighted = Matrix.mulVector(layer.weights, input);
    const biased = Vector.add(layer.bias, weighted);
    const output = activation(biased);
    const actDeriv = activationDeriv(biased);

    return { output, actDeriv };
}

function mulVecAndTransVec(vector: number[], transposed: number[]): number[][] {
    return vector.map(x => transposed.map(y => x * y));
}

// -----------------------------------------------------------------------------

function activation(x: number[]): number[] {
    return x.map(leakyReLin);
}

function activationDeriv(x: number[]): number[] {
    return x.map(leakyReLinDeriv);
}

function leakyReLin(x: number): number {
    return x > 0 ? x : 0.01 * x;
}

function leakyReLinDeriv(x: number): number {
    return x > 0 ? 1 : 0.01;
}

// -----------------------------------------------------------------------------

function error(actual: number[], expected: number[]): number {
    const squaredErrors = actual.map((_, i) => squared(actual[i], expected[i]));
    const sum = squaredErrors.reduce((sum, err) => sum + err, 0);
    return sum / actual.length;
}

function errorDeriv(actual: number[], expected: number[]): number[] {
    const squaredDerivs = actual.map((_, i) => squaredDeriv(actual[i], expected[i]));
    return squaredDerivs.map(x => x / actual.length);
}

function squared(actual: number, expected: number): number {
    return (actual - expected) ** 2;
}

function squaredDeriv(actual: number, expected: number): number {
    return 2 * (actual - expected);
}