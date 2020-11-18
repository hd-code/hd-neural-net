import { sum } from '../../lib/math/vector';

// -----------------------------------------------------------------------------

export enum Error { absolute, meanSquared, squared }

export function calc<T extends number|number[]>(actual: T, expected: T, errorFunc: Error = 1): number {
    const x = actual instanceof Array ? actual : [actual];
    const y = expected instanceof Array ? expected : [expected];
    const errFunc = errorFunc === Error.absolute ? absolute : squared;
    const errors = mapError(errFunc, x as number[], y as number[]);
    const errSum = sum(errors);
    return errorFunc === Error.meanSquared ? errSum / errors.length : errSum;
}

// -----------------------------------------------------------------------------

function mapError(errFunc: (x: number, y: number) => number, x: number[], y: number[]): number[] {
    const result = [];
    for (let i = 0, ie = x.length; i < ie; i++) {
        result.push(errFunc(x[i], y[i]));
    }
    return result;
}

// -----------------------------------------------------------------------------

function absolute(actual: number, expected: number): number {
    return Math.abs(actual - expected);
}

function squared(actual: number, expected: number): number {
    const diff = actual - expected;
    return diff * diff;
}