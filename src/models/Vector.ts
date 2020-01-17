import { isArray, isNumber } from '../lib/hd-helper';

// -----------------------------------------------------------------------------

export type TVector = number[]

export function isVector(vec: any): vec is TVector {
    return isArray(vec, isNumber);
}

export function createRandom(length: number): TVector {
    let result: TVector = []
    for (let i = 0; i < length; i++) result.push(Math.random())
    return result
}

export function add(vec1: TVector, vec2: TVector): TVector {
    return vec1.map((_, i) => vec1[i] + vec2[i]);
}

export function sub(vec1: TVector, vec2: TVector): TVector {
    return vec1.map((_, i) => vec1[i] - vec2[i]);
}

export function mul(vec1: TVector, vec2: TVector): TVector {
    return vec1.map((_, i) => vec1[i] * vec2[i]);
}

export function scale(scale: number, vector: TVector): TVector {
    return vector.map(value => value * scale)
}

export function dot(vec1: TVector, vec2: TVector): number {
    return vec1.reduce((sum, _, i) => sum + vec1[i] * vec2[i], 0);
}