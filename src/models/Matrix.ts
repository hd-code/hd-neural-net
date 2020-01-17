import { isArray } from '../lib/hd-helper';
import * as Vector from './Vector'

// -----------------------------------------------------------------------------

export type TMatrix = Vector.TVector[]

export function isMatrix(matrix: any): matrix is TMatrix {
    if (!isArray(matrix, Vector.isVector))
        return false;

    const rowLength = matrix[0].length;

    for (let i = 1, ie = matrix.length; i < ie; i++) {
        if (rowLength !== matrix[i].length)
            return false;
    }

    return true;
}

export function createRandom(rows: number, columns: number): TMatrix {
    let result: TMatrix = [];
    for (var i = 0; i < rows; i++) {
        result.push(Vector.createRandom(columns));
    }
    return result;
}

export function transpose(matrix: TMatrix): TMatrix {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
}

export function add(matrix1: TMatrix, matrix2: TMatrix): TMatrix {
    return matrix1.map((_,i) => Vector.add(matrix1[i], matrix2[i]));
}

export function sub(matrix1: TMatrix, matrix2: TMatrix): TMatrix {
    return matrix1.map((_,i) => Vector.sub(matrix1[i], matrix2[i]));
}

export function mulVector(vector: Vector.TVector, matrix: TMatrix): Vector.TVector {
    const tMatrix = transpose(matrix);
    return tMatrix.map(row => Vector.dot(vector, row));
}

export function mul(leftMatrix: TMatrix, rightMatrix: TMatrix): TMatrix {
    return leftMatrix.map(row => mulVector(row, rightMatrix));
}

export function scale(scale: number, matrix: TMatrix): TMatrix {
    return matrix.map(row => Vector.scale(scale, row));
}