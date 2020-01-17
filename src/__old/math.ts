/* -------------------------------- Vectors --------------------------------- */

export function addVec(vec1: number[], vec2: number[]): number[] {
    return vec1.map((_, i) => vec1[i] + vec2[i])
}

export function subVec(vec1: number[], vec2: number[]): number[] {
    return vec1.map((_, i) => vec1[i] - vec2[i])
}

export function mulVec(vec1: number[], vec2: number[]): number[] {
    return vec1.map((_, i) => vec1[i] * vec2[i])
}

export function scaleVec(scale: number, vector: number[]): number[] {
    return vector.map(value => value * scale)
}

export function createRandomVector(length: number): number[] {
    let result: number[] = []
    for (let i = 0; i < length; i++) result.push(Math.random())
    return result
}

/* -------------------------------- Matrizes -------------------------------- */

export function transposeMatrix(matrix: number[][]): number[][] {
    return matrix[0].map((_, i) => matrix.map(row => row[i]))
}

export function addMatrix(matrix1: number[][], matrix2: number[][]): number[][] {
    return matrix1.map((_,i) => addVec(matrix1[i], matrix2[i]))
}

export function subMatrix(matrix1: number[][], matrix2: number[][]): number[][] {
    return matrix1.map((_,i) => subVec(matrix1[i], matrix2[i]))
}

export function scaleMatrix(scale: number, matrix: number[][]): number[][] {
    return matrix.map(row => scaleVec(scale, row))
}

export function createRandomMatrix(rows: number, columns: number): number[][] {
    let result: number[][] = []
    for (var i = 0; i < rows; i++) result.push(createRandomVector(columns))
    return result
}

/* --------------------------- Vectors & Matrizes --------------------------- */

export function mulVecMat(vector: number[], matrix: number[][]): number[] {
    return matrix[0].map((_, i) => {
        return matrix.reduce((result, _, j) => {
            return result + vector[j] * matrix[j][i]
        }, 0)
    })
}

/* -------------------------------- Rounding -------------------------------- */

/** 
 * Rounds a number to the desired precision.
 * @param number    The number to be rounded
 * @param precision The number of digits after comma. 
 *      Positive numbers will round after comma, thus making it more precise.
 *      Negative numbers will round before the comma, thus lessening the precision.
 */
export function round(num: number, precision: number): number {
    let factor = 10 ** precision
    return Math.round(num * factor) / factor
}

export function roundVector(vector: number[], precision: number): number[] {
    return vector.map(num => round(num, precision))
}

export function roundMatrix(matrix: number[][], precision: number): number[][] {
    return matrix.map(row => roundVector(row, precision))
}