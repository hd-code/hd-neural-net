/** creates clone of passed object. attention: removes contained functions! */
export function deepClone<T>(original: T): T {
    return JSON.parse(JSON.stringify(original))
}

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

/* ----------------------------- Type Checking ------------------------------ */

export function isNumber(n:any): n is number {
    return typeof n === 'number'
}

export function isString(s:any): s is string {
    return typeof s === 'string'
}

export function isBool(b:any): b is boolean {
    return typeof b === 'boolean'
}

/** checks if an object a is an array of the type specified by the type guard */
export function isArrayOf<T>(a: any, typeGuard: (e:any) => e is T): a is T[] {
    if (!Array.isArray(a)) return false

    for (let i = 0, ie = a.length; i < ie; i++) {
        if (!typeGuard(a[i]))
            return false
    }

    return true
}

export function isVector(v: any): v is number[] {
    return isArrayOf(v, isNumber)
} 

export function isMatrix(m: any): m is number[][] {
    return isArrayOf(m, isVector)
}