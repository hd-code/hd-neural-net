/** creates clone of passed object. doesn't work if object contains functions */
export function deepClone<T>(original: T): T {
    return JSON.parse(JSON.stringify(original))
}

/* ----------------------------- Type Checking ------------------------------ */

export function isNumber(n:any): n is number {
    return typeof n === 'number'
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