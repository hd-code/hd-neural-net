export function isNumber(n:any) :n is number {
    return typeof n === 'number'
}

export function isVector(v :any) :v is number[] {
    return isArrayOf(v, isNumber)
} 

export function isMatrix(m :any) :m is number[][] {
    return isArrayOf(m, isVector)
}

/** checks if an element is a array of the type specified by the type guard */
export function isArrayOf<T>(a :any, typeGuard :(e:any) => e is T) :a is T[] {
    if (!Array.isArray(a)) return false

    for (let i = 0, ie = a.length; i < ie; i++) {
        if (!typeGuard(a[i]))
            return false
    }

    return true
}