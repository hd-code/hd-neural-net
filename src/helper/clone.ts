/**
 * Clones an object, array or primitive value. It creates shallow clones only.
 * So, nested arrays or objects are copied only by reference. Changes to the
 * nested elements in the copy will effect the original and vice versa.
 * 
 * If deep clones are needed, use `deepClone()`. However, deep cloning is a lot
 * slower.
 *
 * _Attention_: Classes are not correctly cloned.
 */
export function clone<T>(original: T): T {
    if (Array.isArray(original)) return (original as any).slice();
    if (original !== null && typeof original === 'object') return {...original};
    return original;
}

/**
 * Clones a passed object, array or primitive value. It creates deep clones.
 * So nested arrays or objects will be copied as well. That means that the
 * original and the clone are completely independent from each other.
 * 
 * _Attention_: Classes are not correctly cloned.
 */
export function deepClone<T>(original: T): T { // TODO: Find solution for Classes and Dates
    if (Array.isArray(original)) {
        const result: any = [];
        for (let i = 0, ie = original.length; i < ie; i++) {
            result[i] = deepClone(original[i]);
        }
        return result;
    }

    if (original !== null && typeof original === 'object') {
        const result: any = {};
        for (const key in original) {
            result[key] = deepClone(original[key]);
        }
        return result;
    }

    return original;
}