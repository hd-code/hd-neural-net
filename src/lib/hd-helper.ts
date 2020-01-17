/*! hd-helper v1.0.0 | Hannes Dröse | https://github.com/hd-code/hd-helper | MIT */

// ---------------------------------- General ----------------------------------

/**
 * Deep-clones the passed object.
 * 
 * Attention: If the object contains any functions, they will be removed. 
 */
export function deepClone<T>(original: T): T {
    return JSON.parse(JSON.stringify(original));
}

/**
 * Flattens a passed `n`-dimensional array. So after flattening the resulting
 * array as `n - 1` dimensions.
 * 
 * This is a pure function, so the original array will not be altered.
 */
export function flattenArray<T>(original: T[][]): T[] {
    return original.reduce((result, item) => result.concat(item), []);
}

// -------------------------------- Type Guards --------------------------------

/** Type guard to check if a value is a `boolean`. */
export function isBool(bool: any): bool is boolean {
    return typeof bool === 'boolean';
}

/** Type guard to check if a value is a `number`. */
export function isNumber(num: any): num is number {
    return typeof num === 'number';
}

/** Type guard to check if a value is a `string`. */
export function isString(str: any): str is string {
    return typeof str === 'string';
}

/** Type guard to check if a value is a JS Date type. */
export function isDate(date: any): date is Date {
    return date instanceof Date;
}

/**
 * Type guard to check if a value is an `array`.
 * 
 * Optional: You can pass a type guard (`function`) as a second argument. This 
 * will perform a type check on each element of the array. If the type check
 * fails on any element, the function will return false;
 */
export function isArray<T>(a: any, typeGuard?: (e:any) => e is T): a is T[] {
    if (!Array.isArray(a)) return false;
    if (!typeGuard) return true;

    for (let i = 0, ie = a.length; i < ie; i++) {
        if (!typeGuard(a[i]))
            return false;
    }

    return true;
}

/**
 * Type guard to check if a value is an object. If the value is `null`, the type 
 * guard will reject the value. However, just an empty object (like this: `{}`) 
 * is valid.
 * 
 * If you want to check the object for specific keys, use `isKeyOfObject()`.
 */
export function isObject(obj:any): obj is object {
    return typeof obj === 'object' && obj !== null && !isArray(obj);
}

/**
 * Type guard to check if a value is an `object` and also contains the specified
 * `key`. (`{ key: ... }`)
 * 
 * Optional: You can pass a type guard as a third argument to this function. If 
 * the given key is found, the value associated with that key is then 
 * type-checked by the type guard.
 */
// TODO: find out if there is a better function signature?
export function isKeyOfObject<T,U>(obj: any, key: keyof T, keyTypeGuard?: (e:any) => e is U): obj is T {
    return typeof obj === 'object' && obj !== null && (obj as T)[key] !== undefined
        && (!keyTypeGuard || keyTypeGuard(obj[key]));
}