/**
 * Error thrown when an assertion fails.
 */
export class AssertionError extends Error {
  /** @param message The error message. */
  constructor(message: string) {
    super(message);
    this.name = "AssertionError";
  }
}

/**
 * Make an assertion.
 *
 * Usage
 * @example
 * ```ts
 * const value: boolean = Math.random() <= 0.5;
 * assert(value, "value is greater than than 0.5!");
 * value // true
 * ```
 *
 * @param expr The expression to test.
 * @param msg The optional message to display if the assertion fails.
 * @throws an {@link Error} if `expression` is not truthy.
 *
 * @copyright Trevor Manz 2026
 * @license MIT
 * @see {@link https://github.com/manzt/manzt/blob/HASH/utils/assert_verbose.ts}
 */
export function assert(expr: unknown, msg = ""): asserts expr {
	if (!expr) throw new AssertionError(msg);
}
