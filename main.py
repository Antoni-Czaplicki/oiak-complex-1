import math
import random


# ======================== Utilities ========================
def rebase(num: int, base: int) -> tuple[list[int], bool]:
    "converts an integer from base ten to given integer base"
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    values, is_negative = [], False
    if abs(base) < 2 or abs(base) > len(digits):
        raise ValueError("invalid base")
    if num < 0 and base > 0: num *= -1; is_negative = True
    while num:
        num, d = divmod(num, base)
        if d < 0: num += 1; d -= base
        values.append(d)
    values.reverse()
    return values, is_negative


def remove_leading_zeros(lst: list[int]) -> list[int]:
    """Removes leading zeros from a list of integers."""
    while lst and lst[0] == 0:
        lst.pop(0)
    return lst if lst else [0]


def remove_trailing_zeros(lst: list[int]) -> list[int]:
    """Removes trailing zeros from a list of integers."""
    while lst and lst[-1] == 0:
        lst.pop()
    return lst if lst else [0]


# ======================== Complex (Python Built-in) ========================
def traditional_add(z1, z2):
    return z1 + z2


def traditional_mul(z1, z2):
    return z1 * z2


# ======================== Radix-j√2 System ========================
def jsqrt2_to_complex(real_digits, imag_digits):
    a = 0
    b = 0
    for i, digit in enumerate(reversed(real_digits)):
        a += digit * (2 ** i)
    for i, digit in enumerate(reversed(imag_digits)):
        b += digit * (2 ** i) * math.sqrt(2)
    return complex(a, b)


def complex_to_jsqrt2(z, precision=8):
    a = z.real
    b = z.imag / math.sqrt(2)
    real_digits = []
    imag_digits = []
    for _ in range(precision):
        real_digits.append(int(a % 2))
        a = (a - real_digits[-1]) // 2
        imag_digits.append(int(b % 2))
        b = (b - imag_digits[-1]) // 2
    return real_digits[::-1], imag_digits[::-1]


def jsqrt2_add(a_real, a_imag, b_real, b_imag):
    z1 = jsqrt2_to_complex(a_real, a_imag)
    z2 = jsqrt2_to_complex(b_real, b_imag)
    return complex_to_jsqrt2(z1 + z2)


def jsqrt2_mul(a_real, a_imag, b_real, b_imag):
    z1 = jsqrt2_to_complex(a_real, a_imag)
    z2 = jsqrt2_to_complex(b_real, b_imag)
    return complex_to_jsqrt2(z1 * z2)


# ======================== Radix-j−1 System ========================
def jminus1_to_complex(digits):
    z = 0
    base = -1 + 1j
    for digit in reversed(digits):
        z = z * base + digit
    return z


def complex_to_jminus1(z, bits=8):
    a = int(z.real)
    b = int(z.imag)
    digits = []
    for _ in range(bits):
        r = (a + b) % 2
        digits.append(r)
        y = (r - a - b) // 2
        x = b + y
        a, b = x, y
    return digits[::-1]


def jminus1_add(a_digits, b_digits):
    # Implementation from paper's carry rules
    max_len = max(len(a_digits), len(b_digits))
    a = a_digits + [0] * (max_len - len(a_digits))
    b = b_digits + [0] * (max_len - len(b_digits))

    result = []
    carry = 0
    for i in range(max_len):
        total = a[i] + b[i] + carry
        if total >= 2:
            result.append(total - 2)
            carry = 1
        elif total < 0:
            result.append(total + 2)
            carry = -1
        else:
            result.append(total)
            carry = 0
    if carry != 0:
        result.append(carry)
    return result


def jminus1_mul(a_digits, b_digits):
    # Partial product generation with carry handling
    partials = []
    for i, digit in enumerate(b_digits):
        partial = [0] * i + [d * digit for d in a_digits]
        partials.append(partial)

    # Summation with carry propagation
    result = []
    for p in partials:
        result = jminus1_add(result, p)

    return result


# ======================== Radix-2j System ========================
def complex_to_radix2j(z: complex) -> tuple[list[int], list[int]]:
    real_part = int(z.real)
    imag_part = math.ceil(z.imag / 2)

    real_part_neg_4 = rebase(real_part, -4)
    imag_part_neg_4 = rebase(imag_part, -4)

    if imag_part_neg_4[1] or real_part_neg_4[1]:
        raise ValueError("Imaginary or real part cannot be negative in radix-2j representation. Something went wrong.")

    result = []

    for i in range(max(len(real_part_neg_4[0]), len(imag_part_neg_4[0]))):
        try:
            result.append(real_part_neg_4[0][-i - 1])
        except IndexError:
            result.append(0)

        try:
            result.append(imag_part_neg_4[0][-i - 1])
        except IndexError:
            result.append(0)

    result.reverse()
    return remove_leading_zeros(result), [2] if z.imag % 2 else []


def radix2j_to_complex(integer_digits, fraction_digits):
    real = 0
    imag = 0
    for i, digit in enumerate(reversed(integer_digits)):
        if i % 2 == 0:
            real += digit * ((-4) ** (i // 2))
        else:
            imag += digit * ((-4) ** (i // 2)) * 2

    for i, digit in enumerate(reversed(fraction_digits)):
        if i % 2 == 0:
            imag += digit * (1 / -4) ** ((i + 2) // 2) * 2
        else:
            real += digit * (1 / -4) ** ((i + 2) // 2)

    return complex(real, imag)


def radix2j_add(a: tuple[list[int], list[int]], b: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
    """
    Add two numbers in radix-2j (quarter-imaginary) system.
    
    Rules:
    - If digit > 3: subtract 4 and carry -1 two places to the left
    - If digit < 0: add 4 and carry +1 two places to the left
    """
    a_int, a_frac = a
    b_int, b_frac = b

    # Pad integer parts to same length, add two to allow for extra carry
    max_int_len = max(len(a_int), len(b_int)) + 4
    a_int_padded = [0] * (max_int_len - len(a_int)) + a_int
    b_int_padded = [0] * (max_int_len - len(b_int)) + b_int

    # Pad fractional parts to same length
    max_frac_len = max(len(a_frac), len(b_frac))
    a_frac_padded = a_frac + [0] * (max_frac_len - len(a_frac))
    b_frac_padded = b_frac + [0] * (max_frac_len - len(b_frac))

    # Combine integer and fractional parts for processing
    # Process from right to left (least significant to most significant)
    combined_a = a_int_padded + a_frac_padded
    combined_b = b_int_padded + b_frac_padded

    result = [0] * len(combined_a)
    carry = [0] * (len(combined_a) + 2)  # Extra space for carries

    # Process from right to left
    for i in range(len(combined_a) - 1, -1, -1):
        # Add digits and any carry from two positions to the right
        total = combined_a[i] + combined_b[i] + carry[i + 2]

        # Apply quarter-imaginary rules
        while total > 3:
            total -= 4
            carry[i] -= 1  # Carry -1 two places to the left

        while total < 0:
            total += 4
            carry[i] += 1  # Carry +1 two places to the left

        result[i] = total

    # Handle any remaining carries to the left of the most significant digit
    extra_digits = []
    for i in range(1, -1, -1):  # Check positions 1 and 0 for leftmost carries
        if carry[i] != 0:
            digit = carry[i]
            while digit > 3:
                digit -= 4
                if i > 0:
                    carry[i - 1] -= 1
                else:
                    extra_digits.insert(0, -1)
            while digit < 0:
                digit += 4
                if i > 0:
                    carry[i - 1] += 1
                else:
                    extra_digits.insert(0, 1)
            if digit != 0:
                extra_digits.append(digit)

    # Split back into integer and fractional parts
    result_int = extra_digits + result[:max_int_len]
    result_frac = result[max_int_len:]

    # Remove leading zeros from integer part
    result_int = remove_leading_zeros(result_int)

    # Remove trailing zeros from fractional part
    result_frac = remove_trailing_zeros(result_frac)

    return result_int, result_frac


def radix2j_mul(a: tuple[list[int], list[int]], b: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
    """
    Multiply two numbers in radix-2j (quarter-imaginary) system using long multiplication.
    
    Algorithm following the long multiplication method:
    1. For each digit in the multiplier (from right to left)
    2. Multiply the multiplicand by that digit, creating a partial product
    3. Shift each partial product by the appropriate number of positions
    4. Add all partial products using radix-2j addition rules
        """
    a_int, a_frac = a
    b_int, b_frac = b

    # Handle special cases
    if (not a_int or all(d == 0 for d in a_int)) and (not a_frac or all(d == 0 for d in a_frac)):
        return [0], []
    if (not b_int or all(d == 0 for d in b_int)) and (not b_frac or all(d == 0 for d in b_frac)):
        return [0], []

    # Combine digits for processing (integer part + fractional part)
    a_combined = a_int + a_frac
    b_combined = b_int + b_frac

    # Track the position of the "decimal point" in the result
    result_frac_len = len(a_frac) + len(b_frac)

    partial_products = []

    # Generate partial products using long multiplication
    for i, b_digit in enumerate(reversed(b_combined)):
        if b_digit == 0:
            # Skip zero digits to avoid unnecessary computation
            continue

        # Create partial product: multiply a_combined by b_digit
        partial = []
        carry = 0

        # Multiply each digit of a by the current digit of b
        for a_digit in reversed(a_combined):
            product = a_digit * b_digit + carry
            partial.append(product)  # Store the raw product temporarily
            carry = 0  # Reset carry for radix-2j (we'll handle normalization later)

        # Reverse to get correct order (most significant first)
        partial.reverse()

        # Shift partial product by i positions to the right (multiply by (2j)^i)
        shifted_partial = partial + [0] * i

        # Convert to (integer, fractional) format
        if result_frac_len >= len(shifted_partial):
            # All digits are in fractional part
            p_int = [0]
            p_frac = [0] * (result_frac_len - len(shifted_partial)) + shifted_partial
        else:
            # Split between integer and fractional parts
            split_pos = len(shifted_partial) - result_frac_len
            p_int = shifted_partial[:split_pos] if split_pos > 0 else [0]
            p_frac = shifted_partial[split_pos:] if split_pos < len(shifted_partial) else []

        partial_products.append((p_int, p_frac))

    # Sum all partial products using radix-2j addition
    if not partial_products:
        return [0], []

    result = partial_products[0]
    for i in range(1, len(partial_products)):
        result = radix2j_add(result, partial_products[i])

    return result


# ======================== Benchmark Setup ========================
def full_benchmark():
    # z = complex(-14, -5)
    # print(z)
    # print(complex_to_radix2j(z))
    # print(radix2j_to_complex(*complex_to_radix2j(z)))

    # Test radix2j addition with the examples from the description
    print("Testing radix-2j addition implementation:")

    z1 = complex(-10, -8)
    z2 = complex(-4, 3)
    expected = z1 + z2

    r2j_z1 = complex_to_radix2j(z1)
    r2j_z2 = complex_to_radix2j(z2)
    r2j_result = radix2j_add(r2j_z1, r2j_z2)
    result_complex = radix2j_to_complex(*r2j_result)

    print(f"Test 1: {z1} + {z2} = {expected}")
    print(f"Radix-2j: {r2j_z1} + {r2j_z2} = {r2j_result}")
    print(f"Expected radix-2j result: {complex_to_radix2j(expected)}")
    print(f"Back to complex: {result_complex}")
    print(f"Correct: {abs(result_complex - expected) < 1e-10}")
    print()

    # Example 2: (3-4i) + (1-8i) = (4-12i)
    # In radix-2j: 1023 + 1001 = 12320
    z3 = complex(3, -4)
    z4 = complex(1, -8)
    expected2 = complex(4, -12)

    r2j_z3 = complex_to_radix2j(z3)
    r2j_z4 = complex_to_radix2j(z4)
    r2j_result2 = radix2j_add(r2j_z3, r2j_z4)
    result_complex2 = radix2j_to_complex(*r2j_result2)

    print(f"Test 2: {z3} + {z4} = {expected2}")
    print(f"Radix-2j: {r2j_z3} + {r2j_z4} = {r2j_result2}")
    print(f"Back to complex: {result_complex2}")
    print(f"Correct: {abs(result_complex2 - expected2) < 1e-10}")
    print()

    # Additional random tests
    print("Random tests:")
    incorrect_count = 0
    for i in range(50000):
        z_a = complex(random.randint(-1000, 1000), random.randint(-1000, 1000))
        z_b = complex(random.randint(-1000, 1000), random.randint(-1000, 1000))
        expected = z_a * z_b

        try:
            r2j_a = complex_to_radix2j(z_a)
            r2j_b = complex_to_radix2j(z_b)
            r2j_mul = radix2j_mul(r2j_a, r2j_b)
            result = radix2j_to_complex(*r2j_mul)

            print(f"{z_a} * {z_b} = {expected}")
            print(f"Radix-2j result: {result}")
            print(f"Correct: {abs(result - expected) < 1e-10}")
            if abs(result - expected) >= 1e-10:
                incorrect_count += 1
            print()
        except Exception as e:
            print(f"Error with {z_a} * {z_b}: {e}")
            print()

    print(f"Total incorrect results: {incorrect_count} out of {i + 1} random tests.")
    print(f"Success rate: {((i + 1) - incorrect_count) / (i + 1) * 100:.2f}%")

    # z1, z2 = complex(3, 4), complex(2, 5)
    #
    # # Traditional Native
    # native_add = timeit.timeit(lambda: z1 + z2, number=100000)
    # native_mul = timeit.timeit(lambda: z1 * z2, number=100000)
    # native_conv = timeit.timeit(lambda: complex(3, 4), number=100000)
    #
    # # Manual Traditional
    # manual_conv = timeit.timeit(lambda: (3.0, 4.0), number=100000)
    # manual_add = timeit.timeit(lambda: ComplexSystems.manual_add((3, 4), (2, 5)), number=100000)
    # manual_mul = timeit.timeit(lambda: ComplexSystems.manual_mul((3, 4), (2, 5)), number=100000)
    #
    # # Radix-j√2
    # jsqrt2_conv = timeit.timeit(lambda: complex_to_jsqrt2(z), number=1000)
    # jsqrt2_add_t = timeit.timeit(
    #     lambda: jsqrt2_add(*complex_to_jsqrt2(z1), *complex_to_jsqrt2(z2)),
    #     number=1000
    # )
    # jsqrt2_mul_t = timeit.timeit(
    #     lambda: jsqrt2_mul(*complex_to_jsqrt2(z1), *complex_to_jsqrt2(z2)),
    #     number=1000
    # )
    #
    # # Radix-j−1
    # jm1 = complex_to_jminus1(z1)
    # jm2 = complex_to_jminus1(z2)
    # jminus1_conv = timeit.timeit(lambda: complex_to_jminus1(z1), number=100)
    # jminus1_add_t = timeit.timeit(lambda: ComplexSystems.jminus1_add(jm1, jm2), number=100)
    # jminus1_mul_t = timeit.timeit(lambda: ComplexSystems.jminus1_mul(jm1, jm2), number=10)
    #
    # # Radix-2j
    # r2j_conv = timeit.timeit(lambda: complex_to_radix2j(z1), number=1000)
    # r2j_add_t = timeit.timeit(lambda: radix2j_add(*map(complex_to_radix2j, [z1, z2])), number=1000)
    # r2j_mul_t = timeit.timeit(lambda: radix2j_mul(*map(complex_to_radix2j, [z1, z2])), number=1000)
    #
    # # Formatting
    # def fmt(time, ops):
    #     return f"{(time * 1e6) / ops:.2f} μs" if time else "N/A"
    #
    # print(f"{'System':<12} {'Convert':<12} {'Add':<12} {'Multiply':<12}")
    # print(f"{'Native':<12} {fmt(native_conv, 100000):<12} {fmt(native_add, 100000):<12} {fmt(native_mul, 100000):<12}")
    # print(f"{'Manual':<12} {fmt(manual_conv, 100000):<12} {fmt(manual_add, 100000):<12} {fmt(manual_mul, 100000):<12}")
    # print(f"{'Radix-j√2':<12} {fmt(jsqrt2_conv, 1000):<12} {fmt(jsqrt2_add_t, 1000):<12} {fmt(jsqrt2_mul_t, 1000):<12}")
    # print(f"{'Radix-j−1':<12} {fmt(jminus1_conv, 100):<12} {fmt(jminus1_add_t, 100):<12} {fmt(jminus1_mul_t, 10):<12}")
    # print(f"{'Radix-2j':<12} {fmt(r2j_conv, 1000):<12} {fmt(r2j_add_t, 1000):<12} {fmt(r2j_mul_t, 1000):<12}")


if __name__ == "__main__":
    full_benchmark()
