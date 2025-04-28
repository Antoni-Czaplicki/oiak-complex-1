import math
import timeit


# ======================== Manual Traditional Implementation ========================
def manual_complex_add(z1_real, z1_imag, z2_real, z2_imag):
    return z1_real + z2_real, z1_imag + z2_imag


def manual_complex_mul(z1_real, z1_imag, z2_real, z2_imag):
    real = z1_real * z2_real - z1_imag * z2_imag
    imag = z1_real * z2_imag + z1_imag * z2_real
    return real, imag


# ======================== Traditional Complex (Python Built-in) ========================
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
def complex_to_radix2j(z, precision=8):
    real_part = int(z.real)
    imag_part = int(z.imag / 2)

    def to_base_neg4(n):
        digits = []
        while n != 0:
            n, rem = divmod(n, -4)
            if rem < 0:
                n += 1
                rem -= -4
            digits.append(rem)
        return digits[::-1] or [0]

    real_digits = to_base_neg4(real_part)
    imag_digits = to_base_neg4(imag_part)

    max_len = max(len(real_digits), len(imag_digits))
    interleaved = []
    for i in range(max_len):
        interleaved.append(real_digits[i] if i < len(real_digits) else 0)
        interleaved.append(imag_digits[i] if i < len(imag_digits) else 0)
    return interleaved[:precision]


def radix2j_to_complex(digits):
    real = 0
    imag = 0
    for i, d in enumerate(reversed(digits)):
        weight = (-4) ** (i // 2)
        if i % 2 == 0:
            real += d * weight
        else:
            imag += d * weight * 2
    return complex(real, imag)


def radix2j_add(a_digits, b_digits):
    z1 = radix2j_to_complex(a_digits)
    z2 = radix2j_to_complex(b_digits)
    return complex_to_radix2j(z1 + z2)


def radix2j_mul(a_digits, b_digits):
    z1 = radix2j_to_complex(a_digits)
    z2 = radix2j_to_complex(b_digits)
    return complex_to_radix2j(z1 * z2)


# ======================== Implementation Core ========================
class ComplexSystems:
    @staticmethod
    # Manual Traditional Implementation
    def manual_add(z1, z2):
        return z1[0] + z2[0], z1[1] + z2[1]

    @staticmethod
    def manual_mul(z1, z2):
        return z1[0] * z2[0] - z1[1] * z2[1], z1[0] * z2[1] + z1[1] * z2[0]

    # Radix-j−1 Arithmetic
    @staticmethod
    def jminus1_add(a, b):
        max_len = max(len(a), len(b))
        a += [0] * (max_len - len(a))
        b += [0] * (max_len - len(b))
        result = []
        carry1 = carry2 = 0
        for i in range(max_len):
            sum_digit = a[i] + b[i] + carry1
            carry1 = 0
            while sum_digit < 0:
                sum_digit += 2
                carry1 -= 1
            while sum_digit >= 2:
                sum_digit -= 2
                carry1 += 1
            result.append(sum_digit)
            carry1, carry2 = carry2, carry1
        while carry1 or carry2:
            sum_digit = carry1
            carry1 = 0
            while sum_digit < 0:
                sum_digit += 2
                carry1 -= 1
            while sum_digit >= 2:
                sum_digit -= 2
                carry1 += 1
            result.append(sum_digit)
            carry1, carry2 = carry2, carry1
        return result

    @staticmethod
    def jminus1_mul(a, b):
        partials = []
        for i, digit in enumerate(b):
            partial = [0] * i + [digit * d for d in a]
            partials.append(partial)
        result = []
        for p in partials:
            result = ComplexSystems.jminus1_add(result, p)
        return result


# ======================== Benchmark Setup ========================
def full_benchmark():
    z = complex(3, 4)
    z1, z2 = complex(3, 4), complex(2, 5)

    # Traditional Native
    native_add = timeit.timeit(lambda: z1 + z2, number=100000)
    native_mul = timeit.timeit(lambda: z1 * z2, number=100000)
    native_conv = timeit.timeit(lambda: complex(3, 4), number=100000)

    # Manual Traditional
    manual_conv = timeit.timeit(lambda: (3.0, 4.0), number=100000)
    manual_add = timeit.timeit(lambda: ComplexSystems.manual_add((3, 4), (2, 5)), number=100000)
    manual_mul = timeit.timeit(lambda: ComplexSystems.manual_mul((3, 4), (2, 5)), number=100000)

    # Radix-j√2
    jsqrt2_conv = timeit.timeit(lambda: complex_to_jsqrt2(z), number=1000)
    jsqrt2_add_t = timeit.timeit(
        lambda: jsqrt2_add(*complex_to_jsqrt2(z1), *complex_to_jsqrt2(z2)),
        number=1000
    )
    jsqrt2_mul_t = timeit.timeit(
        lambda: jsqrt2_mul(*complex_to_jsqrt2(z1), *complex_to_jsqrt2(z2)),
        number=1000
    )

    # Radix-j−1
    jm1 = complex_to_jminus1(z1)
    jm2 = complex_to_jminus1(z2)
    jminus1_conv = timeit.timeit(lambda: complex_to_jminus1(z1), number=100)
    jminus1_add_t = timeit.timeit(lambda: ComplexSystems.jminus1_add(jm1, jm2), number=100)
    jminus1_mul_t = timeit.timeit(lambda: ComplexSystems.jminus1_mul(jm1, jm2), number=10)

    # Radix-2j
    r2j_conv = timeit.timeit(lambda: complex_to_radix2j(z1), number=1000)
    r2j_add_t = timeit.timeit(lambda: radix2j_add(*map(complex_to_radix2j, [z1, z2])), number=1000)
    r2j_mul_t = timeit.timeit(lambda: radix2j_mul(*map(complex_to_radix2j, [z1, z2])), number=1000)

    # Formatting
    def fmt(time, ops):
        return f"{(time * 1e6) / ops:.2f} μs" if time else "N/A"

    print(f"{'System':<12} {'Convert':<12} {'Add':<12} {'Multiply':<12}")
    print(f"{'Native':<12} {fmt(native_conv, 100000):<12} {fmt(native_add, 100000):<12} {fmt(native_mul, 100000):<12}")
    print(f"{'Manual':<12} {fmt(manual_conv, 100000):<12} {fmt(manual_add, 100000):<12} {fmt(manual_mul, 100000):<12}")
    print(f"{'Radix-j√2':<12} {fmt(jsqrt2_conv, 1000):<12} {fmt(jsqrt2_add_t, 1000):<12} {fmt(jsqrt2_mul_t, 1000):<12}")
    print(f"{'Radix-j−1':<12} {fmt(jminus1_conv, 100):<12} {fmt(jminus1_add_t, 100):<12} {fmt(jminus1_mul_t, 10):<12}")
    print(f"{'Radix-2j':<12} {fmt(r2j_conv, 1000):<12} {fmt(r2j_add_t, 1000):<12} {fmt(r2j_mul_t, 1000):<12}")


if __name__ == "__main__":
    full_benchmark()
