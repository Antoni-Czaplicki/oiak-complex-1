import math
import random
from abc import ABC, abstractmethod
from typing import Any, Tuple, List


# ======================== Utilities ========================
def rebase(num: int, base: int) -> tuple[list[int], bool]:
    """Converts an integer from base ten to given integer base"""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    values, is_negative = [], False
    if abs(base) < 2 or abs(base) > len(digits):
        raise ValueError("invalid base")
    if num < 0 and base > 0:
        num *= -1
        is_negative = True
    while num:
        num, d = divmod(num, base)
        if d < 0:
            num += 1
            d -= base
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


# ======================== Complex Number System Interface ========================
class ComplexNumberSystem(ABC):
    """Abstract base class for complex number systems"""

    @abstractmethod
    def from_complex(self, z: complex) -> Any:
        """Convert from Python complex to this system's representation"""
        pass

    @abstractmethod
    def to_complex(self, representation: Any) -> complex:
        """Convert from this system's representation to Python complex"""
        pass

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Add two numbers in this system"""
        pass

    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """Multiply two numbers in this system"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the complex number system"""
        pass


# ======================== Traditional Complex System ========================
class TraditionalComplexSystem(ComplexNumberSystem):
    """Traditional Python complex number system"""

    def from_complex(self, z: complex) -> complex:
        return z

    def to_complex(self, representation: complex) -> complex:
        return representation

    def add(self, a: complex, b: complex) -> complex:
        return a + b

    def multiply(self, a: complex, b: complex) -> complex:
        return a * b

    @property
    def name(self) -> str:
        return "Traditional Complex"


# ======================== Generic Radix-Based System ========================
class GenericRadixComplexSystem(ComplexNumberSystem):
    """
    Generic radix-based complex number system that can be parameterized.

    This class provides a flexible framework for implementing various radix-based
    complex number systems by specifying conversion parameters and arithmetic rules.
    """

    def __init__(self,
                 base: int):
        """
        Initialize a generic radix-based complex number system.

        Args:
            base: The base for conversion (e.g., -4 for radix-2j, 2 for radix-j√2)
        """
        self.base = base

    def from_complex(self, z: complex) -> Tuple[List[int], List[int]]:
        """Convert from Python complex to this system's representation"""
        real_part = int(z.real)
        imag_part = math.ceil(z.imag / 2)

        real_part_rebased = rebase(real_part, self.base)
        imag_part_rebased = rebase(imag_part, self.base)

        if imag_part_rebased[1] or real_part_rebased[1]:
            raise ValueError("Imaginary or real part cannot be negative in radix-Xj representation")

        result = []
        for i in range(max(len(real_part_rebased[0]), len(imag_part_rebased[0]))):
            try:
                result.append(real_part_rebased[0][-i - 1])
            except IndexError:
                result.append(0)

            try:
                result.append(imag_part_rebased[0][-i - 1])
            except IndexError:
                result.append(0)

        result.reverse()
        return remove_leading_zeros(result), [abs(self.base) // 2] if z.imag % 2 else []

    def to_complex(self, representation: Tuple[List[int], List[int]]) -> complex:
        """Convert from this system's representation to Python complex"""
        integer_digits, fraction_digits = representation
        real = 0
        imag = 0

        for i, digit in enumerate(reversed(integer_digits)):
            if i % 2 == 0:
                real += digit * (self.base ** (i // 2))
            else:
                imag += digit * (self.base ** (i // 2)) * 2

        for i, digit in enumerate(reversed(fraction_digits)):
            if i % 2 == 0:
                imag += digit * (1 / self.base) ** ((i + 2) // 2) * 2
            else:
                real += digit * (1 / self.base) ** ((i + 2) // 2)

        return complex(real, imag)

    def add(self, a: Tuple[List[int], List[int]], b: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
        """Add two numbers in radix system"""
        a_int, a_frac = a
        b_int, b_frac = b

        # Pad integer parts to same length, add 4 to allow for extra carry
        max_int_len = max(len(a_int), len(b_int)) + 4
        a_int_padded = [0] * (max_int_len - len(a_int)) + a_int
        b_int_padded = [0] * (max_int_len - len(b_int)) + b_int

        # Pad fractional parts to same length
        max_frac_len = max(len(a_frac), len(b_frac))
        a_frac_padded = a_frac + [0] * (max_frac_len - len(a_frac))
        b_frac_padded = b_frac + [0] * (max_frac_len - len(b_frac))

        # Combine integer and fractional parts for processing
        combined_a = a_int_padded + a_frac_padded
        combined_b = b_int_padded + b_frac_padded

        result = [0] * len(combined_a)
        carry = [0] * (len(combined_a) + 2)

        # Process from right to left
        for i in range(len(combined_a) - 1, -1, -1):
            total = combined_a[i] + combined_b[i] + carry[i + 2]

            # Apply quarter-imaginary rules
            while total > abs(self.base) - 1:
                total -= abs(self.base)
                carry[i] -= 1

            while total < 0:
                total += abs(self.base)
                carry[i] += 1

            result[i] = total

        # Handle remaining carries
        extra_digits = []
        for i in range(1, -1, -1):
            if carry[i] != 0:
                digit = carry[i]
                while digit > abs(self.base) - 1:
                    digit -= abs(self.base)
                    if i > 0:
                        carry[i - 1] -= 1
                    else:
                        extra_digits.insert(0, -1)
                while digit < 0:
                    digit += abs(self.base)
                    if i > 0:
                        carry[i - 1] += 1
                    else:
                        extra_digits.insert(0, 1)
                if digit != 0:
                    extra_digits.append(digit)

        result_int = extra_digits + result[:max_int_len]
        result_frac = result[max_int_len:]

        result_int = remove_leading_zeros(result_int)
        result_frac = remove_trailing_zeros(result_frac)

        return result_int, result_frac

    def multiply(self, a: Tuple[List[int], List[int]], b: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
        """Multiply two numbers in this system"""
        a_int, a_frac = a
        b_int, b_frac = b

        # Handle special cases
        if (not a_int or all(d == 0 for d in a_int)) and (not a_frac or all(d == 0 for d in a_frac)):
            return [0], []
        if (not b_int or all(d == 0 for d in b_int)) and (not b_frac or all(d == 0 for d in b_frac)):
            return [0], []

        # Combine digits for processing
        a_combined = a_int + a_frac
        b_combined = b_int + b_frac
        result_frac_len = len(a_frac) + len(b_frac)

        partial_products = []

        for i, b_digit in enumerate(reversed(b_combined)):
            if b_digit == 0:
                continue

            partial = []
            carry = 0

            for a_digit in reversed(a_combined):
                product = a_digit * b_digit + carry
                partial.append(product)
                carry = 0

            partial.reverse()
            shifted_partial = partial + [0] * i

            if result_frac_len >= len(shifted_partial):
                p_int = [0]
                p_frac = [0] * (result_frac_len - len(shifted_partial)) + shifted_partial
            else:
                split_pos = len(shifted_partial) - result_frac_len
                p_int = shifted_partial[:split_pos] if split_pos > 0 else [0]
                p_frac = shifted_partial[split_pos:] if split_pos < len(shifted_partial) else []

            partial_products.append((p_int, p_frac))

        if not partial_products:
            return [0], []

        result = partial_products[0]
        for i in range(1, len(partial_products)):
            result = self.add(result, partial_products[i])

        return result


# ======================== Radix-j√2 System ========================
class RadixJSqrt2System(GenericRadixComplexSystem):
    """Radix-j√2 complex number system"""

    def __init__(self):
        super().__init__(-2)

    @property
    def name(self) -> str:
        return "Radix-j√2"


# ======================== Radix-j−1 System ========================
class RadixJMinus1System(ComplexNumberSystem):
    """Radix-j−1 complex number system"""

    def __init__(self, bits: int = 8):
        self.bits = bits

    def from_complex(self, z: complex) -> List[int]:
        """Convert complex number to radix-j−1 representation"""
        a = int(z.real)
        b = int(z.imag)
        digits = []

        for _ in range(self.bits):
            r = (a + b) % 2
            digits.append(r)
            y = (r - a - b) // 2
            x = b + y
            a, b = x, y

        return digits[::-1]

    def to_complex(self, representation: List[int]) -> complex:
        """Convert radix-j−1 representation to complex number"""
        digits = representation
        z = 0
        base = -1 + 1j

        for digit in reversed(digits):
            z = z * base + digit

        return z

    def add(self, a: List[int], b: List[int]) -> List[int]:
        """Add two numbers in radix-j−1 system"""
        max_len = max(len(a), len(b))
        a_padded = a + [0] * (max_len - len(a))
        b_padded = b + [0] * (max_len - len(b))

        result = []
        carry = 0

        for i in range(max_len):
            total = a_padded[i] + b_padded[i] + carry
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

    def multiply(self, a: List[int], b: List[int]) -> List[int]:
        """Multiply two numbers in radix-j−1 system"""
        partials = []

        for i, digit in enumerate(b):
            partial = [0] * i + [d * digit for d in a]
            partials.append(partial)

        result = []
        for p in partials:
            result = self.add(result, p)

        return result

    @property
    def name(self) -> str:
        return "Radix-j−1"


# ======================== Radix-2j System ========================
class Radix2jSystem(GenericRadixComplexSystem):
    """Radix-2j (quarter-imaginary) complex number system"""

    def __init__(self):
        super().__init__(-4)

    @property
    def name(self) -> str:
        return "Radix-2j"


# ======================== Benchmark Setup ========================
def run_tests():
    # Configuration constants
    OPERATION = "multiply"  # Choose "add" or "multiply"
    MIN_RANDOM = -10  # Minimum value for random numbers
    MAX_RANDOM = 10   # Maximum value for random numbers
    TEST_COUNT = 10000  # Number of random tests to run
    VERBOSE = False   # Set to True for detailed output of each test

    current_system = RadixJMinus1System()

    # Example of testing specific values (uncomment and modify as needed)
    """
    # Custom test cases
    test_cases = [
        (complex(3, 4), complex(2, -1)),  # (3+4i) and (2-i)
        (complex(0, -4), complex(0, -3)),  # (0-4i) and (0-3i)
        (complex(3, -4), complex(1, -8)),  # (3-4i) and (1-8i)
    ]
    
    for z1, z2 in test_cases:
        r2j_z1 = current_system.from_complex(z1)
        r2j_z2 = current_system.from_complex(z2)
        
        if OPERATION == "add":
            expected = z1 + z2
            r2j_result = current_system.add(r2j_z1, r2j_z2)
        else:  # multiply
            expected = z1 * z2
            r2j_result = current_system.multiply(r2j_z1, r2j_z2)
            
        result_complex = current_system.to_complex(r2j_result)
        expected_r2j = current_system.from_complex(expected)
        
        print(f"Test: {z1} {'+' if OPERATION == 'add' else '*'} {z2} = {expected}")
        print(f"Radix-2j: {r2j_z1} {'+' if OPERATION == 'add' else '*'} {r2j_z2} = {r2j_result}")
        print(f"Expected radix-2j result: {expected_r2j}")
        print(f"Back to complex: {result_complex}")
        print(f"Correct: {abs(result_complex - expected) < 1e-10}")
        print()
    """

    # Fixed test case
    z1 = complex(7, -5)
    z2 = complex(0, -1)
    if OPERATION == "add":
        expected = z1 + z2
        operation_symbol = "+"
    else:  # multiply
        expected = z1 * z2
        operation_symbol = "*"

    r2j_z1 = current_system.from_complex(z1)
    r2j_z2 = current_system.from_complex(z2)

    if OPERATION == "add":
        r2j_result = current_system.add(r2j_z1, r2j_z2)
    else:  # multiply
        r2j_result = current_system.multiply(r2j_z1, r2j_z2)

    result_complex = current_system.to_complex(r2j_result)

    print(f"Fixed test: {z1} {operation_symbol} {z2} = {expected}")
    print(f"Radix-2j: {r2j_z1} {operation_symbol} {r2j_z2} = {r2j_result}")
    print(f"Expected radix-2j result: {current_system.from_complex(expected)}")
    print(f"Back to complex: {result_complex}")
    print(f"Correct: {abs(result_complex - expected) < 1e-10}")
    print()

    # Random tests
    if TEST_COUNT > 0:
        print(f"Random tests ({OPERATION}):")
    incorrect_count = 0

    for i in range(TEST_COUNT):
        z_a = complex(random.randint(MIN_RANDOM, MAX_RANDOM),
                     random.randint(MIN_RANDOM, MAX_RANDOM))
        z_b = complex(random.randint(MIN_RANDOM, MAX_RANDOM),
                     random.randint(MIN_RANDOM, MAX_RANDOM))

        try:
            r2j_a = current_system.from_complex(z_a)
            r2j_b = current_system.from_complex(z_b)

            if OPERATION == "add":
                expected_result = z_a + z_b
                r2j_result = current_system.add(r2j_a, r2j_b)
            else:  # multiply
                expected_result = z_a * z_b
                r2j_result = current_system.multiply(r2j_a, r2j_b)

            result = current_system.to_complex(r2j_result)
            is_correct = abs(result - expected_result) < 1e-10

            if VERBOSE:
                print(f"{z_a} {operation_symbol} {z_b} = {expected_result}")
                print(f"Radix-2j result: {result}")
                print(f"Correct: {is_correct}")
                print()

            if not is_correct:
                incorrect_count += 1
                if not VERBOSE:
                    print(f"Error case: {z_a} {operation_symbol} {z_b}")
                    print(f"Expected: {expected_result}, Got: {result}")
                    print()

        except Exception as e:
            print(f"Error with {z_a} {operation_symbol} {z_b}: {e}")
            incorrect_count += 1

    if TEST_COUNT > 0:
        print(f"Total incorrect results: {incorrect_count} out of {TEST_COUNT} random tests.")
        print(f"Success rate: {(TEST_COUNT - incorrect_count) / TEST_COUNT * 100:.2f}%")

if __name__ == "__main__":
    run_tests()
