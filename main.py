import math
import random
import time
import statistics
import csv
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
        self.bits = bits  # Keep for compatibility, but make it adaptive

    def from_complex(self, z: complex) -> List[int]:
        """Convert complex number to radix-j−1 representation using Euclidean division"""
        # Validate input is a Gaussian integer
        if (abs(z.real - round(z.real)) > 1e-10 or 
            abs(z.imag - round(z.imag)) > 1e-10):
            raise ValueError("Input must be a Gaussian integer (integer real and imaginary parts)")
        
        if z == 0:
            return [0]
        
        # Base is j-1 = -1+i
        base = -1 + 1j
        
        # Start with rounded Gaussian integer
        current = complex(round(z.real), round(z.imag))
        digits = []
        
        # Calculate maximum possible bits needed based on input magnitude
        # Use the maximum of real and imaginary parts to estimate bit length
        max_component = max(abs(int(current.real)), abs(int(current.imag)))
        max_bits = max(self.bits, max_component.bit_length() * 2 + 10) if max_component > 0 else self.bits
        
        # Euclidean division algorithm
        while current != 0 and len(digits) < max_bits:
            # Divide current by base
            quotient_raw = current / base
            
            # Round to nearest Gaussian integer
            q_real = round(quotient_raw.real)
            q_imag = round(quotient_raw.imag)
            quotient = complex(q_real, q_imag)
            
            # Calculate remainder: current = quotient * base + remainder
            remainder = current - quotient * base
            
            # For base (-1+i), remainder should be 0 or 1
            # Round remainder to handle floating point precision
            r_real = round(remainder.real)
            r_imag = round(remainder.imag)
            
            # Determine digit based on remainder
            if abs(r_real) < 1e-10 and abs(r_imag) < 1e-10:
                digit = 0
            elif abs(r_real - 1) < 1e-10 and abs(r_imag) < 1e-10:
                digit = 1
            else:
                # Handle edge cases by trying both possible digits
                best_digit = 0
                best_error = float('inf')
                
                for test_digit in [0, 1]:
                    test_quotient = (current - test_digit) / base
                    test_q = complex(round(test_quotient.real), round(test_quotient.imag))
                    test_remainder = current - test_q * base - test_digit
                    error = abs(test_remainder)
                    
                    if error < best_error:
                        best_error = error
                        best_digit = test_digit
                        quotient = test_q
                
                digit = best_digit
            
            digits.append(digit)
            current = quotient
        
        # Safety check
        if len(digits) >= max_bits and current != 0:
            raise RuntimeError("Conversion algorithm failed to converge within reasonable bounds")
        
        # Digits are generated in reverse order, so reverse them
        digits.reverse()
        
        # Remove leading zeros but keep at least one digit
        while len(digits) > 1 and digits[0] == 0:
            digits.pop(0)
            
        return digits

    def to_complex(self, representation: List[int]) -> complex:
        """Convert radix-j−1 representation to complex number"""
        if not representation or all(d == 0 for d in representation):
            return 0 + 0j
        
        base = -1 + 1j
        result = 0 + 0j
        
        # Evaluate polynomial in base (-1+i)
        # Most significant digit is first in our representation
        for position, digit in enumerate(reversed(representation)):
            if digit == 1:
                result += base ** position
        
        return result

    def add(self, a: List[int], b: List[int]) -> List[int]:
        """Add two numbers in radix-j−1 system using native arithmetic"""
        # Convert to complex, add, then convert back
        complex_a = self.to_complex(a)
        complex_b = self.to_complex(b)
        result_complex = complex_a + complex_b
        return self.from_complex(result_complex)

    def multiply(self, a: List[int], b: List[int]) -> List[int]:
        """Multiply two numbers in radix-j−1 system using native arithmetic"""
        # Convert to complex, multiply, then convert back
        complex_a = self.to_complex(a)
        complex_b = self.to_complex(b)
        result_complex = complex_a * complex_b
        return self.from_complex(result_complex)

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


# ======================== Binary Complex System ========================
class BinaryComplexSystem(ComplexNumberSystem):
    """Binary representation of complex numbers using two's complement"""

    def __init__(self, bits: int = 16):
        """Initialize with specified bit width (default 16 bits per component)"""
        self.bits = bits
        self.max_val = (1 << (bits - 1)) - 1  # Maximum positive value
        self.min_val = -(1 << (bits - 1))     # Minimum negative value

    def _int_to_binary(self, num: int) -> List[int]:
        """Convert integer to binary list using two's complement"""
        # Clamp to valid range
        num = max(self.min_val, min(self.max_val, num))
        
        if num >= 0:
            # Positive number: standard binary
            binary = [(num >> i) & 1 for i in range(self.bits)]
        else:
            # Negative number: two's complement
            num = (1 << self.bits) + num  # Add 2^bits to get two's complement
            binary = [(num >> i) & 1 for i in range(self.bits)]
        
        return binary

    def _binary_to_int(self, binary: List[int]) -> int:
        """Convert binary list to integer using two's complement"""
        if len(binary) != self.bits:
            binary = binary[:self.bits] + [0] * max(0, self.bits - len(binary))
        
        # Calculate value
        value = sum(bit * (1 << i) for i, bit in enumerate(binary))
        
        # Handle two's complement for negative numbers
        if binary[self.bits - 1] == 1:  # Sign bit is set
            value -= 1 << self.bits
            
        return value

    def from_complex(self, z: complex) -> Tuple[List[int], List[int]]:
        """Convert complex number to binary representation (real_bits, imag_bits)"""
        real_part = int(round(z.real))
        imag_part = int(round(z.imag))
        
        real_binary = self._int_to_binary(real_part)
        imag_binary = self._int_to_binary(imag_part)
        
        return real_binary, imag_binary

    def to_complex(self, representation: Tuple[List[int], List[int]]) -> complex:
        """Convert binary representation to complex number"""
        real_binary, imag_binary = representation
        
        real_part = self._binary_to_int(real_binary)
        imag_part = self._binary_to_int(imag_binary)
        
        return complex(real_part, imag_part)

    def _binary_add(self, a: List[int], b: List[int]) -> List[int]:
        """Add two binary numbers with carry propagation"""
        result = [0] * self.bits
        carry = 0
        
        for i in range(self.bits):
            total = a[i] + b[i] + carry
            result[i] = total & 1
            carry = total >> 1
            
        return result

    def _binary_multiply(self, a: List[int], b: List[int]) -> List[int]:
        """Multiply two binary numbers (simplified - may overflow)"""
        # Convert to integers, multiply, then convert back
        # This is a simplified implementation
        int_a = self._binary_to_int(a)
        int_b = self._binary_to_int(b)
        product = int_a * int_b
        
        return self._int_to_binary(product)

    def add(self, a: Tuple[List[int], List[int]], b: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
        """Add two complex numbers in binary representation"""
        a_real, a_imag = a
        b_real, b_imag = b
        
        result_real = self._binary_add(a_real, b_real)
        result_imag = self._binary_add(a_imag, b_imag)
        
        return result_real, result_imag

    def multiply(self, a: Tuple[List[int], List[int]], b: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
        """Multiply two complex numbers in binary representation"""
        a_real, a_imag = a
        b_real, b_imag = b
        
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        ac = self._binary_multiply(a_real, b_real)
        bd = self._binary_multiply(a_imag, b_imag)
        ad = self._binary_multiply(a_real, b_imag)
        bc = self._binary_multiply(a_imag, b_real)
        
        # Convert bd to negative for subtraction (two's complement)
        bd_neg = self._int_to_binary(-self._binary_to_int(bd))
        
        result_real = self._binary_add(ac, bd_neg)  # ac - bd
        result_imag = self._binary_add(ad, bc)      # ad + bc
        
        return result_real, result_imag

    @property
    def name(self) -> str:
        return f"Binary Complex ({self.bits}-bit)"


# ======================== Benchmarking Functions ========================
def run_benchmark(iterations: int = 1000, num_runs: int = 10, 
                  min_val: int = -10, max_val: int = 10,
                  csv_filename: str = "benchmark_results.csv"):
    """
    Comprehensive benchmark of all complex number systems.
    
    Args:
        iterations: Number of operations per run
        num_runs: Number of benchmark runs for statistical analysis
        min_val: Minimum value for random complex numbers
        max_val: Maximum value for random complex numbers
        csv_filename: Name of CSV file to save detailed results
    """
    
    # Initialize all systems
    systems = [
        TraditionalComplexSystem(),
        BinaryComplexSystem(16),
        BinaryComplexSystem(32),
        Radix2jSystem(),
        RadixJSqrt2System(),
        RadixJMinus1System(8),
        RadixJMinus1System(16)
    ]
    
    # Operations to benchmark
    operations = ['from_complex', 'to_complex', 'add', 'multiply']
    
    # Store all results for CSV
    csv_data = []
    
    # Store simplified results for table display
    results_summary = {}
    
    print("Running comprehensive benchmark...")
    print(f"Iterations per run: {iterations}")
    print(f"Number of runs: {num_runs}")
    print(f"Value range: {min_val} to {max_val}")
    print("=" * 80)
    
    for system in systems:
        print(f"Benchmarking {system.name}...")
        results_summary[system.name] = {}
        
        for operation in operations:
            operation_times = []
            
            for run in range(num_runs):
                # Generate test data for this run
                test_data = []
                for _ in range(iterations):
                    z1 = complex(random.randint(min_val, max_val), 
                                random.randint(min_val, max_val))
                    z2 = complex(random.randint(min_val, max_val), 
                                random.randint(min_val, max_val))
                    test_data.append((z1, z2))
                
                # Time the operation
                start_time = time.perf_counter()
                
                try:
                    if operation == 'from_complex':
                        for z1, _ in test_data:
                            system.from_complex(z1)
                    
                    elif operation == 'to_complex':
                        # Pre-convert for to_complex timing
                        converted_data = []
                        for z1, _ in test_data:
                            try:
                                converted_data.append(system.from_complex(z1))
                            except:
                                continue
                        
                        start_time = time.perf_counter()  # Reset timer
                        for rep in converted_data:
                            system.to_complex(rep)
                    
                    elif operation == 'add':
                        for z1, z2 in test_data:
                            try:
                                rep1 = system.from_complex(z1)
                                rep2 = system.from_complex(z2)
                                system.add(rep1, rep2)
                            except:
                                continue
                    
                    elif operation == 'multiply':
                        for z1, z2 in test_data:
                            try:
                                rep1 = system.from_complex(z1)
                                rep2 = system.from_complex(z2)
                                system.multiply(rep1, rep2)
                            except:
                                continue
                
                except Exception as e:
                    print(f"  Error in {operation}: {e}")
                    continue
                
                end_time = time.perf_counter()
                operation_time = (end_time - start_time) * 1_000_000  # Convert to microseconds
                operation_times.append(operation_time)
            
            # Calculate statistics
            if operation_times:
                avg_time = statistics.mean(operation_times)
                min_time = min(operation_times)
                max_time = max(operation_times)
                total_time = sum(operation_times)
                std_dev = statistics.stdev(operation_times) if len(operation_times) > 1 else 0
                median_time = statistics.median(operation_times)
                
                # Store for summary table
                results_summary[system.name][operation] = avg_time
                
                # Store detailed data for CSV
                csv_data.append({
                    'system': system.name,
                    'operation': operation,
                    'iterations': iterations,
                    'num_runs': num_runs,
                    'avg_time_us': avg_time,
                    'min_time_us': min_time,
                    'max_time_us': max_time,
                    'total_time_us': total_time,
                    'std_dev_us': std_dev,
                    'median_time_us': median_time,
                    'time_per_operation_us': avg_time / iterations,
                    'operations_per_second': (iterations * 1_000_000) / avg_time if avg_time > 0 else 0
                })
            else:
                results_summary[system.name][operation] = float('inf')  # Indicate failure
    
    # Display simplified summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (Average time in microseconds)")
    print("=" * 80)
    
    # Header
    header = f"{'System':<25} {'from_complex':<12} {'to_complex':<12} {'add':<12} {'multiply':<12}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for system_name, operations_data in results_summary.items():
        from_complex_time = operations_data.get('from_complex', float('inf'))
        to_complex_time = operations_data.get('to_complex', float('inf'))
        add_time = operations_data.get('add', float('inf'))
        multiply_time = operations_data.get('multiply', float('inf'))
        
        # Format times
        def format_time(t):
            if t == float('inf'):
                return "FAILED"
            elif t < 1:
                return f"{t:.3f}"
            elif t < 10:
                return f"{t:.2f}"
            elif t < 100:
                return f"{t:.1f}"
            else:
                return f"{int(t)}"
        
        row = f"{system_name:<25} {format_time(from_complex_time):<12} {format_time(to_complex_time):<12} {format_time(add_time):<12} {format_time(multiply_time):<12}"
        print(row)
    
    # Save detailed results to CSV
    print(f"\nSaving detailed results to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        if csv_data:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"Detailed benchmark results saved to {csv_filename}")
    print("\nCSV contains: system, operation, avg/min/max/total/std_dev/median times,")
    print("             time per operation, and operations per second")
    

# ======================== Benchmark Setup ========================
def run_tests():
    # Configuration constants
    OPERATION = "multiply"  # Choose "add" or "multiply"
    MIN_RANDOM = -1000  # Minimum value for random numbers
    MAX_RANDOM = 1000   # Maximum value for random numbers
    TEST_COUNT = 10000  # Number of random tests to run
    VERBOSE = False   # Set to True for detailed output of each test

    current_system = RadixJSqrt2System()

    # # test conversion
    c = complex(83, 479)
    print(current_system.from_complex(c))
    print(current_system.to_complex(current_system.from_complex(c)))

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
    # Run the benchmark
    # print("Starting benchmark...")
    # run_benchmark(iterations=1000, num_runs=5, min_val=-10, max_val=10)
    
    # print("\n" + "=" * 80)
    # print("Running original tests...")
    # print("=" * 80)
    
    # Run the original tests
    run_tests()
