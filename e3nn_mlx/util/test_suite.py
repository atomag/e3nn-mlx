"""
Comprehensive equivariance test suite for e3nn-mlx.

This module provides a complete test suite for validating the equivariance properties
of all major operations in the e3nn-mlx library.
"""

import mlx.core as mx
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from .test_equivariance import EquivarianceTester
from ..o3 import Irreps, spherical_harmonics, TensorProduct, Linear
from ..nn import Activation, Gate
from ..o3._rotation import rand_matrix


class EquivarianceTestSuite:
    """
    Comprehensive test suite for e3nn-mlx equivariance validation.
    
    This class provides methods to run all equivariance tests and generate
    detailed reports on the mathematical correctness of the implementation.
    """
    
    def __init__(self, tolerance: float = 1e-5, num_samples: int = 10):
        """
        Initialize the test suite.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed error for equivariance to hold
        num_samples : int
            Number of random test samples to generate
        """
        self.tolerance = tolerance
        self.num_samples = num_samples
        self.tester = EquivarianceTester(tolerance=tolerance, num_samples=num_samples)
        self.results = {}
        
    def run_all_tests(self) -> Dict:
        """
        Run all equivariance tests and return comprehensive results.
        
        Returns
        -------
        Dict
            Complete test results with detailed metrics
        """
        print("🧪 Running comprehensive equivariance test suite...")
        print(f"   Tolerance: {self.tolerance:.1e}")
        print(f"   Samples per test: {self.num_samples}")
        print("=" * 60)
        
        results = {
            'summary': {'passed': 0, 'failed': 0, 'total': 0},
            'spherical_harmonics': {},
            'tensor_products': {},
            'linear_layers': {},
            'activations': {},
            'gates': {},
            'performance_metrics': {}
        }
        
        # Test spherical harmonics
        print("\n📐 Testing Spherical Harmonics...")
        sh_results = self._test_spherical_harmonics()
        results['spherical_harmonics'] = sh_results
        self._update_summary(results, sh_results)
        
        # Test tensor products
        print("\n🔗 Testing Tensor Products...")
        tp_results = self._test_tensor_products()
        results['tensor_products'] = tp_results
        self._update_summary(results, tp_results)
        
        # Test linear layers
        print("\n📊 Testing Linear Layers...")
        linear_results = self._test_linear_layers()
        results['linear_layers'] = linear_results
        self._update_summary(results, linear_results)
        
        # Test activations
        print("\n⚡ Testing Activation Functions...")
        act_results = self._test_activations()
        results['activations'] = act_results
        self._update_summary(results, act_results)
        
        # Test gates
        print("\n🚪 Testing Gate Activations...")
        gate_results = self._test_gates()
        results['gates'] = gate_results
        self._update_summary(results, gate_results)
        
        # Performance metrics
        print("\n⏱️  Measuring Performance...")
        perf_results = self._measure_performance()
        results['performance_metrics'] = perf_results
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        self._print_summary(results['summary'])
        
        return results
        
    def _test_spherical_harmonics(self) -> Dict:
        """Test spherical harmonics equivariance."""
        results = {}
        
        # Test rotational equivariance for different l values
        for l in range(1, 6):
            test_name = f"l={l}_rotational"
            print(f"   Testing {test_name}...")
            
            def sh_operation(positions):
                return spherical_harmonics(l, positions)
            
            irreps_out = Irreps(f"1x{l}{'e' if l % 2 == 0 else 'o'}")
            
            is_equivariant, test_results = self.tester.assert_equivariant(
                operation=sh_operation,
                irreps_in="1x1o",  # positions are vectors
                irreps_out=irreps_out,
                input_generator=lambda ir: mx.random.normal((10, 3)),
                test_rotations=True,
                test_inversions=True,
                test_translations=False
            )
            
            results[test_name] = {
                'passed': is_equivariant,
                'max_error': max([r['max_error'] for r in test_results.values()]),
                'details': test_results
            }
            
            status = "✅ PASS" if is_equivariant else "❌ FAIL"
            print(f"     {status} (max_error: {results[test_name]['max_error']:.2e})")
        
        return results
        
    def _test_tensor_products(self) -> Dict:
        """Test tensor product equivariance."""
        results = {}
        
        test_cases = [
            ("1x0e", "1x0e", "1x0e", "scalar×scalar→scalar"),
            ("1x1o", "1x1o", "1x0e+1x1o+1x2e", "vector×vector→mixed"),
            ("1x0e", "1x1o", "1x1o", "scalar×vector→vector"),
            ("2x0e+1x1o", "1x0e+1x1o", "2x0e+1x1o+1x1o+1x2e", "mixed×mixed→mixed"),
        ]
        
        for irreps1, irreps2, irreps_out, description in test_cases:
            test_name = f"tp_{description.replace('×', '_x_').replace('→', '_to_').replace(' ', '_')}"
            print(f"   Testing {description}...")
            
            tp = TensorProduct(
                irreps_in1=Irreps(irreps1),
                irreps_in2=Irreps(irreps2),
                irreps_out=Irreps(irreps_out)
            )
            
            def tp_operation(combined_input):
                dim1 = Irreps(irreps1).dim
                input1 = combined_input[:dim1].reshape(1, -1)
                input2 = combined_input[dim1:].reshape(1, -1)
                return tp(input1, input2).reshape(-1)
            
            is_equivariant, test_results = self.tester.assert_equivariant(
                operation=tp_operation,
                irreps_in=Irreps(irreps1) + Irreps(irreps2),
                irreps_out=Irreps(irreps_out),
                test_rotations=True,
                test_inversions=True,
                test_translations=False
            )
            
            results[test_name] = {
                'passed': is_equivariant,
                'max_error': max([r['max_error'] for r in test_results.values()]),
                'details': test_results
            }
            
            status = "✅ PASS" if is_equivariant else "❌ FAIL"
            print(f"     {status} (max_error: {results[test_name]['max_error']:.2e})")
        
        return results
        
    def _test_linear_layers(self) -> Dict:
        """Test linear layer equivariance."""
        results = {}
        
        test_cases = [
            ("1x0e", "1x0e", "scalar→scalar"),
            ("1x1o", "1x1o", "vector→vector"),
            ("1x2e", "1x2e", "tensor→tensor"),
            ("1x0e+1x1o", "1x0e+1x1o", "mixed→mixed"),
            ("2x0e+1x1o", "3x0e+2x1o", "different_multiplicities"),
        ]
        
        for irreps_in, irreps_out, description in test_cases:
            test_name = f"linear_{description.replace('→', '_to_').replace(' ', '_')}"
            print(f"   Testing {description}...")
            
            linear = Linear(Irreps(irreps_in), Irreps(irreps_out))
            
            def linear_operation(input_features):
                return linear(input_features.reshape(1, -1)).reshape(-1)
            
            is_equivariant, test_results = self.tester.assert_equivariant(
                operation=linear_operation,
                irreps_in=Irreps(irreps_in),
                irreps_out=Irreps(irreps_out),
                test_rotations=True,
                test_inversions=True,
                test_translations=False
            )
            
            results[test_name] = {
                'passed': is_equivariant,
                'max_error': max([r['max_error'] for r in test_results.values()]),
                'details': test_results
            }
            
            status = "✅ PASS" if is_equivariant else "❌ FAIL"
            print(f"     {status} (max_error: {results[test_name]['max_error']:.2e})")
        
        return results
        
    def _test_activations(self) -> Dict:
        """Test activation function equivariance."""
        results = {}
        
        test_cases = [
            ("1x0e", [mx.abs], "scalar_abs"),
            ("1x0e", [mx.tanh], "scalar_tanh"),
            ("1x0e+1x1o", [mx.abs, None], "mixed_activation"),
            ("2x0e+1x1o", [mx.tanh, None], "multiple_scalars"),
        ]
        
        for irreps_in, acts, description in test_cases:
            test_name = f"act_{description}"
            print(f"   Testing {description}...")
            
            activation = Activation(Irreps(irreps_in), acts)
            
            def act_operation(input_features):
                return activation(input_features)
            
            is_equivariant, test_results = self.tester.assert_equivariant(
                operation=act_operation,
                irreps_in=Irreps(irreps_in),
                irreps_out=activation.irreps_out,
                test_rotations=True,
                test_inversions=True,
                test_translations=False
            )
            
            results[test_name] = {
                'passed': is_equivariant,
                'max_error': max([r['max_error'] for r in test_results.values()]),
                'details': test_results
            }
            
            status = "✅ PASS" if is_equivariant else "❌ FAIL"
            print(f"     {status} (max_error: {results[test_name]['max_error']:.2e})")
        
        return results
        
    def _test_gates(self) -> Dict:
        """Test gate activation equivariance."""
        results = {}
        
        test_cases = [
            ("1x0e", [mx.sigmoid], "1x0e", [mx.tanh], "1x1o", "simple_gate"),
            ("2x0e", [mx.sigmoid, mx.tanh], "2x0e", [mx.tanh, mx.sigmoid], "2x1o", "multiple_gates"),
        ]
        
        for scalars, act_scalars, gates, act_gates, gated, description in test_cases:
            test_name = f"gate_{description}"
            print(f"   Testing {description}...")
            
            gate = Gate(
                Irreps(scalars), act_scalars,
                Irreps(gates), act_gates,
                Irreps(gated)
            )
            
            def gate_operation(input_features):
                return gate(input_features)
            
            is_equivariant, test_results = self.tester.assert_equivariant(
                operation=gate_operation,
                irreps_in=gate.irreps_in,
                irreps_out=gate.irreps_out,
                test_rotations=True,
                test_inversions=True,
                test_translations=False
            )
            
            results[test_name] = {
                'passed': is_equivariant,
                'max_error': max([r['max_error'] for r in test_results.values()]),
                'details': test_results
            }
            
            status = "✅ PASS" if is_equivariant else "❌ FAIL"
            print(f"     {status} (max_error: {results[test_name]['max_error']:.2e})")
        
        return results
        
    def _measure_performance(self) -> Dict:
        """Measure performance metrics."""
        results = {}
        
        # Measure spherical harmonics performance
        start_time = time.time()
        for _ in range(100):
            positions = mx.random.normal((100, 3))
            for l in range(1, 4):
                _ = spherical_harmonics(l, positions)
        sh_time = time.time() - start_time
        results['spherical_harmonics_time'] = sh_time
        
        # Measure tensor product performance
        tp = TensorProduct(
            Irreps("1x1o"), Irreps("1x1o"), 
            Irreps("1x0e+1x1o+1x2e")
        )
        start_time = time.time()
        for _ in range(100):
            input1 = mx.random.normal((10, 3))
            input2 = mx.random.normal((10, 3))
            _ = tp(input1, input2)
        tp_time = time.time() - start_time
        results['tensor_product_time'] = tp_time
        
        print(f"   Spherical harmonics (100x3x100): {sh_time:.3f}s")
        print(f"   Tensor product (100x10x3): {tp_time:.3f}s")
        
        return results
        
    def _update_summary(self, results: Dict, category_results: Dict):
        """Update the summary statistics."""
        for test_name, test_result in category_results.items():
            results['summary']['total'] += 1
            if test_result['passed']:
                results['summary']['passed'] += 1
            else:
                results['summary']['failed'] += 1
                
    def _print_summary(self, summary: Dict):
        """Print the test summary."""
        total = summary['total']
        passed = summary['passed']
        failed = summary['failed']
        pass_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ({pass_rate:.1f}%)")
        print(f"Failed: {failed} ({100-pass_rate:.1f}%)")
        
        if failed == 0:
            print("\n🎉 All equivariance tests passed!")
        else:
            print(f"\n⚠️  {failed} test(s) failed - review detailed results")
            
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed report of the test results."""
        report = []
        report.append("# e3nn-mlx Equivariance Validation Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        summary = results['summary']
        total = summary['total']
        passed = summary['passed']
        failed = summary['failed']
        pass_rate = (passed / total) * 100 if total > 0 else 0
        
        report.append("## Summary")
        report.append(f"- Total tests: {total}")
        report.append(f"- Passed: {passed} ({pass_rate:.1f}%)")
        report.append(f"- Failed: {failed} ({100-pass_rate:.1f}%)")
        report.append(f"- Tolerance: {self.tolerance:.1e}")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        # Spherical harmonics
        report.append("### Spherical Harmonics")
        for test_name, result in results['spherical_harmonics'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"- {test_name}: {status} (max_error: {result['max_error']:.2e})")
        report.append("")
        
        # Tensor products
        report.append("### Tensor Products")
        for test_name, result in results['tensor_products'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"- {test_name}: {status} (max_error: {result['max_error']:.2e})")
        report.append("")
        
        # Linear layers
        report.append("### Linear Layers")
        for test_name, result in results['linear_layers'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"- {test_name}: {status} (max_error: {result['max_error']:.2e})")
        report.append("")
        
        # Activations
        report.append("### Activation Functions")
        for test_name, result in results['activations'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"- {test_name}: {status} (max_error: {result['max_error']:.2e})")
        report.append("")
        
        # Gates
        report.append("### Gate Activations")
        for test_name, result in results['gates'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"- {test_name}: {status} (max_error: {result['max_error']:.2e})")
        report.append("")
        
        # Performance metrics
        report.append("### Performance Metrics")
        perf = results['performance_metrics']
        report.append(f"- Spherical harmonics time: {perf['spherical_harmonics_time']:.3f}s")
        report.append(f"- Tensor product time: {perf['tensor_product_time']:.3f}s")
        report.append("")
        
        # Conclusion
        report.append("## Conclusion")
        if failed == 0:
            report.append("🎉 All equivariance tests passed! The implementation demonstrates")
            report.append("correct mathematical behavior under rotations and inversions.")
        else:
            report.append("⚠️  Some equivariance tests failed. The implementation may have")
            report.append("mathematical issues that need to be addressed.")
        
        return "\n".join(report)


def run_equivariance_tests(tolerance: float = 1e-5, num_samples: int = 10) -> Dict:
    """
    Convenience function to run the full equivariance test suite.
    
    Parameters
    ----------
    tolerance : float
        Maximum allowed error for equivariance to hold
    num_samples : int
        Number of random test samples to generate
        
    Returns
    -------
    Dict
        Complete test results
    """
    suite = EquivarianceTestSuite(tolerance=tolerance, num_samples=num_samples)
    return suite.run_all_tests()


if __name__ == "__main__":
    # Run the full test suite
    results = run_equivariance_tests()
    
    # Generate and print report
    suite = EquivarianceTestSuite()
    report = suite.generate_report(results)
    print("\n" + report)
    
    # Save report to file
    with open("equivariance_validation_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: equivariance_validation_report.md")