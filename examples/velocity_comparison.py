#!/usr/bin/env python
"""
Beispielskript: Vergleich der Velocity-Berechnungsmethoden

Demonstriert den Unterschied zwischen Olsen 2D und Ray 3D Methoden
mit verschiedenen Koordinaten-Rounding-Strategien.
"""
from pathlib import Path
import pandas as pd
import sys

# Füge Parent-Directory zum Path hinzu für lokale Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.io import read_tsv


def compare_velocity_methods(input_file: str):
    """
    Vergleicht Olsen 2D vs Ray 3D Velocity-Berechnung.
    
    Args:
        input_file: Pfad zur preprocessed TSV-Datei
    """
    print("=" * 80)
    print("Velocity Method Comparison")
    print("=" * 80)
    print()
    
    # Daten laden
    print(f"Loading data from: {input_file}")
    df = read_tsv(input_file)
    print(f"Loaded {len(df)} samples")
    print()
    
    # Konfigurationen erstellen
    configs = [
        ("Olsen 2D (no rounding)", OlsenVelocityConfig(
            velocity_method="olsen2d",
            coordinate_rounding="none"
        )),
        ("Olsen 2D (halfup rounding)", OlsenVelocityConfig(
            velocity_method="olsen2d",
            coordinate_rounding="halfup"
        )),
        ("Ray 3D (no rounding)", OlsenVelocityConfig(
            velocity_method="ray3d",
            coordinate_rounding="none"
        )),
        ("Ray 3D (halfup rounding)", OlsenVelocityConfig(
            velocity_method="ray3d",
            coordinate_rounding="halfup"
        )),
    ]
    
    results = {}
    
    # Berechne Velocities für jede Konfiguration
    for name, config in configs:
        print(f"Computing velocities: {name}")
        df_result = compute_olsen_velocity(df.copy(), config)
        velocities = df_result["velocity_deg_per_sec"].dropna()
        
        results[name] = {
            "min": velocities.min(),
            "max": velocities.max(),
            "mean": velocities.mean(),
            "median": velocities.median(),
            "std": velocities.std(),
            "count": len(velocities),
        }
        print(f"  Computed {len(velocities)} velocity values")
        print()
    
    # Statistiken ausgeben
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    
    for name, stats in results.items():
        print(f"{name}:")
        print(f"  Count:  {stats['count']}")
        print(f"  Min:    {stats['min']:.2f} deg/s")
        print(f"  Max:    {stats['max']:.2f} deg/s")
        print(f"  Mean:   {stats['mean']:.2f} deg/s")
        print(f"  Median: {stats['median']:.2f} deg/s")
        print(f"  Std:    {stats['std']:.2f} deg/s")
        print()
    
    # Vergleiche
    print("=" * 80)
    print("Method Comparisons")
    print("=" * 80)
    print()
    
    olsen_no_round = results["Olsen 2D (no rounding)"]["mean"]
    ray_no_round = results["Ray 3D (no rounding)"]["mean"]
    diff_methods = olsen_no_round - ray_no_round
    diff_pct = (diff_methods / ray_no_round) * 100
    
    print(f"Olsen 2D vs Ray 3D (no rounding):")
    print(f"  Difference: {diff_methods:.2f} deg/s ({diff_pct:.1f}%)")
    print(f"  Ray 3D is typically {diff_pct:.1f}% lower")
    print()
    
    olsen_no = results["Olsen 2D (no rounding)"]["mean"]
    olsen_round = results["Olsen 2D (halfup rounding)"]["mean"]
    diff_round = olsen_no - olsen_round
    diff_round_pct = (diff_round / olsen_no) * 100
    
    print(f"Rounding Effect (Olsen 2D):")
    print(f"  No rounding:     {olsen_no:.2f} deg/s")
    print(f"  Halfup rounding: {olsen_round:.2f} deg/s")
    print(f"  Difference:      {diff_round:.2f} deg/s ({diff_round_pct:.1f}%)")
    print()
    
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    print()
    print("1. Olsen 2D (no rounding):")
    print("   - Standard method, fast, backward compatible")
    print("   - Use for: Tobii-like filtering, standard I-VT classification")
    print()
    print("2. Ray 3D (no rounding):")
    print("   - Physically correct, more accurate")
    print("   - Use for: Research, accurate velocity measurements")
    print("   - ~1-5% lower velocities than Olsen 2D")
    print()
    print("3. Coordinate Rounding:")
    print("   - Use 'halfup' or 'nearest' to match Tobii's integer rounding")
    print("   - Use 'none' for maximum precision")
    print("   - Effect: typically 1-3% velocity reduction")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python velocity_comparison.py <input_file.tsv>")
        print()
        print("Example:")
        print("  python velocity_comparison.py ../data/processed/ivt_input.tsv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    compare_velocity_methods(input_file)
