"""Nozzle clogging model.

This module implements the Monte Carlo-Enhanced Physics-Based Modeling of
Sediment-Induced Clogging Risk in Semi-Solid Set Sprinkler Irrigation Systems.

The model provides:
1. Uncertainty quantification via Monte Carlo simulation
2. Physics-based sediment transport and clogging mechanisms
3. Risk assessment and classification for agricultural irrigation systems

Key features:
- Support for semi-solid set sprinkler systems (latersections moved periodically)
- Regime switching based on operational boundaries
- Comprehensive sensitivity and convergence analysis
- Document-grounded academic readiness for publication

Unit System:
- All calculations use SI units with pint Quantity support
- Temperature: °C, Pressure: kPa, Length: mm, Velocity: m/s
- Volume fraction and dimensionless indices for modeling convenience

Module Structure:
- config: Physical constants and simulation parameters
- physics: Core clogging physics calculations
- probability: Risk assessment and probability calculations
- generation: Latin Hypercube sampling and parameter generation
- orchestration: Pipeline coordination and model integration
- schemas: Data validation and unit-aware schemas
- simulation: Batched simulation execution and result aggregation

Workflow:
1. Generate input parameters (LHS sampling)
2. Compute physics parameters (Stokes number, shear factors, etc.)
3. Calculate clogging probability using physics-based dimensionless indices
4. Classify risk levels (Low/Moderate/High)
5. Perform comprehensive sensitivity and convergence analysis

This module provides a robust foundation for sediment-induced clogging risk
assessment in pressurized irrigation systems, with applications in sustainable
agricultural water management and crop yield optimization.
"""
