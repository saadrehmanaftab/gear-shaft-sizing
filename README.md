# Gear shaft sizing and material comparison

Self initiated study, written and run in Python (NumPy, SciPy, Matplotlib). It is not coursework and not an employer project.
It is modelled on the kind of drivetrain shaft work done in a mechanical engineering role, but every input below is assumed.

## What it does
1. Computes tooth loads of a helical gear (tangential, radial, axial) from an assumed power and speed.
2. Builds a Euler Bernoulli finite element beam model of a stepped shaft on two bearings (static deflection, natural frequencies).
3. Validates that model against closed form results for a uniform simply supported shaft (center load deflection, first three natural frequencies).
4. Checks static strength (von Mises), fatigue (Goodman line, constant stress concentration factor) and the first natural frequency against the shaft speed.
5. Finds, per material, the smallest uniform diameter scale that meets all assumed limits, and reports mass and an assumed price index.

Run `python src/shaft.py` (writes `results/*.csv`, `results/summary.json`, `results/study.png`) and `python -m pytest`.

## Assumed data (placeholders, not measured, not from a datasheet)
- Power 12 kW, speed 6000 rpm, gear pitch diameter 60 mm, normal pressure angle 20 deg, helix angle 20 deg.
- Bearing span 160 mm, gear at 60 mm, stepped shaft 20 / 25 / 20 mm, gear mass 0.8 kg and inertia 4.0e-4 kg m^2.
- Fatigue stress concentration factor 1.5 everywhere, acceptance limits: static safety factor 2.0, fatigue safety factor 1.5, first natural frequency at least 1.5 times shaft speed.
- Material values (E, density, yield or strength limit, tensile strength, endurance ratio, surface factor, relative price per kg) in `MATS`, typical order of magnitude only.

## Limits
- One shaft, planar bending in two planes, no keyways, no bearing stiffness, no gear mesh dynamics, no temperature or humidity effects, no creep.
- Polymer strength is taken as a short term limit. Real polymer shafts need creep and moisture data this study does not have.
- The FE beam is exact for point loads, so agreement with closed form shows a correct implementation, not model accuracy.
- Results depend entirely on the assumed data. Do not read them as design values.
