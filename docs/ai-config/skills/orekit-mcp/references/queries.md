# Orekit MCP: Query Cookbook

Use `orekit_search_symbols` with short, high-signal queries, then pull docs with
`orekit_get_class_doc` / `orekit_get_member_doc`.

## General templates

- Class discovery: `<ClassName>`
- Method discovery: `<ClassName> <methodName>`
- Domain discovery: `<DomainWord> <ClassName>`
- When unsure: run 2-3 separate searches rather than one long prompt.

## Time and dates

- `AbsoluteDate constructor`
- `AbsoluteDate shiftedBy`
- `TimeScalesFactory getUTC`
- `DateComponents`
- `TimeComponents`

## Frames and transforms

- `FramesFactory getGCRF`
- `FramesFactory getEME2000`
- `Frame getTransformTo`
- `Transform transformPVCoordinates`

## Orbits and state

- `KeplerianOrbit constructor`
- `CartesianOrbit constructor`
- `OrbitType`
- `PositionAngleType`
- `SpacecraftState`
- `PVCoordinates`

## Propagation (numerical)

- `NumericalPropagatorBuilder`
- `NumericalPropagatorBuilder buildPropagator`
- `NumericalPropagator addForceModel`
- `Propagator propagate`

## Integrators

- `DormandPrince853Integrator`
- `ClassicalRungeKuttaIntegrator`
- `AdaptiveStepsizeIntegrator`

## Forces

- `HolmesFeatherstoneAttractionModel`
- `NewtonianAttraction`
- `ThirdBodyAttraction`
- `Relativity`
- `SolarRadiationPressure`
- `DragForce`
- `IsotropicDrag`
- `HarrisPriester`
- `NRLMSISE00`

## DSST (semi-analytical)

- `DSSTPropagator`
- `DSSTPropagatorBuilder`
- `DSSTZonal`
- `DSSTTesseral`
- `DSSTAtmosphericDrag`
- `DSSTSolarRadiationPressure`
- `DSSTThirdBody`

## Attitude

- `AttitudeProvider`
- `InertialProvider`
- `NadirPointing`
- `YawSteering`
- `LofOffset`

## Events / detectors

- `EventDetector`
- `EventsLogger`
- `ApsideDetector`
- `NodeDetector`
- `ElevationDetector`
- `DateDetector`

## TLE

- `TLE constructor`
- `TLEPropagator selectExtrapolator`
- `TLEPropagator`

