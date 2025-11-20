# Station Matching Algorithm Documentation

## Overview

The station matching algorithm pairs APPA air quality stations with Meteo Trentino weather stations, ensuring high-quality data coverage for all required weather variables.

## Algorithm Strategy: 1-to-1 Mapping Constraint

**Key Constraint**: No weather station should be shared between different APPA stations.

**Flexibility**: Each APPA station can use **different weather stations for different variables** (this is allowed and expected).

### Processing Order

APPA stations are processed in order of distance to nearest weather station (closest first). This ensures stations with better options get assigned first.

### Variable-Specific Matching

For each APPA station, for each variable:
1. **Search through weather stations** sorted by distance (closest first)
2. **Skip stations already assigned** to other APPA stations (if 1-to-1 mode enabled)
3. **Check data quality** for each station-variable pair
4. **Prefer unassigned stations**: Give +30% bonus to stations not yet assigned to any APPA station
5. **Quality thresholds**:
   - Preferred: ≥80% valid data (use immediately)
   - Minimum: ≥50% valid data (use if no better option)
   - Reject: 0% valid data (skip)
6. **Assignment tracking**: Once a weather station is assigned to an APPA station, it cannot be used by other APPA stations

### Example Scenario

**APPA Station 402203**:
- Temperature → T0189 (closest unused station with good temp data)
- Rain → T0189 (same station, OK - one APPA can use same station for multiple variables)
- Wind Speed → T0414 (different station, OK - different variables can use different stations)
- Wind Direction → T0414 (same station)
- Pressure → T0414 (same station)
- Radiation → T0327 (different station)
- Humidity → T0135 (different station)

**Result**: APPA 402203 uses 4 different weather stations (T0189, T0414, T0327, T0135), but none of these stations are used by other APPA stations.

## Quality Assessment

### Quality Codes
- **1**: Good/valid data ✅
- **140, 145**: Uncertain/unvalidated data ⚠️ (counted as invalid)
- **151, 255**: Missing/no data ❌

### Quality Calculation
```
valid_percent = (valid_records / total_records) * 100
```
Where:
- `valid_records` = records with quality code = 1 AND value not null
- `total_records` = all records in date range

## Selection Criteria

### Primary Criteria (in order of importance):
1. **Data Quality**: ≥80% valid data preferred, ≥50% minimum
2. **Variable Availability**: Station must have data file for the variable
3. **Distance**: Closer stations preferred (among equal quality)
4. **Uniqueness**: Unused stations preferred (for 1-to-1 mapping)

### Number of Stations Selected

- **Per APPA station**: 
  - **Ideal**: 1 weather station (has all variables)
  - **Fallback**: 1-7 weather stations (one per variable if needed)
  
- **Total stations checked**: 
  - All stations in distance matrix (typically 207 Meteo Trentino stations)
  - Algorithm checks stations in order of distance until finding suitable match

## Matching Process Flow

```
Sort APPA stations by distance to nearest weather station (closest first)

For each APPA station (in order):
  └─ For each variable:
     ├─ Check stations by distance (closest first)
     ├─ Skip stations already assigned to other APPA stations
     ├─ Verify file exists and has data
     ├─ Calculate quality percentage
     ├─ Apply unassigned bonus (+30% if station not assigned to any APPA station)
     ├─ Select best station (≥80% preferred, ≥50% minimum)
     └─ Assign station to this APPA station (prevents reuse by other APPA stations)
     
Result: Each APPA station gets dedicated weather stations (no sharing between APPA stations)
        But each APPA station can use different weather stations for different variables
```

## Output

The algorithm produces:
- **Variable matching CSV**: Maps each APPA station-variable pair to a weather station
- **Coverage analysis**: Valid data percentage for each match
- **Summary statistics**: 
  - Number of 1-to-1 mappings achieved
  - Number of shared weather stations
  - Coverage percentages per variable

## Example

**APPA Station 402203 (Monte Gazza)**:
- Temperature → T0189 (closest unused station with ≥80% quality)
- Rain → T0189 (same station - OK, one APPA can use same station for multiple variables)
- Wind Speed → T0414 (different station, closest unused with good wind data)
- Wind Direction → T0414 (same station)
- Pressure → T0414 (same station)
- Radiation → T0327 (different station, closest unused with radiation data)
- Humidity → T0135 (different station, closest unused with humidity data)

**Result**: Uses 4 different weather stations (T0189, T0414, T0327, T0135)
**Constraint**: None of these stations can be used by other APPA stations

## Performance Considerations

- **Quality checking**: Loads CSV files to assess data quality (can be slow)
- **Parallelization**: Coverage analysis runs in parallel (configurable with `--n-jobs`)
- **Caching**: Extracted CSV files are cached (not re-extracted unless `--force-reextract`)

## Configuration

- `prefer_one_to_one=True`: Enables 1-to-1 preference (default)
- `start_date`, `end_date`: Date range for quality assessment
- Quality thresholds: Hardcoded (80% preferred, 50% minimum)

