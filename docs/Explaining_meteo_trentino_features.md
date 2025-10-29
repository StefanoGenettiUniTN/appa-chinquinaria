# Meteo Trentino Features - Quick Reference

## Dataset Overview

The Meteo Trentino dataset combines two data sources from weather stations across Trentino, Italy:

1. **Annale Idrologico** (1923-2025): Official, validated, consolidated historical data
2. **Recent Measurements** (1990-2025): Current station readings, pending consolidation

## Key Features

### Precipitation
- **Tot da Annale Idrologico** (1923-2025): Official daily total precipitation. ✅ Validated, reliable for trend analysis.
- **Pioggia** (1990-2025): Recent rainfall measurements. ⚠️ Raw data, good for current monitoring.

### Temperature
- **Min da Annale Idrologico** (1990-2025): Daily minimum temperature. ✅ Official validated extremes.
- **Max da Annale Idrologico** (1990-2025): Daily maximum temperature. ✅ Official validated extremes.
- **Temperatura** (1990-2025): Recent temperature readings. ⚠️ Raw data, pending validation.

| Feature | Period | Status | Use Case |
|---------|--------|--------|----------|
| Tot Annale | 1923-2025 | ✅ Validated | Historical trends, climate analysis |
| Pioggia | 1990-2025 | ⚠️ Raw | Current weather monitoring |
| Min/Max Annale | 1990-2025 | ✅ Validated | Temperature extremes analysis |
| Temperatura | 1990-2025 | ⚠️ Raw | Operational monitoring |

## Quick Rules

- **For historical analysis**: Use Annale Idrologico data (highest confidence)
- **For current conditions**: Use recent measurements (Pioggia, Temperatura)
- **For combined analysis**: Use both, noting validation status
- **Annale data**: Official publications with QA/QC
- **Recent data**: May contain errors, subject to future corrections

---