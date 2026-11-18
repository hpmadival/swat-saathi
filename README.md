# 💧 SWAT Bharat

**Every SWAT input for any catchment in India, in one zip.**

A free web app that prepares ready-to-load inputs for **SWAT 2012** (ArcSWAT, QSWAT) and
**SWAT+** (QSWAT+, SWAT+ Editor). Upload your watershed or subbasins, or drop a point, choose a
safety margin, press one button, and download one zip.

## Your study area

- **Shapes** (watershed, subbasins; any coordinate system): a margin of a few km all round
  (5 km by default), either following the outline or as a rectangle.
- **Points** (an outlet, a gauge): a circle or a square of the size you choose.
- Margins are worked out in metres in the area's UTM zone, and shown on the map as a dashed line
  before anything is downloaded.

## What you get

| Folder | What is in it | Made from |
|---|---|---|
| `Maps/` | `dem.tif`, `landuse.tif`, `soil.tif` on **one UTM grid** (30 m or 90 m), plus `studyarea.shp`, `places.shp`, `points.shp` | Copernicus DEM GLO-30 / GLO-90; ESA WorldCover 10 m; SoilGrids 2.0 |
| `LookupTables/` | Land use and soil lookup tables for QSWAT / QSWAT+ (`.csv`) and ArcSWAT (`.txt`), with plain-language explanations | |
| `SoilDatabase/` | `usersoil.csv`: one row per soil type in your area, SWAT 2012 usersoil layout (also used by QSWAT+) | SoilGrids + pedotransfer functions |
| `SWAT2012/` | Observed weather (`pcp.txt`, `tmp.txt`, `rh.txt`, `solar.txt`, `wind.txt` + station files) and `WGEN_user.csv` | IMD gridded data, NASA POWER |
| `SWATplus/` | Observed weather (`.cli` lists + `.pcp .tmp .hmd .slr .wnd`) and the weather generator CSVs for SWAT+ Editor | IMD gridded data, NASA POWER |
| `CheckTables/` | All weather as plain CSV files | |

**Map viewer:** after making the inputs, an interactive map shows the DEM, land use and soil as
layers you can switch on and off over satellite or street maps, with the study area, your places
and the weather stations, plus colour legends.

A `README.txt` inside the zip gives step-by-step loading instructions for each SWAT interface.

## How the soil database is made

1. SoilGrids sand, silt, clay, bulk density, organic carbon, gravel and pH are read at five depths (0-100 cm).
2. Each 250 m cell gets a **soil type** from its topsoil (0-30 cm) and subsoil (30-100 cm) USDA texture,
   e.g. `SG_CL_C` = clay loam over clay.
3. For each soil type, the cell averages make four SWAT layers (0-150, 150-300, 300-600, 600-1000 mm), with:
   - available water (AWC) and saturated conductivity (Ksat): Saxton and Rawls (2006)
   - USLE K: Williams (1995); moist soil albedo: Baumer (1990)
   - hydrologic soil group: NRCS (2009), from the slowest layer's Ksat

## Land use codes

| ESA WorldCover | SWAT code |
|---|---|
| Tree cover | FRST (or FRSE / FRSD) |
| Shrubland | RNGB |
| Grassland, moss and lichen | RNGE |
| Cropland | AGRL (or RICE, WWHT, CORN, SUGC, COTS, AGRR) |
| Built-up | URMD (or URHD, URLD, UCOM) |
| Bare or sparse vegetation, snow and ice | BARR |
| Water | WATR |
| Herbaceous wetland | WETN |
| Mangroves | WETF |

## Weather

IMD rainfall (0.25°) and temperature (1°), NASA POWER humidity, solar radiation and wind, station
elevations, and the 14 monthly weather generator values. Stations are either every IMD grid cell
in the study area, or one area-averaged station per shape (for subbasins). Missing days are `-99`.

## The files

| File | What it does |
|---|---|
| `app.py` | The screens |
| `study_area.py` | Buffers: circles, squares, outlines, rectangles in km |
| `swat_maps.py` | DEM, land use, soil, lookup tables, usersoil |
| `swat_weather.py` | Weather stations, weather generator, SWAT 2012 and SWAT+ weather files |
| `power_tools.py` | NASA POWER and station elevations |
| `imd_tools.py` | The IMD engine, the same as in the [IMD Data Puller](https://github.com/hpmadival/imd-data-puller) |
| `bundle.py` | The README and the zip |

## Running it

**Online:** sign in at share.streamlit.io with GitHub, choose this repository, branch `main`,
file `app.py`, Python 3.12 under Advanced settings, and Deploy.
Free-server limits: 15,000 km² at 30 m, 150,000 km² at 90 m.

**On your laptop** (no size limits, faster):

```
pip install -r requirements.txt
streamlit run app.py
```

## Data sources (please cite)

- Copernicus DEM GLO-30 / GLO-90 (ESA), via the AWS Registry of Open Data.
- Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. https://doi.org/10.5281/zenodo.7254221
- Poggio, L. et al. (2021). SoilGrids 2.0. *SOIL* 7, 217-240.
- Saxton, K.E. and Rawls, W.J. (2006). *SSSA Journal* 70, 1569-1578.
- Pai, D.S. et al. (2014). *Mausam* 65(1), 1-18. Srivastava, A.K. et al. (2009). *Atmos. Sci. Lett.* 10, 249-254.
- NASA Langley Research Center POWER Project (https://power.larc.nasa.gov).
- Madival, H.P. (2026). IMD Data Puller. https://doi.org/10.5281/zenodo.24352511
