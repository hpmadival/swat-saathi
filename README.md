# 🏞️ SWAT Saathi: watershed delineation and SWAT input builder for India

**Delineate a watershed and make SWAT input data for any catchment in India in a few minutes:
catchment, streams and subbasins; DEM, land use, soil (usersoil), lookup tables, weather (IMD rainfall
and temperature) and the weather generator (WGEN), in one zip.**

SWAT Saathi is a free web app that prepares ready-to-load inputs for **SWAT 2012** (ArcSWAT, QSWAT)
and **SWAT+** (QSWAT+, SWAT+ Editor). It works in three steps:

1. **Your area**, in one of three ways:
   - 📍 **Point + area**: a circle or square of the size you choose around a point.
   - 🗺️ **Your shape**: upload (or draw) your watershed or subbasins, with an optional safety margin.
   - 🌊 **Delineate**: click your outlet and get the catchment, streams (with Strahler order) and
     subbasins from the Copernicus DEM, with a threshold slider, a subbasin table (area, elevation,
     slope, stream length, flows-into), catchment numbers (area, longest flow path, drainage density,
     shape factors, relief, time of concentration) and flow direction / accumulation grids.
2. **What to make**: maps, soil, weather and the report.
3. **Results**: one zip, a PDF report, a health check, suggested SWAT settings and a map viewer.

If you are searching for **how to prepare SWAT input data**, **SWAT weather input files for India**,
**IMD gridded data for SWAT**, **SWAT usersoil table from SoilGrids**, **land use lookup table for
QSWAT**, or **how to delineate a watershed for SWAT**, this tool does those steps for you.

> SWAT Saathi is an independent tool. It is not the official SWAT software; SWAT is developed by
> Texas A&M AgriLife Research and USDA-ARS (https://swat.tamu.edu).

## Saving you time

- **Delineation suite:** click your outlet and get the catchment (first at 90 m to find the extent,
  then at 30 m for a sharp boundary), the outlet moved onto the main river, streams with Strahler
  order and one subbasin per stream segment, numbered from the outlet upstream with "flows into"
  links (pyflwdir, Deltares). Move the threshold slider to see more or fewer subbasins instantly.
  Download everything as shapefiles (UTM, for QSWAT / ArcSWAT / QGIS), GeoTIFF flow grids, CSV
  tables and GeoJSON (for Google Earth). The subbasins can also be used as "one weather station
  per subbasin".
- **Health check before SWAT:** empty DEM or land use cells, very flat land (burn-in advice), large water
  bodies, tiny soil types, missing weather days, an incomplete weather generator, a short record, and
  whether the traced outlet moved far. Each item says OK, Check or Fix, and what to do.
- **Suggested settings:** a stream threshold for about 20 to 40 subbasins; land use, soil and slope HRU
  thresholds, each with a table of what every threshold would drop; slope class breaks (ready to type
  into QSWAT); and whether to use elevation bands, how many, with TLAPS and PLAPS (PLAPS fitted from the
  IMD rainfall stations when there is a clear link between height and rain).
- **River lines for burn-in:** rivers (and optionally streams and canals) from OpenStreetMap,
  cut to the study area, as `Maps/rivers.shp`, ready for QSWAT / ArcSWAT "burn-in".
- **Methods paragraph:** a ready-to-paste paragraph describing exactly the data and settings used,
  with the full reference list (`Report/Methods paragraph.txt`, also shown in the app).
- **Data report (PDF):** three A4 pages with the elevation, land use and soil maps, land use and soil
  tables, the monthly climate chart, the methods paragraph and references.

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
| `catchment.py` | Tracing a catchment from an outlet point |
| `rivers.py` | River lines from OpenStreetMap |
| `report.py` | Methods paragraph and PDF data report |
| `setup_advice.py` | Health check and suggested SWAT settings |
| `bundle.py` | The README and the zip |

## Look and speed

- Light, Dark or System theme switch at the top right (the same as the IMD Data Puller), with a dark
  base map in dark mode.
- Heavy libraries (imdlib, rasterio, pyflwdir, matplotlib) load only when they are needed, so the
  first page opens in about 1.5 seconds once the app is awake.
- **Keep it awake:** Streamlit's free hosting puts an app to sleep after about 12 hours without
  visitors, and the next visitor then waits a minute or two. The file
  `.github/workflows/keep-awake.yml` is a small GitHub robot that opens the app every 6 hours.
  Put your app's real link on its `APP_URL` line. If you upload through the GitHub website, create
  the file with exactly that path (type `.github/workflows/keep-awake.yml` as the file name).

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
- OpenStreetMap contributors (river lines, ODbL licence).
- Eilander, D. et al. (2021). pyflwdir. *HESS* 25, 5287-5313.
- Madival, H.P. (2026). IMD Data Puller. https://doi.org/10.5281/zenodo.24352511

## Finding this tool (for the GitHub page)

Suggested GitHub "About" description: *Watershed delineation and SWAT input builder for India: catchment,
streams and subbasins; DEM, land use, soil (usersoil), lookup tables, IMD weather and weather generator
for SWAT 2012 and SWAT+, in one zip.*
Suggested topics: `swat` `swat-plus` `qswat` `arcswat` `hydrology` `watershed` `india` `imd`
`soilgrids` `watershed-delineation` `hydrological-modelling` `streamlit`
