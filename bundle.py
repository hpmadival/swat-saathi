"""
Packs everything SWAT Bharat made (maps, lookup tables, soil database, weather)
into one zip, with a README that explains each file and how to load it.
"""

import datetime
import io
import zipfile

APP = "SWAT Bharat"

HOW_TO = """HOW TO LOAD IT
--------------
QSWAT (SWAT 2012) and QSWAT+ (SWAT+)
  1. New project. DEM: Maps/dem.tif. Optionally use Maps/studyarea.shp as the mask
     and Maps/points.shp to place your outlet near.
  2. Land use: Maps/landuse.tif with LookupTables/landuse lookup QSWAT.csv
     (columns LANDUSE_ID, SWAT_CODE).
  3. Soil: Maps/soil.tif with LookupTables/soil lookup QSWAT.csv (SOIL_ID, SNAM),
     and choose the "usersoil" option. First add the rows of SoilDatabase/usersoil.csv
     to the usersoil table of the project database (QSWAT: the project .mdb, using
     Microsoft Access; QSWAT+: the project .sqlite, using DB Browser for SQLite,
     File > Import > Table from CSV, then copy the rows into usersoil).
  4. Weather: see below.
ArcSWAT (SWAT 2012)
  1. DEM: Maps/dem.tif. Land use and soil: the .tif maps with the ArcSWAT .txt
     lookup tables in LookupTables (choose "Table" lookup, then the .txt file).
  2. Soil: paste SoilDatabase/usersoil.csv rows into the usersoil table of
     SWAT2012.mdb (Microsoft Access) before the HRU step; choose "User Soil".
  3. Weather generator: paste SWAT2012/WeatherGenerator/WGEN_user.csv rows into the
     WGEN_user table of SWAT2012.mdb.
  4. Write Input Tables > Weather Stations: WGEN_user for the weather generator,
     then "Simulation" for rainfall, temperature, relative humidity, solar radiation
     and wind, picking pcp.txt, tmp.txt, rh.txt, solar.txt and wind.txt from
     SWAT2012/ObservedWeather.
SWAT+ Editor (after QSWAT+)
  1. Weather Generator > Import data > CSV files: SWATplus/WeatherGenerator/
     wgn-stations.csv and wgn-monthly-values.csv.
  2. Weather Stations > Import data > SWAT+ format: the folder
     SWATplus/ObservedWeather. The Editor matches each part of the watershed to
     its nearest station.
Tip: unzip to a short path with no spaces (e.g. C:/SWAT/myproject): ArcSWAT and
QSWAT dislike spaces in folder names.
"""

GOOD_TO_KNOW = """GOOD TO KNOW
------------
- All maps share one grid in the study area's UTM zone, so they line up cell for cell.
- The DEM is a surface model (Copernicus): in dense forest and towns it includes some
  tree and building height. QSWAT/ArcSWAT stream delineation works well with it.
- Land use classes come from satellite images (ESA WorldCover): cropland is one class,
  so all crops share one SWAT code (change it in the lookup table if you know better).
- Soil types are made from SoilGrids (a global 250 m soil map made by machine learning).
  Check them against local soil survey (NBSS&LUP) maps when you can.
- Weather: -99 means "no value"; SWAT fills those days from its weather generator.
  Relative humidity is a fraction (0.75 = 75 %). Wind is NASA POWER's 2 m wind.
- The largest half-hour rain (RAINHHMX / pcp_hhr) cannot be measured from daily data;
  it is estimated from the largest daily rain with IMD's rule P(t) = P(24 h) x (t/24)^(1/3).
"""

SOURCES = """DATA SOURCES (please cite)
--------------------------
- Elevation: Copernicus DEM GLO-30 / GLO-90 (ESA, 2021), from the AWS Registry of Open Data.
- Land use: Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200.
  https://doi.org/10.5281/zenodo.7254221 (2020 map: v100, doi 10.5281/zenodo.5571936)
- Soil: Poggio, L. et al. (2021). SoilGrids 2.0: producing soil information for the globe
  with quantified spatial uncertainty. SOIL 7, 217-240.
- Soil properties: Saxton, K.E. and Rawls, W.J. (2006). Soil water characteristic estimates
  by texture and organic matter for hydrologic solutions. SSSA Journal 70, 1569-1578.
  Williams, J.R. (1995). The EPIC model (USLE K). NRCS (2009). National Engineering
  Handbook Part 630, Chapter 7: Hydrologic Soil Groups.
- Rainfall: Pai, D.S. et al. (2014). Mausam 65(1), 1-18 (IMD 0.25 degree gridded rainfall).
- Temperature: Srivastava, A.K. et al. (2009). Atmospheric Science Letters 10, 249-254.
- Humidity, solar radiation, wind: NASA Langley Research Center POWER Project
  (https://power.larc.nasa.gov).
- Station elevations: Copernicus DEM GLO-90, via Open-Meteo.
- IMD download engine: Madival, H.P. (2026). IMD Data Puller.
  https://doi.org/10.5281/zenodo.24352511
"""


def readme(area_text, maps=None, weather=None, today=None):
    today = today or datetime.date.today()
    out = [f"SWAT inputs made with {APP}", "=" * 40, f"Made on {today:%d-%m-%Y}.", "",
           "STUDY AREA", "----------", area_text, ""]
    out += ["WHAT IS IN THIS ZIP", "-------------------"]
    if maps is not None:
        out += ["Maps/            dem.tif, landuse.tif, soil.tif (GeoTIFF, same grid),",
                "                 studyarea.shp (your area + buffer), places.shp / points.shp (yours)",
                "LookupTables/    land use and soil lookup tables, QSWAT (.csv) and ArcSWAT (.txt),",
                "                 and plain-language explanations of every class and soil type",
                "SoilDatabase/    usersoil.csv (one row per soil type, SWAT 2012 usersoil layout,",
                "                 also used by QSWAT+), soil types explained.csv"]
    if weather is not None:
        out += ["SWAT2012/        ObservedWeather (pcp.txt, tmp.txt, rh.txt, solar.txt, wind.txt",
                "                 + one file per station), WeatherGenerator/WGEN_user.csv",
                "SWATplus/        ObservedWeather (.cli lists + .pcp .tmp .hmd .slr .wnd files),",
                "                 WeatherGenerator (CSV files for SWAT+ Editor, weather-wgn.cli)",
                "CheckTables/     the weather as plain CSV files, to look at in Excel"]
    out += ["", HOW_TO, GOOD_TO_KNOW]
    if maps is not None:
        out += ["MAPS", "----"] + [f"{k}: {v}" for k, v in maps.settings.items()]
        if maps.landuse_lookup is not None:
            out += ["", "Land use classes:", maps.landuse_lookup.to_string(index=False)]
        if maps.soil_units is not None:
            out += ["", "Soil types:", maps.soil_units[["Soil code", "Soil name", "What it is", "Area (%)",
                                                        "Hydrologic group"]].to_string(index=False)]
        for x in maps.problems + maps.notes:
            out.append(f"- {x}")
        out.append("")
    if weather is not None:
        out += ["WEATHER", "-------", f"Period: {weather['period']}", f"Stations: {weather['counts']}"]
        out += [f"{k}: {v}" for k, v in weather["settings"].items()]
        out += [f"- {x}" for x in weather["cautions"]]
        out += ["", weather["stations"], ""]
    out.append(SOURCES)
    text = "\n".join(out).replace("°", " deg").replace("²", "2")
    return text.encode("ascii", "ignore").decode().replace("\n", "\r\n")


def build_zip(files, readme_text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.txt", readme_text.encode("utf-8"))
        for path, data in files.items():
            z.writestr(path, data if isinstance(data, bytes) else data.encode("utf-8"))
    return buf.getvalue()
