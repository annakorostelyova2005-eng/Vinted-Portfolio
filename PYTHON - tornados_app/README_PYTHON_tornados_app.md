# USA Tornado Analysis and Streamlit App

## Project overview

This project analyzes tornado events in the United States in the 21st century and turns the analysis into an interactive Streamlit app. The app explores tornado locations, damage, injuries, deaths, and model-based predictions.

The project includes:

- exploratory data analysis notebook
- Streamlit app code
- documentation for the tornado dataset
- trained model files used by the app

## Main files

- `app creation/tornados.py` — main Streamlit app
- `tornados_starter.ipynb` — analysis and modeling notebook
- `app creation/tornados_docs.md` — dataset documentation
- `app creation/tornados_background_light.png` — app background image
- `link to the app.txt` — deployed app link

## Data source

The dataset is based on NOAA/NCDC Storm Events data and combines tornado-related files for the United States in the 21st century.

Dataset source mentioned in the project:

- NOAA Storm Events database

## What the project does

The project:

1. Loads and cleans tornado event data.
2. Converts damage values into numeric format.
3. Creates additional features such as tornado area and path distance.
4. Visualizes tornado frequency and impact by state.
5. Analyzes injuries, deaths, property damage, and crop damage.
6. Builds predictive models for tornado outcomes.
7. Presents results in an interactive Streamlit application.

## Tools and libraries

- Python
- Streamlit
- pandas
- numpy
- DuckDB
- Plotly
- requests
- joblib
- scikit-learn
- geopy
- matplotlib
- seaborn

## How to run the app

Install the required libraries, then run:

```bash
streamlit run tornados.py
```

If running from the project root, first move into the app folder:

```bash
cd "app creation"
streamlit run tornados.py
```

## Important note about large files

For application submission, large data and model files may need to be removed because some upload forms have strict file-size limits. If those files are missing, the project is still useful as a code sample, but the app may not run until the data/model artifacts are restored.

The main large files in the original project are:

- `usa_tornados_xxi.csv`
- `tornado_evolution_model.pkl`

If these are removed, they can be regenerated or restored from the original project files.

## Deployed app

The project includes a text file with the deployed app link:

```text
tornados.streamlit.app
```

## Author

Anna Korostelyova
