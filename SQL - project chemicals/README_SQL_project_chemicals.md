# Chemicals TRI Emissions Analysis

## Project overview

This project analyzes whether annual releases of hazardous chemicals reported in the TRI dataset decreased over time. The analysis focuses on total TRI pounds by year and identifies the chemicals that contribute the most to total emissions.

The main work is contained in the notebook:

- `chemicals - Anna.ipynb`

## Research question

Despite regulatory measures, have total TRI pounds of hazardous chemicals released annually significantly decreased over time?

## What the project does

The notebook:

1. Loads chemical emissions data from an Excel file.
2. Creates a MySQL table for the chemicals dataset.
3. Inserts the data into the SQL table.
4. Calculates annual total TRI pounds.
5. Runs trend analysis using regression and Pearson correlation.
6. Visualizes annual emissions over time.
7. Identifies the top chemical contributors to total emissions.

## Tools and libraries

- Python
- MySQL
- `mysql.connector`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scipy`

## SQL usage

The project uses SQL/MySQL to create a structured table for the chemicals data and insert the dataset into it. The table includes fields such as submission year, facility ID, facility name, chemical name, TRI pounds, and hazard-related indicators.

Example table structure from the notebook:

```sql
CREATE TABLE IF NOT EXISTS chemicals5 (
    Submission_Year INT NOT NULL,
    TRI_Facility_ID VARCHAR(255) NOT NULL,
    TRI_Facility_Name VARCHAR(255) NOT NULL,
    Chemical VARCHAR(900) NOT NULL,
    TRI_Pounds DECIMAL(38, 2) NOT NULL,
    RSEI_Hazard DECIMAL(38, 2),
    RSEI_Hazard_Cancer DECIMAL(38, 2)
);
```

## Data note

The original Excel dataset is not included in this upload to keep the project small and easy to submit. The notebook shows the expected data structure and the analysis workflow.

Expected input file in the original project:

- `chemicals.xlsx`

## Main conclusion

The regression and correlation analysis indicate a statistically significant decrease in annual TRI pounds over time. However, a small number of chemicals still account for a large share of total emissions, so targeted reductions in those chemicals would be especially important.

## How to run

1. Install the required Python libraries.
2. Prepare a MySQL connection.
3. Place the chemicals Excel file in the expected path or update `file_path` in the notebook.
4. Open and run `chemicals - Anna.ipynb`.

## Author

Anna Korostelyova
