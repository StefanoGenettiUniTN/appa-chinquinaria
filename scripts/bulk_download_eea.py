from typing import Dict, List, Set, Tuple
import os
import zipfile
import pandas as pd
import argparse
import datetime
import requests
from pathlib import Path

def extract_parquet_from_zip(zip_path: str, extract_to: str) -> List[str]:
    """
    Extracts parquet files from a zip archive.
    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract files to.
    Returns:
        List[str]: List of paths to extracted parquet files.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(extract_to)
    parquet_files = [os.path.join(f"{extract_to}/E1a", f) for f in os.listdir(f"{extract_to}/E1a") if f.endswith('.parquet')]
    return parquet_files

def parquet_to_csv(parquet_files: List[str], output_csv: str) -> None:
    """
    Converts a list of parquet files to a single CSV file.
    Args:
        parquet_files (List[str]): List of paths to parquet files.
        output_csv (str): Path to the output CSV file.
    """
    df_list = [pd.read_parquet(pf) for pf in parquet_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")

def merge_metadata(df_eea: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    """
    Merges the EEA data with the metadata.
    Args:
        df_eea (pd.DataFrame): DataFrame containing EEA data.
        metadata_path (str): Path to the metadata CSV file.
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    df_metadata = pd.read_csv(metadata_path)

    # The Samplingpoint field contains "IT/SPO.IT2155A_5_BETA_2014-09-01_00:00:00"
    # We need to extract the Sampling Point Id part after "IT/"
    df_eea['station-id'] = df_eea['Samplingpoint'].str.split("IT/").str[-1]
    df_metadata['station-id'] = df_metadata["Sampling Point Id"]

    df_merged = pd.merge(df_eea, df_metadata, on="station-id", how="left", indicator=True)

    # rows that didn’t merge (no match in metadata)
    df_unmatched = df_merged[df_merged["_merge"] == "left_only"]
    print(f"Unmatched rows saved as unmatched_measurements.csv (total: {len(df_unmatched)})")

    return df_merged

def download_eea( output_folder: str,
                  metadata_path: str,
                  api_countries: List[str] = None,
                  api_cities: List[str] = None,
                  api_pollutants: List[str] = None,
                  api_dateTimeStart: str = None,
                  api_dateTimeEnd: str = None,
                  api_aggregationType: str = None,
                  zip_path: str = None) -> None:
    """
    Downloads EEA data function
    Args:
        output_folder (str): Path of the output folder.
        api_countries (List[str]): List of country codes to fetch data for.
        api_cities (List[str]): List of city names to fetch data for.
        api_pollutants (List[str]): List of pollutants to fetch data for.
        api_dateTimeStart (str): Start date for data retrieval in Format: yyyy-mm-dd.
        api_dateTimeEnd (str): End date for data retrieval in Format: yyyy-mm-dd.
        api_aggregationType (str): Aggregation type to filter data.
        metadata_path (str): Path of the metadata csv file.
        zip_path (str, optional): Path of the downloaded zip. Defaults to None.
    """
    output_csv = f"{output_folder}/eea_data.csv"
    apiUrl = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
    endpoint = "ParquetFile"

    # collect parquet files
    if zip_path is None:
        # get parquet zip from API
        zip_path = f"{output_folder}/eea_data.zip"

        # request body
        request_body = {
            "countries": api_countries if api_countries else [],
            "cities": api_cities if api_cities else [],
            "pollutants": api_pollutants if api_pollutants else [],
            "dataset": 2,
            "dateTimeStart": f"{api_dateTimeStart}T00:00:00.000Z",
            "dateTimeEnd": f"{api_dateTimeEnd}T23:59:00.000Z",
            "aggregationType": f"{api_aggregationType}",
            "email": None
        }

        # a get request to the API
        print("Requesting zip file from API...")
        downloadFile = requests.post(apiUrl+endpoint, json=request_body).content
        
        # store in local path
        output = open(zip_path, 'wb')
        output.write(downloadFile)
        output.close()
        print(f"Downloaded zip file from API and saved to {zip_path}")

    parquet_files: List[str] = extract_parquet_from_zip(zip_path, output_folder)
    print(f"Found {len(parquet_files)} parquet files")

    # convert to csv
    parquet_to_csv(parquet_files, output_csv)

    # merge with metadata
    df_eea = pd.read_csv(output_csv)
    df_merged = merge_metadata(df_eea, metadata_path)
    df_merged.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to {output_csv}")

    # columns to extract
    columns_to_keep = [
        "station-id",
        "Start",
        "End",
        "Value",
        "Unit",
        "AggType",
        "Country",
        "Air Pollutant",
        "Longitude",
        "Latitude",
        "Altitude",
        "Altitude Unit",
        "Air Quality Station Area",
        "Air Quality Station Type",
        "Municipality",
        "Duration Unit",
        "Cadence Unit"
    ]
    df_filtered = df_merged[columns_to_keep]
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved as {output_csv}")


def read_arguments():
    parser = argparse.ArgumentParser(description="Python to get csv from parquet file from the European Environment Agency.")
    
    parser.add_argument('--zip_path', type=str, help='Path of the downloaded zip.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path of the output folder.')
    parser.add_argument('--output_csv', type=str, required=True, help='Name of the output csv.')
    parser.add_argument('--metadata', type=str, required=True, help='Path of the metadata csv file downloaded from https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements#.')

    parser.add_argument('--api_countries', type=str, nargs='+', help='List of country codes to fetch data for (e.g., IT,FR,DE). If not provided, data for all countries will be fetched.')
    parser.add_argument('--api_cities', type=str, nargs='+', help='List of city names to fetch data for (e.g., Rome,Paris,Berlin). If not provided, data for all cities will be fetched.')
    parser.add_argument('--api_pollutants', type=str, nargs='+', help='List of pollutants to fetch data for (e.g., NO2,PM10,PM2.5). If not provided, data for all pollutants will be fetched.')
    parser.add_argument('--api_dateTimeStart', type=str, help='Start date for data retrieval in Format: yyyy-mm-dd. Example: 2024-05-28. (default: If dataTimeStart and dateTimeEnd parameters are not included in the request, the filter for the temporal coverage will not be applied and the entire set of data will be downloaded.).')
    parser.add_argument('--api_dateTimeEnd', type=str, help='End date for data retrieval in Format: yyyy-mm-dd. Example: 2024-05-28. (default: If dataTimeStart and dateTimeEnd parameters are not included in the request, the filter for the temporal coverage will not be applied and the entire set of data will be downloaded.).')
    parser.add_argument('--api_aggregationType', type=str, help="Aggregation type to filter data (e.g., hour, day).")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    zip_path = args['zip_path']
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Always save in data/eea-data folder in project root
    project_root = Path(__file__).parent.parent
    eea_data_dir = project_root / "data" / "eea-data"
    output_folder = f"{eea_data_dir}/{current_datetime}"
    os.makedirs(output_folder, exist_ok=True)
    output_csv = f"{output_folder}/{args['output_csv']}"
    apiUrl = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
    pollutant_dict = {"PM10": "PM10"}
    endpoint = "ParquetFile"

    # collect parquet files
    if zip_path is None:
        # get parquet zip from API
        zip_path = f"{output_folder}/metadata.zip"

        # request body
        request_body = {
            "countries": args['api_countries'] if args['api_countries'] else [],
            "cities": args['api_cities'] if args['api_cities'] else [],
            "pollutants": [pollutant_dict[p] for p in args['api_pollutants']] if args['api_pollutants'] else [],
            "dataset": 2,
            "dateTimeStart": f"{args['api_dateTimeStart']}T00:00:00.000Z",
            "dateTimeEnd": f"{args['api_dateTimeEnd']}T23:59:00.000Z",
            "aggregationType": f"{args['api_aggregationType']}",
            "email": None
        }

        # a get request to the API
        print("Requesting zip file from API...")
        downloadFile = requests.post(apiUrl+endpoint, json=request_body).content
        
        # store in local path
        output = open(zip_path, 'wb')
        output.write(downloadFile)
        output.close()
        print(f"Downloaded zip file from API and saved to {zip_path}")

    parquet_files: List[str] = extract_parquet_from_zip(zip_path, output_folder)
    print(f"Found {len(parquet_files)} parquet files")

    # convert to csv
    parquet_to_csv(parquet_files, output_csv)

    # merge with metadata
    df_eea = pd.read_csv(output_csv)
    df_merged = merge_metadata(df_eea, args['metadata'])
    df_merged.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to {output_csv}")

    # columns to extract
    columns_to_keep = [
        "station-id",
        "Start",
        "End",
        "Value",
        "Unit",
        "AggType",
        "Country",
        "Air Pollutant",
        "Longitude",
        "Latitude",
        "Altitude",
        "Altitude Unit",
        "Air Quality Station Area",
        "Air Quality Station Type",
        "Municipality",
        "Duration Unit",
        "Cadence Unit"
    ]
    df_filtered = df_merged[columns_to_keep]
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved as {output_csv}")
    