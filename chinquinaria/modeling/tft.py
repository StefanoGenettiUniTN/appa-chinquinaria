try:
    import lightning.pytorch as pl
    import pandas as pd
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.tuner import Tuner
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    print("Everything imported smoothly. Proceed to class declaration...\n")
except Exception as e:
    raise ImportError(f"Import failed, check out src. Error: {e}")

CONFIG = {
    "start_training_date": "2013-01-01",
    "end_training_date": "2023-02-01",
    "start_testing_date": "2023-02-01",
    "end_testing_date": "2024-02-01",
    "max_encoder_length": 60,
    "max_prediction_length": 20,
    "num_of_workers": 4,
    "max_epochs": 30,
    "accelerator": "gpu",
    "enable_model_summary": True,
    "gradient_clip_val": 1e-1,
    "monitor": "val_loss",
    "min_delta": 1e-4,
    "patience": 10,
    "verbose": False,
    "mode": "min",
    "suggested_lr": True,
    "batch_size": 128,
    "seed": 42,
    "learning_rate": 0.03,
    "hidden_size": 8,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "optimizer": "ranger",
    "lr_find_max": 10.0,
    "lr_find_min": 1e-6,
}


def load_data() -> pd.DataFrame:
    file_id = "1iIOLm-jpBpZWl9kkKVxhFD1H3rrUgw1k"
    df = pd.read_csv(f"https://drive.google.com/uc?id={file_id}")
    return df


def preprocess_timeseries_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for TimeSeriesDataSet creation and plotting.
    Returns a copy with normalized column names, datetime column,
    and a monotonic time_idx per station.
    """
    df = dataset.copy()
    df["data"] = pd.to_datetime(df["Data"], format="%Y-%m-%d")
    df.sort_values("Data", inplace=True)
    df["time_idx"] = df.groupby("Stazione_APPA").cumcount()
    df.columns = df.columns.str.replace(".", "_", regex=False)
    return df


def create_ts_dataset_with_covariates(
    dataset,
    max_encoder_length,
    max_prediction_length,
    val,
    test,
    preprocess=True,
) -> TimeSeriesDataSet:
    try:
        if preprocess:
            dataset = preprocess_timeseries_dataframe(dataset)
        context_length = max_encoder_length
        prediction_length = max_prediction_length
        if val:
            last_idx_per_series = dataset.groupby("Stazione_APPA")["time_idx"].max()
            global_last_idx = last_idx_per_series.min()
            training_cutoff = global_last_idx - max_prediction_length
            training = TimeSeriesDataSet(
                dataset[lambda x: x.time_idx <= training_cutoff],
                group_ids=["Stazione_APPA"],
                target="PM10_(ug_m-3)",
                time_idx="time_idx",
                min_encoder_length=context_length,
                max_encoder_length=context_length,
                min_prediction_length=prediction_length,
                max_prediction_length=prediction_length,
                static_categoricals=["Stazione_APPA"],
                static_reals=["Latitudine", "Longitudine"],
                time_varying_known_reals=[
                    "Precipitazione_(mm)",
                    "Temperatura_(°C)",
                    "Umid_relativa_(%)",
                    "Direzione_Vento_media_(°)",
                    "Vel_Vento_media_(m/s)",
                    "Pressione_Atm_(hPa)",
                    "Radiaz_Solare_tot_(kJ/m2)",
                    "Humidity_550hPa",
                    "Humidity_950hPa",
                    "Temperature_550hPa",
                    "Temperature_850hPa",
                    "Temperature_950hPa",
                    "Uwind_550hPa",
                    "Uwind_850hPa",
                    "Uwind_950hPa",
                    "Vwind_550hPa",
                    "Vwind_850hPa",
                    "Vwind_950hPa",
                    "boundary_layer_height",
                ],
                time_varying_unknown_reals=[
                    "PM10_(ug_m-3)",
                    "BG_Calusco_D_Adda_PM10_(ug_m-3)",
                    "BG_Osio_Sotto_PM10_(ug_m-3)",
                    "BG_Treviglio_PM10_(ug_m-3)",
                    "BG_Via_Garibaldi_PM10_(ug_m-3)",
                    "BL_Area_Feltrina_PM10_(ug_m-3)",
                    "BL_Parco_Città_di_Bologna_PM10_(ug_m-3)",
                    "BL_Pieve_D_Alpago_PM10_(ug_m-3)",
                    "BS_Palazzo_del_Broletto_PM10_(ug_m-3)",
                    "BS_Sarezzo_PM10_(ug_m-3)",
                    "CR_Piazza_Cadorna_PM10_(ug_m-3)",
                    "FE_Corso_Isonzo_PM10_(ug_m-3)",
                    "LC_Valmadrera_PM10_(ug_m-3)",
                    "MN_Ponti_sul_Mincio_PM10_(ug_m-3)",
                    "MN_Sant_Agnese_PM10_(ug_m-3)",
                    "MO_Via_Ramesina_PM10_(ug_m-3)",
                    "PD_Alta_Padovana_PM10_(ug_m-3)",
                    "PD_Arcella_PM10_(ug_m-3)",
                    "PD_Este_PM10_(ug_m-3)",
                    "PD_Granze_PM10_(ug_m-3)",
                    "PR_Via_Saragat_PM10_(ug_m-3)",
                    "RE_San_Rocco_PM10_(ug_m-3)",
                    "RO_Largo_Martiri_PM10_(ug_m-3)",
                    "TV_Conegliano_PM10_(ug_m-3)",
                    "TV_Mansuè_PM10_(ug_m-3)",
                    "TV_Via_Lancieri_di_Novara_PM10_(ug_m-3)",
                    "VE_Sacca_Fisola_PM10_(ug_m-3)",
                    "VE_Via_Tagliamento_PM10_(ug_m-3)",
                    "VI_Quartiere_Italia_PM10_(ug_m-3)",
                    "VR_Borgo_Milano_PM10_(ug_m-3)",
                    "VR_Bosco_Chiesanuova_PM10_(ug_m-3)",
                    "VR_Legnago_PM10_(ug_m-3)",
                    "VR_San_Bonifacio_PM10_(ug_m-3)",
                ],
            )
            validation = TimeSeriesDataSet.from_dataset(
                training, dataset, min_prediction_idx=training_cutoff + 1
            )
            return training, validation
        if test:
            return TimeSeriesDataSet(
                dataset,
                group_ids=["Stazione_APPA"],
                target="PM10_(ug_m-3)",
                time_idx="time_idx",
                min_encoder_length=context_length,
                max_encoder_length=context_length,
                min_prediction_length=prediction_length,
                max_prediction_length=prediction_length,
                static_categoricals=["Stazione_APPA"],
                static_reals=["Latitudine", "Longitudine"],
                time_varying_known_reals=[
                    "Precipitazione_(mm)",
                    "Temperatura_(°C)",
                    "Umid_relativa_(%)",
                    "Direzione_Vento_media_(°)",
                    "Vel_Vento_media_(m/s)",
                    "Pressione_Atm_(hPa)",
                    "Radiaz_Solare_tot_(kJ/m2)",
                    "Humidity_550hPa",
                    "Humidity_950hPa",
                    "Temperature_550hPa",
                    "Temperature_850hPa",
                    "Temperature_950hPa",
                    "Uwind_550hPa",
                    "Uwind_850hPa",
                    "Uwind_950hPa",
                    "Vwind_550hPa",
                    "Vwind_850hPa",
                    "Vwind_950hPa",
                    "boundary_layer_height",
                ],
                time_varying_unknown_reals=[
                    "PM10_(ug_m-3)",
                    "BG_Calusco_D_Adda_PM10_(ug_m-3)",
                    "BG_Osio_Sotto_PM10_(ug_m-3)",
                    "BG_Treviglio_PM10_(ug_m-3)",
                    "BG_Via_Garibaldi_PM10_(ug_m-3)",
                    "BL_Area_Feltrina_PM10_(ug_m-3)",
                    "BL_Parco_Città_di_Bologna_PM10_(ug_m-3)",
                    "BL_Pieve_D_Alpago_PM10_(ug_m-3)",
                    "BS_Palazzo_del_Broletto_PM10_(ug_m-3)",
                    "BS_Sarezzo_PM10_(ug_m-3)",
                    "CR_Piazza_Cadorna_PM10_(ug_m-3)",
                    "FE_Corso_Isonzo_PM10_(ug_m-3)",
                    "LC_Valmadrera_PM10_(ug_m-3)",
                    "MN_Ponti_sul_Mincio_PM10_(ug_m-3)",
                    "MN_Sant_Agnese_PM10_(ug_m-3)",
                    "MO_Via_Ramesina_PM10_(ug_m-3)",
                    "PD_Alta_Padovana_PM10_(ug_m-3)",
                    "PD_Arcella_PM10_(ug_m-3)",
                    "PD_Este_PM10_(ug_m-3)",
                    "PD_Granze_PM10_(ug_m-3)",
                    "PR_Via_Saragat_PM10_(ug_m-3)",
                    "RE_San_Rocco_PM10_(ug_m-3)",
                    "RO_Largo_Martiri_PM10_(ug_m-3)",
                    "TV_Conegliano_PM10_(ug_m-3)",
                    "TV_Mansuè_PM10_(ug_m-3)",
                    "TV_Via_Lancieri_di_Novara_PM10_(ug_m-3)",
                    "VE_Sacca_Fisola_PM10_(ug_m-3)",
                    "VE_Via_Tagliamento_PM10_(ug_m-3)",
                    "VI_Quartiere_Italia_PM10_(ug_m-3)",
                    "VR_Borgo_Milano_PM10_(ug_m-3)",
                    "VR_Bosco_Chiesanuova_PM10_(ug_m-3)",
                    "VR_Legnago_PM10_(ug_m-3)",
                    "VR_San_Bonifacio_PM10_(ug_m-3)",
                ],
            )
    except Exception as e:
        raise Exception(f"Failed to parse the dataset to ts object: {str(e)}")


def split_train_test(df, training_start_date, training_end_date, testing_start_date, testing_end_date):
    train_df = df.loc[(df["Data"] >= training_start_date) & (df["Data"] < training_end_date)]
    test_df = df.loc[(df["Data"] >= testing_start_date) & (df["Data"] < testing_end_date)]
    print(test_df[test_df["Stazione_APPA"] == "Monte Gaza"].info())
    return train_df, test_df


def tft():
    try:
        print("\n\nClass correctly declared, init data loading...\n\n")
        dataset = load_data()
        train_df, test_df = split_train_test(
            dataset.copy(),
            CONFIG["start_training_date"],
            CONFIG["end_training_date"],
            CONFIG["start_testing_date"],
            CONFIG["end_testing_date"],
        )
        train_df_processed = preprocess_timeseries_dataframe(train_df)
        test_df_processed = preprocess_timeseries_dataframe(test_df)
        print("\n\nData loaded correctly, proced to ts object parsing of pd DataFrame...\n\n")
    except Exception as e:
        raise ValueError(f"Error in loading data: {e}") from e

    try:
        train_tsdataset_with_cov, val_tsdataset_with_cov = create_ts_dataset_with_covariates(
            train_df_processed,
            max_encoder_length=CONFIG["max_encoder_length"],
            max_prediction_length=CONFIG["max_prediction_length"],
            val=True,
            test=False,
            preprocess=False,
        )
        test_tsdataset_with_cov = create_ts_dataset_with_covariates(
            test_df_processed,
            max_encoder_length=CONFIG["max_encoder_length"],
            max_prediction_length=CONFIG["max_prediction_length"],
            val=False,
            test=True,
            preprocess=False,
        )
        print("\n\nDataset parsed correctly to TimeSeriesDataSet Object, URRAY!\n\n")
    except Exception as e:
        raise ValueError(f"Something went wrong during TimeSeriesDataSet creation: {e}") from e

    try:
        pl.seed_everything(CONFIG["seed"])
        trainer = pl.Trainer(
            accelerator=CONFIG["accelerator"],
            max_epochs=CONFIG["max_epochs"],
            gradient_clip_val=CONFIG["gradient_clip_val"],
            enable_model_summary=CONFIG["enable_model_summary"],
            callbacks=EarlyStopping(
                monitor=CONFIG["monitor"],
                min_delta=CONFIG["min_delta"],
                patience=CONFIG["patience"],
                verbose=CONFIG["verbose"],
                mode=CONFIG["mode"],
            ),
        )
        model = TemporalFusionTransformer.from_dataset(
            train_tsdataset_with_cov,
            learning_rate=CONFIG["learning_rate"],
            hidden_size=CONFIG["hidden_size"],
            attention_head_size=CONFIG["attention_head_size"],
            dropout=CONFIG["dropout"],
            hidden_continuous_size=CONFIG["hidden_continuous_size"],
            loss=QuantileLoss(),
            optimizer=CONFIG["optimizer"],
        )
        print("\n\nModel instantiated correctly from the dataset structure\n\n")
    except Exception as e:
        raise ValueError(f"Something went wrong during model instantiation: {e}") from e

    try:
        print("\n\nSwitching data to dataloaders for training...\n\n")
        batch_size = CONFIG["batch_size"]
        train_dataloader = train_tsdataset_with_cov.to_dataloader(
            train=True, batch_size=batch_size, num_workers=CONFIG["num_of_workers"]
        )
        val_dataloader = val_tsdataset_with_cov.to_dataloader(
            train=False, batch_size=batch_size, num_workers=CONFIG["num_of_workers"]
        )
        test_dataloader = test_tsdataset_with_cov.to_dataloader(
            train=False, batch_size=batch_size, num_workers=CONFIG["num_of_workers"]
        )
        print("\n\nDataloaders are ready.\n\n")
    except Exception as e:
        raise ValueError(f"to_dataloader failed: {e}") from e

    if CONFIG["suggested_lr"]:
        res = Tuner(trainer).lr_find(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            max_lr=CONFIG["lr_find_max"],
            min_lr=CONFIG["lr_find_min"],
        )
        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=True, suggest=True)
        fig.show()


if __name__ == "__main__":
    tft()
