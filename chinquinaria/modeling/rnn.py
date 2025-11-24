"""
RNN model implementation for pollutant forecasting. The subsequent implementation 
was created following closely the documentation of pytorch-forecasting tutorials
that can be found at the subsequent link: 
https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/building.html
"""
from typing import Dict, List, Tuple

try:
    import os
    import re
    import torch
    import pandas as pd
    import lightning.pytorch as pl
    import matplotlib.pyplot as plt
    from lightning.pytorch.tuner import Tuner
    from pytorch_forecasting import TimeSeriesDataSet
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import Baseline, TimeSeriesDataSet
    from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
    from pytorch_forecasting.models.base import AutoRegressiveBaseModelWithCovariates
    from pytorch_forecasting.metrics import MAE, RMSE
    #print("\n\nEverything imported smoothly. Proceed to class declaration...\n\n")
except Exception as e:
    raise ImportError(f"Import failed, check out src. Error: {e}")

CONFIG_RNN = {
    "start_training_date": "2013-01-01",
    "end_training_date": "2023-02-01",
    "start_testing_date": "2023-02-01",
    "end_testing_date": "2024-02-01",
    "max_encoder_length": 30,
    "max_prediction_length": 10,
    "val":False,
    "test":True,
    "n_layers": 2,
    "hidden_size": 10,
    "num_of_workers": 0,
    "run_mode": "Test",
    "max_epochs": 30,
    "accelerator": "gpu",
    "enable_model_summary": True,
    "gradient_clip_val": 1e-1,
    "monitor": "val_loss",
    "min_delta": 1e-4,
    "patience": 10,
    "verbose": False,
    "mode": "min",
    "dropout":0.1,
    "plots_dir": "plots",
    "sample_plot": False,
    "full_test_plot": True,
}

def _safe_filename(name: str) -> str:
    """
    Sanitize variable names for filesystem use.
    Replaces any non-alphanumeric character with underscore.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class LSTModel(AutoRegressiveBaseModelWithCovariates):
    def __init__(
            self,
            target: str,
            target_lags: Dict[str, Dict[str, int]],
            n_layers: int,
            hidden_size: int,
            x_reals: List[str],
            x_categoricals: List[str],
            embedding_sizes: Dict[str, Tuple[int, int]],
            embedding_labels: Dict[str, List[str]],
            static_categoricals: List[str],
            static_reals: List[str],
            time_varying_categoricals_encoder: List[str],
            time_varying_categoricals_decoder: List[str],
            time_varying_reals_encoder: List[str],
            time_varying_reals_decoder: List[str],
            embedding_paddings: List[str],
            categorical_groups: Dict[str, List[str]],
            dropout: float = CONFIG_RNN["dropout"],            
            **kwargs
    ):
        # mandatory calls
        self.save_hyperparameters(ignore=['loss', 'logging_metrics'])
        super().__init__(**kwargs)

        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )
        n_features = sum(
            embedding_size
            for classes_size, embedding_size in self.hparams.embedding_sizes.values()
        ) + len(self.reals)

        # use version of LSTM that can handle zero-length sequences
        self.lstm = LSTM(
            hidden_size=self.hparams.hidden_size, 
            input_size=n_features,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True, # [batch_size, seq_len, input_size]
        )
        self.output_layer = torch.nn.Linear(self.hparams.hidden_size, 1)

    def encode(self, x: Dict[str, torch.Tensor]):
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_features"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        input_vector = input_vector[:, 1:]
        effective_encoder_lengths = x["encoder_lengths"] - 1
        _, hidden_state = self.lstm(
            input_vector,
            lengths = effective_encoder_lengths,
            enforce_sorted = False, 
        )
        return hidden_state
    
    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        input_vector = x["decoder_features"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        last_encoder_target = x["encoder_features"][
            torch.arange(x['encoder_features'].size(0), device=x["encoder_features"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target

        if self.training:
            lstm_output, _ = self.lstm(
                input_vector,
                hidden_state,
                lengths=x["decoder_lengths"],
                enforce_sorted=False
            )
            prediction = self.output_layer(lstm_output)
            prediction = self.transform_output(
                prediction, target_scale=x['target_scale']
            )

            return prediction
        else: # inference
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                lstm_output, hidden_state = self.lstm(x, hidden_state)
                prediction = self.output_layer(lstm_output)[:, 0]
                return prediction, hidden_state
            
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1), # batch size
            )

            return output
    
    def forward(self, x:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_lengths"].size(0)
        
        encoder_embeddings = self.input_embeddings(
            x["encoder_cat"]
        )  # returns dictionary with embedding tensors
        # Concatenate continuous features with relevant embeddings
        lstm_input = torch.cat(
            [x["encoder_cont"]]
            + [
                emb
                for name, emb in encoder_embeddings.items()
                if name in self.encoder_variables or name in self.static_variables
            ],
            dim=-1,
        )
        x["encoder_features"] = lstm_input

        decoder_embeddings = self.input_embeddings(x["decoder_cat"])
        lstm_output = torch.cat(
            [x["decoder_cont"]]
            + [
                emb
                for name, emb in decoder_embeddings.items()
                if name in self.decoder_variables or name in self.static_variables
            ],
            dim=-1,
        )
        x["decoder_features"] = lstm_output
        
        hidden_state = self.encode(x) # encode to hidden state
        output = self.decode(x, hidden_state)

        return self.to_network_output(prediction=output)
    
    def fit(
        self,
        train_dataloader, 
        val_dataloader,
        checkpoint_dir,
        model_name, 
        **trainer_kwargs,
    ):
        """
            Note: this fit method assumes the hyperparameters of the 
                  of the model already present in memory, if any, are
                  the same that the current caller's routine expects.
                  If not, undefined behaviour.
        """
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, model_name)
            if os.path.exists(checkpoint_path):
                print("\n\nCheckpoint found, loading model...\n\n")
                loaded = LSTModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
                self.load_state_dict(loaded.state_dict())
            else:
                print("\n\nNo checkpoint found, training model...\n\n")
                trainer = pl.Trainer(default_root_dir=checkpoint_dir, **trainer_kwargs, logger=False)
                trainer.fit(
                    self,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                )
                # Canonicalize checkpoint path: move best checkpoint to checkpoint_path
                best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", None)

                if best_model_path and os.path.exists(best_model_path):
                    if best_model_path != checkpoint_path:
                        # move/overwrite to have a stable checkpoint file name
                        os.replace(best_model_path, checkpoint_path)
                    print(f"Model trained and saved to {checkpoint_path}")
                    loaded = LSTModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
                    self.load_state_dict(loaded.state_dict())
                else:
                    raise RuntimeError("Training finished but no best_model_path was found.")
        except Exception as e:
            raise RuntimeError(f"Full training with early stopping failed: {e}") from e

    def get_val_loss(
        self,
        val_dataloader,
    ):
        try:
            predictions = self.predict(
                val_dataloader,
                trainer_kwargs=dict(accelerator=CONFIG_RNN["accelerator"]),
                return_y=True,
            )
            # Ensure both tensors are on the same device
            device = predictions.output.device
            target = predictions.y
            if isinstance(target, tuple):
                target = target[0]
            mae_metric = MAE().to(device)(predictions.output.to(device), target.to(device))
            return mae_metric
        except Exception as e:
            print(f"Failed to get loss to: {e}")

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        # example for dataset validation
        assert (
            dataset.max_prediction_length == dataset.min_prediction_length
        ), "Decoder only supports a fixed length"
        assert (
            dataset.min_encoder_length == dataset.max_encoder_length
        ), "Encoder only supports a fixed length"
        return super().from_dataset(dataset, **kwargs)

    @staticmethod
    def build_full_length_prediction_frame(
            raw_predictions,
            ts_dataset: TimeSeriesDataSet,
            reference_df: pd.DataFrame,
            target_column: str = "PM10_(ug_m-3)",
        ) -> pd.DataFrame:
        """
        Expand overlapping prediction windows into a single dataframe indexed by time_idx.
        """
        if not hasattr(raw_predictions, "output"):
            raise ValueError("raw_predictions should be the object returned by model.predict with mode='raw'")

        prediction = raw_predictions.output.get("prediction")
        if prediction is None:
            raise ValueError("Prediction output missing 'prediction' key")
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        prediction = prediction.detach().cpu()
        if prediction.ndim == 3 and prediction.size(-1) == 1:
            prediction = prediction.squeeze(-1)

        decoder_time_idx = raw_predictions.x["decoder_time_idx"].detach().cpu()
        decoder_lengths = raw_predictions.x.get("decoder_lengths")
        if decoder_lengths is not None:
            decoder_lengths = decoder_lengths.detach().cpu()

        metadata_index = ts_dataset.x_to_index(raw_predictions.x).reset_index(drop=True)
        rows = []
        for row_id, row in metadata_index.iterrows():
            station = row["Stazione_APPA"]
            horizon = decoder_lengths[row_id].item() if decoder_lengths is not None else prediction.size(1)
            for step in range(min(horizon, prediction.size(1))):
                rows.append(
                    {
                        "Stazione_APPA": station,
                        "time_idx": int(decoder_time_idx[row_id, step].item()),
                        "prediction": float(prediction[row_id, step].item()),
                    }
                )

        prediction_df = pd.DataFrame(rows)
        if prediction_df.empty:
            return prediction_df
        prediction_df = (
            prediction_df.groupby(["Stazione_APPA", "time_idx"], as_index=False)["prediction"]
            .mean()
        )

        lookup_columns = ["Stazione_APPA", "time_idx"]
        if "data" in reference_df.columns:
            lookup_columns.append("data")
        if target_column in reference_df.columns:
            lookup_columns.append(target_column)
        lookup_df = (
            reference_df[lookup_columns]
            .drop_duplicates(subset=["Stazione_APPA", "time_idx"])
        )
        prediction_df = prediction_df.merge(
            lookup_df,
            on=["Stazione_APPA", "time_idx"],
            how="left",
        )
        return prediction_df

    @staticmethod
    def plot_full_length_predictions(
        prediction_df: pd.DataFrame,
        target_column: str = "PM10_(ug_m-3)",
        output_path="output"
    ):
        """
        Plot aggregated predictions (and actuals where available) for every station.
        """
        plot_dir = os.path.join(output_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if prediction_df.empty:
            print("No aggregated predictions available to plot.")
            return

        for station, station_df in prediction_df.groupby("Stazione_APPA"):
            station_df = station_df.sort_values("time_idx")
            x_axis = station_df["data"] if "data" in station_df.columns else station_df["time_idx"]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x_axis, station_df["prediction"], label="Prediction", color="tab:orange")
            if target_column in station_df.columns:
                ax.plot(x_axis, station_df[target_column], label="Actual", color="tab:blue", alpha=0.7)
            # Use the first month value for the filename
            if "data" in station_df.columns:
                month = station_df["data"].dt.month.iloc[0]
            else:
                month = "all"
            ax.set_title(f"Period Test {month} - {station}")
            ax.set_xlabel("Date" if "data" in station_df.columns else "time_idx")
            ax.set_ylabel(target_column)
            ax.legend()
            fig.autofmt_xdate()
            safe_station = station.replace("/", "_").replace(" ", "_")
            plot_path = os.path.join(plot_dir, f"{month}_{safe_station}.png")
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"Saved plot for {station} to {plot_path}")


def load_data() -> pd.DataFrame:
    file_id = "1iIOLm-jpBpZWl9kkKVxhFD1H3rrUgw1k"
    # Load the dataset from Google Drive
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
    df.columns = df.columns.str.replace('.', '_', regex=False)
    return df


def create_ts_dataset_with_covariates_for_V5_5(
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
            # Since each stations has its own max.time_idx we need to be creative
            last_idx_per_series = dataset.groupby("Stazione_APPA")["time_idx"].max()
            global_last_idx = last_idx_per_series.min()
            training_cutoff = global_last_idx - max_prediction_length
            """
            ATTENTION: the above choice implies that different timeseries
            for different stations will have unbalanced validation set.
            To solve for this, it is required to compute metrics for
            each stations and then averaging them during validation.
            """
            training = TimeSeriesDataSet(
                    dataset[lambda x: x.time_idx <= training_cutoff],
                    group_ids=["Stazione_APPA"],
                    target="PM10_(ug_m-3)",
                    time_idx='time_idx',
                    min_encoder_length=context_length,
                    max_encoder_length=context_length,
                    min_prediction_length=prediction_length,
                    max_prediction_length=prediction_length,
                    
                    # Static variables (don't change over time for each station)
                    static_categoricals=["Stazione_APPA"],
                    static_reals=["Latitudine", "Longitudine"],
                    # Time-varying known variables (meteorological data - can be forecasted)
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
                        "blh_mean_daily",
                    ],
                
                    # Time-varying unknown variables (PM10 measurements from other stations)
                    # These are not known in the future
                    time_varying_unknown_reals=[
                        "PM10_(ug_m-3)",  # Your target variable
                        # All other stations' PM10 measurements
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
                        "VR_San_Bonifacio_PM10_(ug_m-3)"
                    ]
                )
            validation = TimeSeriesDataSet.from_dataset(
                training, 
                dataset,                    
                min_prediction_idx=training_cutoff+1,
            )
            return training, validation
        if test:
            return TimeSeriesDataSet(
                    dataset,
                    group_ids=["Stazione_APPA"],
                    target="PM10_(ug_m-3)",
                    time_idx='time_idx',
                    min_encoder_length=context_length,
                    max_encoder_length=context_length,
                    min_prediction_length=prediction_length,
                    max_prediction_length=prediction_length,
                    
                    # Static variables (don't change over time for each station)
                    static_categoricals=["Stazione_APPA"],
                    static_reals=["Latitudine", "Longitudine"],
                    # Time-varying known variables (meteorological data - can be forecasted)
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
                        "blh_mean_daily",
                    ],
                
                    # Time-varying unknown variables (PM10 measurements from other stations)
                    # These are not known in the future
                    time_varying_unknown_reals=[
                        "PM10_(ug_m-3)",  # Your target variable
                        # All other stations' PM10 measurements
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
                        "VR_San_Bonifacio_PM10_(ug_m-3)"
                    ]
                )
    except Exception as e:
        raise Exception(f"Failed to parse the dataset to ts object: {str(e)}")


def split_train_test(df, training_start_date, training_end_date, testing_start_date, testing_end_date):
    train_df = df.loc[(df['Data'] >= training_start_date) & (df['Data'] < training_end_date)]
    test_df = df.loc[(df['Data'] >= testing_start_date) & (df['Data'] < testing_end_date)]
    print(test_df[test_df["Stazione_APPA" ] == "Monte Gaza"].info())
    return train_df, test_df

# RNN example of execution
def rnn():
    # ----------------
    # Data loading
    # ----------------
    try:
        print("\n\nClass correctly declared, init data loading...\n\n")
        dataset = load_data()
        train_df, test_df = split_train_test(
            dataset.copy(), 
            CONFIG_RNN["start_training_date"], 
            CONFIG_RNN["end_training_date"],
            CONFIG_RNN["start_testing_date"], 
            CONFIG_RNN["end_testing_date"]
        )
        train_df_processed = preprocess_timeseries_dataframe(train_df)
        test_df_processed = preprocess_timeseries_dataframe(test_df)
        print("\n\nData loaded correctly, proced to ts object parsing of pd DataFrame...\n\n")
    except Exception as e:
        raise ValueError(f"Error in loading data: {e}") from e
    # ----------------
    # TimeSeriesDataSet creation
    # ----------------
    try:
        train_tsdataset_with_cov, val_tsdataset_with_cov = create_ts_dataset_with_covariates_for_V5_5(
            train_df_processed,
            max_encoder_length=CONFIG_RNN["max_encoder_length"],
            max_prediction_length=CONFIG_RNN["max_prediction_length"],
            val=True,
            test=False,
            preprocess=False,
        )
        test_tsdataset_with_cov = create_ts_dataset_with_covariates_for_V5_5(
            test_df_processed,
            max_encoder_length=CONFIG_RNN["max_encoder_length"],
            max_prediction_length=CONFIG_RNN["max_prediction_length"],
            val=False,
            test=True,
            preprocess=False,
        )
        print("\n\nDataset parsed correctly to TimeSeriesDataSet Object, URRAY!\n\n")
    except Exception as e:
        raise ValueError(f"Something went wrong during TimeSeriesDataSet creation: {e}") from e
    # ----------------
    # Model from dataset
    # ----------------
    try:
        model_kwargs = dict(
            n_layers=CONFIG_RNN["n_layers"],
            hidden_size=CONFIG_RNN["hidden_size"],
        )
        model = LSTModel.from_dataset(
            train_tsdataset_with_cov,
            **model_kwargs,
        )
        print("\n\nModel instantiated correctly from the dataset structure\n\n")
    except Exception as e:
        raise ValueError(f"Something went wrong during model instantiation: {e}") from e
    # ----------------
    # Dataloaders
    # ----------------
    try:
        print("\n\nSwitching data to dataloaders for training...\n\n")
        batch_size = 128
        train_dataloader = train_tsdataset_with_cov.to_dataloader(
            train=True, batch_size=batch_size, num_workers=CONFIG_RNN["num_of_workers"]
        )
        val_dataloader = val_tsdataset_with_cov.to_dataloader(
            train=False, batch_size=batch_size, num_workers=CONFIG_RNN["num_of_workers"]
        )
        test_dataloader = test_tsdataset_with_cov.to_dataloader(
            train=False, batch_size=batch_size, num_workers=CONFIG_RNN["num_of_workers"]
        )
        print("\n\nDataloaders are ready.\n\n")
    except Exception as e:
        raise ValueError(f"to_dataloader failed: {e}") from e
    
    # ----------------
    # Try training
    # ----------------
    try:
        trainer_kwargs = dict(
            max_epochs=CONFIG_RNN["max_epochs"],
            accelerator=CONFIG_RNN["accelerator"],
            enable_model_summary=CONFIG_RNN["enable_model_summary"],
            gradient_clip_val=CONFIG_RNN["gradient_clip_val"],
            callbacks=EarlyStopping(
                monitor=CONFIG_RNN["monitor"],
                min_delta=CONFIG_RNN["min_delta"],
                patience=CONFIG_RNN["patience"],
                verbose=CONFIG_RNN["verbose"],
                mode=CONFIG_RNN["mode"],
            ),
            #limit_train_batches=50,
            enable_checkpointing=True,
        )
        model.fit(train_dataloader, val_dataloader, **trainer_kwargs)
    except Exception as e:
        raise ValueError(f"Training failed to: {e}")

    # ----------------
    # Getting best model and validation loss
    # ----------------
    try:
        best_model, mae_metric = model.get_val_loss(
            val_dataloader,
            checkpoint_dir="checkpoints",
            model_name="best_model.ckpt", 
            raw_predictions = None,
        )
        print(f"\n\nMAE metric: {mae_metric}\n\n")
    except Exception as e:
        raise ValueError(f"Failed to perfom prediction to: {e}") from e
    # ----------------
    # Getting raw predictions on validation set and plotting them
    # ----------------
    if CONFIG_RNN["val"]:
        raw_predictions = best_model.predict(
            val_dataloader,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator=CONFIG_RNN["accelerator"]),
        )
        if raw_predictions is not None:
            try:
                print("\n\nPlotting results...\n\n")
                metadata_index = val_tsdataset_with_cov.x_to_index(raw_predictions.x)
                which_stations = metadata_index["Stazione_APPA"].reset_index(drop=True)
                for station in which_stations.unique():
                    idx = which_stations[which_stations == station].index[0] # select first available index 0 inside the metadata dataframe
                    best_model.plot_prediction(
                        raw_predictions.x,
                        raw_predictions.output,
                        idx=idx,
                        add_loss_to_title=True,
                    )
                    plt.suptitle(f"Series: {station}")
                    plt.show()
                plt.close("all")  # avoid piling up open figures
            except Exception as e:
                print(f"Failed to plot results :( to {e}")
            finally:
                plt.close("all")

    if CONFIG_RNN["test"]:
        try:
            predictions = best_model.predict(
                test_dataloader,
                mode="raw",
                return_x=True,
                trainer_kwargs=dict(accelerator=CONFIG_RNN["accelerator"]),
            )
            print("\n\nSuccessfully made predictions\n\n")
        except Exception as e:
            raise ValueError(f"Error during predictions to: {e}")
        if predictions is not None and CONFIG_RNN["sample_plot"]:
            try:
                metadata_index = test_tsdataset_with_cov.x_to_index(predictions.x)
                which_stations = metadata_index["Stazione_APPA"].reset_index(drop=True)
                for station in which_stations.unique():
                    idx = which_stations[which_stations == station].index[0]
                    best_model.plot_prediction(
                        predictions.x,
                        predictions.output,
                        idx=idx,
                        add_loss_to_title=True,
                    )
                    plt.suptitle(f"Series: {station}")
                    plt.show()
            except Exception as e:
                print(f"Failed to plot sample test windows: {e}")
            finally:
                plt.close("all")

        if predictions is not None and CONFIG_RNN["full_test_plot"]:
            try:
                full_predictions_df = LSTModel.build_full_length_prediction_frame(
                    predictions,
                    test_tsdataset_with_cov,
                    test_df_processed,
                )
                LSTModel.plot_full_length_predictions(full_predictions_df)
            except Exception as e:
                print(f"Failed to build comprehensive test plots: {e}")
            finally:
                plt.close("all")
