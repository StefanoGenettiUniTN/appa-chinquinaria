"""
RNN model implementation for pollutant forecasting. The subsequent implementation 
was created following closely the documentation of pytorch-forecasting tutorials
that can be found at the subsequent link: 
https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/building.html
"""
from typing import Dict, List, Tuple

try:
    import os
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
    from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
    print("Everything imported smoothly. Proceed to class declaration...\n")
except Exception as e:
    raise ImportError(f"Import failed, check out src. Error: {e}")


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
            dropout: float = 0.1,            
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
        # Encoder time steps cover:
        #   ..., t-2, t-1, t
        #
        # Decoder is tasked with predicting:
        #   t+1, t+2, ..., t+H
        #
        # Decoder input construction follows an autoregressive pattern:
        #
        # - At decoder step 0 (predicting t+1):
        #       use y_t  -> the last target value from the encoder
        #
        # - At decoder step 1 (predicting t+2):
        #       use y_{t+1}  -> the previous decoder target (ground truth during training)
        #
        # - At decoder step k (predicting t+k+1):
        #       use y_{t+k}  -> the decoder target from the previous step
        #
        # In summary:
        #   decoder_input[k] contains the target value from time (t + k - 1),
        #   with decoder_input[0] initialized using the last encoder target y_t.
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
        train_dataloader, 
        val_dataloader,
        checkpoint_dir="checkpoints",
        model_name="best_model.ckpt", 
        **trainer_kwargs,
    ) -> str:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.ckpt")
            if os.path.exists(checkpoint_path):
                print("Checkpoint found, loading model...")
                model = LSTModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
            else:
                print("No checkpoint found, training model...")
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(
                    model,
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
                else:
                    raise RuntimeError("Training finished but no best_model_path was found.")
        except Exception as e:
            raise RuntimeError(f"Full training with early stopping failed: {e}") from e

    def get_best_model_and_val_loss(
        val_dataloader,
        checkpoint_dir="checkpoints",
        model_name="best_model.ckpt", 
        raw_predictions = None,
    )->torch.Tensor:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.ckpt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found at {checkpoint_path}. Cannot perform prediction."
                )

            print("Loading checkpoint of best model and making prediction...")
            best_model = LSTModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
            predictions = best_model.predict(
                val_dataloader,
                trainer_kwargs=dict(accelerator="gpu"),
                return_y=True,
            )
            mae_metric = MAE()(predictions.output, predictions.y)
            return best_model, mae_metric
        except Exception as e:
            print(f"Failed to perfom prediction to: {e}")

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


def load_data() -> pd.DataFrame:
    file_id = "1iIOLm-jpBpZWl9kkKVxhFD1H3rrUgw1k"
    # Load the dataset from Google Drive
    df = pd.read_csv(f"https://drive.google.com/uc?id={file_id}")
    return df


def create_ts_dataset_with_covariates(
        dataset: pd.DataFrame,
        max_encoder_length = 60,
        max_prediction_length = 20
) -> TimeSeriesDataSet:
    try:
        dataset["data"] = pd.to_datetime(dataset["Data"], format="%Y-%m-%d")
        dataset.sort_values("Data", inplace=True)
        dataset["time_idx"] = dataset.groupby("Stazione_APPA").cumcount()
        dataset.columns = dataset.columns.str.replace('.', '_', regex=False)
    
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
        context_length = max_encoder_length
        prediction_length = max_prediction_length

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
                    "boundary_layer_height"
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
    except Exception as e:
        raise Exception(f"Failed to parse the dataset to ts object: {str(e)}")


def main():
    # ----------------
    # Data loading
    # ----------------
    try:
        print("Class correctly declared, init data loading...")
        dataset = load_data()
        print("Data loaded correctly, proced to ts object parsing of pd DataFrame...")
    except Exception as e:
        raise ValueError(f"Error in loading data: {e}") from e
    # ----------------
    # TimeSeriesDataSet creation
    # ----------------
    try:
        train_tsdataset_with_cov, val_tsdataset_with_cov = create_ts_dataset_with_covariates(dataset)
        print("Dataset parsed correctly to TimeSeriesDataSet Object, URRAY!")
    except Exception as e:
        raise ValueError(f"Something went wrong during TimeSeriesDataSet creation: {e}") from e
    # ----------------
    # Model from dataset
    # ----------------
    try:
        model_kwargs = dict(
            n_layers=4,
            hidden_size=20,
        )
        model = LSTModel.from_dataset(
            train_tsdataset_with_cov,
            model_kwargs,
        )
        print("Model instantiated correctly from the dataset structure")
    except Exception as e:
        raise ValueError(f"Something went wrong during model instantiation: {e}") from e
    # ----------------
    # Dataloaders
    # ----------------
    try:
        print("Switching data to dataloaders for training...")
        batch_size = 128
        train_dataloader = train_tsdataset_with_cov.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = val_tsdataset_with_cov.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        print("Dataloaders are ready.")
    except Exception as e:
        raise ValueError(f"to_dataloader failed: {e}") from e
    # ----------------
    # Getting best model and validation loss
    # ----------------
    try:
        best_model, mae_metric = model.get_best_model_and_val_loss(
            val_dataloader,
            checkpoint_dir="checkpoints",
            model_name="best_model.ckpt", 
            raw_predictions = None,
        )
        print(f"MAE metric: {mae_metric}")
    except Exception as e:
        raise ValueError(f"Failed to perfom prediction to: {e}") from e
    # ----------------
    # Getting raw predictions on validation set and plotting them
    # ----------------
    eval = True
    if eval:
        raw_predictions = best_model.predict(
            val_dataloader,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator="gpu"),
        )
        if raw_predictions is not None:
            try:
                print("Plotting results...")
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
            except Exception as e:
                print(f"Failed to plot results :( to {e}")


if __name__ == '__main__':
    main()