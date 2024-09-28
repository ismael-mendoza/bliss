# Models

## How many epochs to use?

- **Autoencoder**: `10_000` epochs seems sufficient, in fact val_loss goes slightly up afterwards.
                *Learning_rate* = 1e-5 and *batch_size* = 128.

- **Deblender**: `10_000` epochs should be sufficient but remains to be tested, 5_000 already does well.
                *Learning_rate* = 1e-4 and *batch_size* = 128.
