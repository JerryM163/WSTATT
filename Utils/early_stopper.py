
import val


class EarlyStopper:
    def __init__(self, patience=8, min_delta=0.001, warmup_epochs=10):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs

        self.counter = 0
        self.best_loss = float('inf')
        self.epoch = 0

    def early_stop(self, validation_loss):
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
            return False

        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        stop = self.counter >= self.patience

        if stop:
            print(
                f"Epoch {self.epoch}: "
                f"val={validation_loss:.5f}, "
                f"best={self.best_loss:.5f}, "
                f"patience={self.counter}/{self.patience}"
            )

        return stop