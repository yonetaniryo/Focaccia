import cv2
import optuna

from ..focaccia import Focaccia


class focacciaBlur(Focaccia):
    def __init__(
        self,
        path,
        target_point,
        lambda_,
        inside = True,
    ):
        super().__init__(path, target_point, lambda_, inside)

    def apply(self, kernel_size, sigma):
        bgr_image = self.toBgr()
        blured_image = bgr_image.copy()
        blured_image = cv2.GaussianBlur(blured_image, (kernel_size, kernel_size), sigma)
        blured_image[self.mask == 1] = bgr_image[self.mask == 1]
        return self.bgr2rgb(blured_image)

    def function(self, kernel_size, sigma):
        output = self.apply(kernel_size, sigma)
        return self.score(output)

    def objective(self, trial):
        kernel_size = trial.suggest_int("kernel_size", 1, 101, 2)
        sigma = trial.suggest_uniform("sigma", -30, 30)
        return self.function(kernel_size, sigma)

    def optimize(self, n_trials):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        self.best_kernel_size = study.best_params["kernel_size"]
        self.best_sigma = study.best_params["sigma"]
        self.applied = self.apply(self.best_kernel_size, self.best_sigma)
        print(self.best_kernel_size, self.best_sigma)
