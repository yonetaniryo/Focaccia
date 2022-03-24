import cv2
import optuna

from ..focaccia import Focaccia


class focacciaEllipse(Focaccia):
    def __init__(
        self,
        path,
        target_point,
        lambda_,
        inside = True,
    ):
        super().__init__(path, target_point, lambda_, inside)
        self.center = target_point

    def apply(
        self,
        axes_x,
        axes_y,
        angle,
        color_b,
        color_g,
        color_r,
        thickness,
    ):
        bgr_image = self.toBgr()
        applied_image = bgr_image.copy()
        cv2.ellipse(
            applied_image,
            ((self.center[0], self.center[1]), (axes_x, axes_y), angle),
            (color_b, color_g, color_r),
            thickness=thickness,
        )
        return self.bgr2rgb(applied_image)

    def function(
        self,
        axes_x,
        axes_y,
        angle,
        color_b,
        color_g,
        color_r,
        thickness,
    ):
        output = self.apply(axes_x, axes_y, angle, color_b, color_g, color_r, thickness)
        return self.score(output)

    def objective(self, trial):
        axes_x = trial.suggest_uniform("axes_x", 1, 400)
        axes_y = trial.suggest_uniform("axes_y", 1, 400)
        angle = trial.suggest_uniform("angle", 0, 360)
        color_b = trial.suggest_uniform("color_b", 0, 255)
        color_g = trial.suggest_uniform("color_g", 0, 255)
        color_r = trial.suggest_uniform("color_r", 0, 255)
        thickness = trial.suggest_int("thickness", 1, 100)
        return self.function(
            axes_x, axes_y, angle, color_b, color_g, color_r, thickness
        )

    def optimize(self, n_trials):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        self.best_axes_x = study.best_params["axes_x"]
        self.best_axes_y = study.best_params["axes_y"]
        self.best_angle = study.best_params["angle"]
        self.best_color_b = study.best_params["color_b"]
        self.best_color_g = study.best_params["color_g"]
        self.best_color_r = study.best_params["color_r"]
        self.best_thickness = study.best_params["thickness"]
        self.applied = self.apply(
            self.best_axes_x,
            self.best_axes_y,
            self.best_angle,
            self.best_color_b,
            self.best_color_g,
            self.best_color_r,
            self.best_thickness,
        )
        print(
            self.best_axes_x,
            self.best_axes_y,
            self.best_angle,
            self.best_color_b,
            self.best_color_g,
            self.best_color_r,
            self.best_thickness,
        )
