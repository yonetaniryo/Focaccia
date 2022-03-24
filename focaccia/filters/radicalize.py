from focaccia import Focaccia
import optuna
import cv2
import numpy as np

class focacciaRadicalize(Focaccia):
  def __init__ (self, path: str, target_point: tuple[int, int], lambda_: float, inside: bool = True):
    super().__init__(path, target_point, lambda_, inside)
  
  def apply (self, k: float):
    bgr_image = self.toBgr()
    kernel = np.array([
      [-k / 9, -k / 9, -k / 9],
      [-k / 9, 1 + 8 * k / 9, k / 9],
      [-k / 9, -k / 9, -k / 9]
    ], np.float32)
    applied_image = bgr_image.copy()
    applied_image = cv2.filter2D(applied_image, -1, kernel)
    applied_image[self.mask == int(self.inside)] = bgr_image[self.mask == int(self.inside)]
    return self.bgr2rgb(applied_image)
  
  def function (self, k: float):
    output = self.apply(k)
    return self.score(output)
  
  def objective (self, trial):
    k = trial.suggest_uniform("k", -100, 100)
    return self.function(k)
  
  def optimize (self, n_trials: int):
    study = optuna.create_study(direction="minimize")
    study.optimize(self.objective, n_trials=n_trials)
    self.best_k = study.best_params['k']
    self.applied = self.apply(self.best_k)
    print(self.best_k)
