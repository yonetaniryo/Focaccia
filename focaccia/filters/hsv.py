from focaccia import Focaccia
from skimage.color import rgb2hsv, hsv2rgb
import optuna

class focacciaHsv(Focaccia):
  def __init__ (self, path: str, target_point: tuple[int, int], lambda_: float, inside: bool = True):
    super().__init__(path, target_point, lambda_, inside)
    self.switch(True, True, True)

  def apply (self, h_mul: float, s_mul: float, v_mul: float):
    hsv_img = rgb2hsv(self.img)
    applied_hsv_img = hsv_img.copy()
    applied_hsv_img[:, :, 0] = applied_hsv_img[:, :, 0] * h_mul
    applied_hsv_img[:, :, 1] = applied_hsv_img[:, :, 1] * s_mul
    applied_hsv_img[:, :, 2] = applied_hsv_img[:, :, 2] * v_mul
    applied_hsv_img[self.mask == int(self.inside)] = hsv_img[self.mask == int(self.inside)]
    return hsv2rgb(applied_hsv_img)
  
  def function (self, h_mul: float, s_mul: float, v_mul: float):
    output = self.apply(h_mul, s_mul, v_mul)
    return self.score(output)
  
  def objective (self, trial):
    h_mul = trial.suggest_uniform("h_mul", 0, 1) if self.h_switch else 1.0
    s_mul = trial.suggest_uniform("s_mul", 0, 1) if self.s_switch else 1.0
    v_mul = trial.suggest_uniform("v_mul", 0, 1) if self.v_switch else 1.0
    return self.function(h_mul, s_mul, v_mul)

  def optimize (self, n_trials: int):
    study = optuna.create_study(direction="minimize")
    study.optimize(self.objective, n_trials=n_trials)
    self.best_h_mul = study.best_params['h_mul'] if self.h_switch else 1.0
    self.best_s_mul = study.best_params['s_mul'] if self.s_switch else 1.0
    self.best_v_mul = study.best_params['v_mul'] if self.v_switch else 1.0
    self.applied = self.apply(self.best_h_mul, self.best_s_mul, self.best_v_mul)
    print(self.best_h_mul, self.best_s_mul, self.best_v_mul)
  
  def switch (self, h_switch: bool, s_switch: bool, v_switch: bool):
    self.h_switch = h_switch
    self.s_switch = s_switch
    self.v_switch = v_switch
