from divkgrad_1d_hd_analyt import Model as Model_divkgrad_1d_hd_analyt
from divkgrad_2d_hd_analyt import Model as Model_divkgrad_2d_hd_analyt
from divkgrad_3d_hd_analyt import Model as Model_divkgrad_3d_hd_analyt

def get_models():
  models = {}
  models['divkgrad_1d_hd_analyt'] = Model_divkgrad_1d_hd_analyt
  models['divkgrad_2d_hd_analyt'] = Model_divkgrad_2d_hd_analyt
  models['divkgrad_3d_hd_analyt'] = Model_divkgrad_3d_hd_analyt
  return models