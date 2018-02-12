from divkgrad_1d_hd_msc import Model as Model_divkgrad_1d_hd_msc
from divkgrad_2d_hd_msc import Model as Model_divkgrad_2d_hd_msc
from divkgrad_3d_hd_msc import Model as Model_divkgrad_3d_hd_msc

def get_models():
  models = {}
  models['divkgrad_1d_hd_msc'] = Model_divkgrad_1d_hd_msc
  models['divkgrad_2d_hd_msc'] = Model_divkgrad_2d_hd_msc
  models['divkgrad_3d_hd_msc'] = Model_divkgrad_3d_hd_msc
  return models