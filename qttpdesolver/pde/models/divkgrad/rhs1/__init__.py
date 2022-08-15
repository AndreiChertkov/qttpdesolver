from divkgrad_1d_hd_rhs1 import Model as Model_divkgrad_1d_hd_rhs1
from divkgrad_2d_hd_rhs1 import Model as Model_divkgrad_2d_hd_rhs1
from divkgrad_3d_hd_rhs1 import Model as Model_divkgrad_3d_hd_rhs1

def get_models():
  models = {}
  models['divkgrad_1d_hd_rhs1'] = Model_divkgrad_1d_hd_rhs1
  models['divkgrad_2d_hd_rhs1'] = Model_divkgrad_2d_hd_rhs1
  models['divkgrad_3d_hd_rhs1'] = Model_divkgrad_3d_hd_rhs1
  return models