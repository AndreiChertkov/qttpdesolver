from analyt import get_models as gm1
from rhs1 import get_models as gm2
from msc import get_models as gm3

def get_models():
  models = {}
  for gm in [gm1, gm2, gm3]:
  	models.update(gm())
  return models